#!/usr/bin/env python3
"""gen_moe_f16_act.py SRC.loom DST.loom : write f16 directly from prefill MoE gate-up GEMMs, eliminating
g_row_scale_f16_tiles, the f32 act_sorted scratch buffer (~113 MB at t=2048), and the s_rows unscaling
epilogue in g_moe_gdn_*.

Mathematical bit-identity proof:
  In all 48 Inkling-2-Mini layers (NVFP4 layers 0-19 and Q8_0 layers 20-23), the MoE gate-up activation is
    act = (4.0 * tanh(g / 4.0) * sigmoid(g)) * (25.0 * tanh(u / 25.0))
  which is strictly bounded by |act| <= 4.0 * 1.0 * 25.0 = 100.0 < 2^7.
  g_row_scale_f16_tiles computes e = clamp(unbiased_exp(max |act|) - 13, 0, 100). Since unbiased_exp <= 6,
  raw_e <= -7, so e == 0, scale == 1.0, and inverse == 1.0 for every row of every layer.
  Staging the 64x64 f32 result tile through the already-allocated (and free) w_stage / w2_stage LDS buffers
  (32x68xf32 each = 8704 B <= 9216 B, zero extra LDS), converting with vector.fptrunc to vector<16xf16>,
  and issuing coalesced 32-byte stores directly into act16 is bit-identical while eliminating 48 kernel
  launches and ~11.3 GB of global memory traffic per 2048-token prefill chunk.
"""
import re
import sys

GGU_KERNELS = (
    "g_moe_ggu_nvfp4_fused_sc_k2304",
    "g_moe_ggu_nvfp4_fused_sc_k2304_rdna4",
    "g_moe_ggu_q8_0_fused_k2304",
    "g_moe_ggu_q8_0_fused_k2304_rdna4",
)

GDN_KERNELS = (
    "g_moe_gdn_nvfp4_k1152",
    "g_moe_gdn_nvfp4_k1152_rdna4",
    "g_moe_gdn_q8_0_k1152",
    "g_moe_gdn_q8_0_k1152_rdna4",
)

OLD_GGU_YVIEW = """  // 64x64 output tile with the full row stride n.
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x64xf32, %y_layout>"""

NEW_GGU_YVIEW = """  // Direct f16 output into act16 (|act| <= 100.0 fits f16 with unit scale).
  %y16_view = buffer.view %y[%origin] : buffer -> view<[%rows]x[%n]xf16>"""

OLD_GGU_STORE = """  vector.fragment.store<result> %act0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %act1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>"""

NEW_GGU_STORE = """  // Stage 64x64 f32 result tile through free w_stage (rows 0..31) and w2_stage (rows 32..63),
  // padded to 68 floats/row (32*68*4 = 8704 B <= 9216 B) to avoid LDS bank conflicts, then fptrunc -> f16.
  %f16_out_lo = buffer.view %w_stage[%origin] : buffer -> view<32x68xf32>
  %f16_out_hi = buffer.view %w2_stage[%origin] : buffer -> view<32x68xf32>
  %f16_half = index.cmp ult, %wave_row_group, %c2 : index
  %f16_wrow_m = index.rem %wave_row_group, %c2 : index
  %f16_wrow_r = index.mul %f16_wrow_m, %c16 : index
  %f16_wrow = index.assume %f16_wrow_r [range(%f16_wrow_r, 0, 16)] : index
  scf.if %f16_half {
    vector.fragment.store<result> %act0, %f16_out_lo[%f16_wrow, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>
    vector.fragment.store<result> %act1, %f16_out_lo[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>
  } else {
    vector.fragment.store<result> %act0, %f16_out_hi[%f16_wrow, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>
    vector.fragment.store<result> %act1, %f16_out_hi[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %f16_st_lo = index.cmp ult, %stage_row, %c32 : index
  %f16_v32 = scf.if %f16_st_lo -> (vector<16xf32>) {
    %f16_sr_lo = index.assume %stage_row [range(%stage_row, 0, 31)] : index
    %f16_ld0 = vector.load %f16_out_lo[%f16_sr_lo, %stage_col] : view<32x68xf32> -> vector<16xf32>
    scf.yield %f16_ld0 : vector<16xf32>
  } else {
    %f16_sr_h0 = index.sub %stage_row, %c32 : index
    %f16_sr_hi = index.assume %f16_sr_h0 [range(%f16_sr_h0, 0, 31)] : index
    %f16_ld1 = vector.load %f16_out_hi[%f16_sr_hi, %stage_col] : view<32x68xf32> -> vector<16xf32>
    scf.yield %f16_ld1 : vector<16xf32>
  }
  %f16_v16 = vector.fptrunc %f16_v32 : vector<16xf32> to vector<16xf16>
  %f16_gr0 = index.add %row_origin, %stage_row : index
  %f16_gr = index.assume %f16_gr0 [lt(%f16_gr0, %rows)] : index
  %f16_gc0 = index.add %out_origin, %stage_col : index
  %f16_gc = index.assume %f16_gc0 [lt(%f16_gc0, %n)] : index
  vector.store %f16_v16, %y16_view[%f16_gr, %f16_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>"""

OLD_GDN_DEAD_SCALES = """  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>
  %x_scale = view.load %s_view[%bounded_x_row, %c0] : view<[%xrows]x2xf32> -> f32
  %x_scales = vector.splat %x_scale : vector<16xf32>\n"""

OLD_GDN_EPILOGUE = """  %inv_bytes = index.constant 4096 : offset
  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %xrows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%xrows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>
  %a0 = vector.mulf %a0_raw, %inv_frag : vector<8xf32>
  %a1 = vector.mulf %a1_raw, %inv_frag : vector<8xf32>
  vector.fragment.store<result> %a0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>"""

NEW_GDN_EPILOGUE = """  // act16 is written directly at unit scale (|act| <= 100.0), so inverse == 1.0 always.
  vector.fragment.store<result> %a0_raw, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1_raw, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>"""


def find_kernel_span(text, export_name):
    marker = f'export("{export_name}") @{export_name}('
    s = text.find(marker)
    if s < 0:
        raise ValueError(f"kernel {export_name} not found")
    line_start = text.rfind("\n", 0, s) + 1
    end_marker = "\n  kernel.return\n}\n"
    e = text.find(end_marker, s)
    if e < 0:
        raise ValueError(f"end of kernel {export_name} not found")
    return line_start, e + len(end_marker)


def patch(text):
    if "%f16_out_lo = buffer.view %w_stage" in text:
        raise ValueError("already patched by gen_moe_f16_act.py")

    # 1. Patch all 4 g_moe_ggu_* kernels to write f16 directly into act16.
    for name in GGU_KERNELS:
        lo, hi = find_kernel_span(text, name)
        body = text[lo:hi]
        if OLD_GGU_YVIEW not in body or OLD_GGU_STORE not in body:
            raise ValueError(f"{name}: expected y_view/store pattern not found")
        body = body.replace(OLD_GGU_YVIEW, NEW_GGU_YVIEW, 1)
        body = body.replace(OLD_GGU_STORE, NEW_GGU_STORE, 1)
        text = text[:lo] + body + text[hi:]

    # 2. Patch all 4 g_moe_gdn_* kernels to drop dead %scales reads and inverse-scale epilogue.
    for name in GDN_KERNELS:
        lo, hi = find_kernel_span(text, name)
        body = text[lo:hi]
        if OLD_GDN_DEAD_SCALES not in body or OLD_GDN_EPILOGUE not in body:
            raise ValueError(f"{name}: expected scales/epilogue pattern not found")
        body = body.replace(OLD_GDN_DEAD_SCALES, "", 1)
        body = body.replace(OLD_GDN_EPILOGUE, NEW_GDN_EPILOGUE, 1)
        text = text[:lo] + body + text[hi:]

    # 3. Patch llm_advance and llm_verify:
    #    - remove %act_sorted and %s_rows allocations (keep %fe4/%b8 since other allocs may use them)
    #    - route sel10 / sel13 output directly to %act16
    #    - drop all 96 g_row_scale_f16_tiles launches
    #    - pass %act16 for the now-unused %scales arg of sel11 / sel14 (keeps template signatures untouched)
    n_alloc_as, n_alloc_sr = 0, 0
    n_ggu_nvfp4, n_ggu_q8, n_rscale, n_gdn_nvfp4, n_gdn_q8 = 0, 0, 0, 0, 0

    out_lines = []
    for ln in text.splitlines(keepends=True):
        if "%as_size = index.scale %max_rows, %fe4" in ln or "%act_sorted = buffer.alloca<global>" in ln:
            n_alloc_as += 1
            continue
        if "%sr_size = index.scale %max_rows, %b8" in ln or "%s_rows = buffer.alloca<global>" in ln:
            n_alloc_sr += 1
            continue
        if "kernel.launch @g_row_scale_f16_tiles[" in ln:
            n_rscale += 1
            continue
        if "template.apply<@sel10_g_moe_ggu_nvfp4_fused_sc_k2304>" in ln and ", %act_sorted)" in ln:
            ln = ln.replace(", %act_sorted)", ", %act16)")
            n_ggu_nvfp4 += 1
        elif "template.apply<@sel13_g_moe_ggu_q8_0_fused_k2304>" in ln and ", %act_sorted)" in ln:
            ln = ln.replace(", %act_sorted)", ", %act16)")
            n_ggu_q8 += 1
        elif "template.apply<@sel11_g_moe_gdn_nvfp4_k1152>" in ln and ", %act16, %s_rows," in ln:
            ln = ln.replace(", %act16, %s_rows,", ", %act16, %act16,")
            n_gdn_nvfp4 += 1
        elif "template.apply<@sel14_g_moe_gdn_q8_0_k1152>" in ln and ", %act16, %s_rows," in ln:
            ln = ln.replace(", %act16, %s_rows,", ", %act16, %act16,")
            n_gdn_q8 += 1
        out_lines.append(ln)

    res = "".join(out_lines)
    assert n_alloc_as == 4, f"expected 4 act_sorted alloc lines, got {n_alloc_as}"
    assert n_alloc_sr == 4, f"expected 4 s_rows alloc lines, got {n_alloc_sr}"
    assert n_rscale == 96, f"expected 96 g_row_scale_f16_tiles launches, got {n_rscale}"
    assert n_ggu_nvfp4 == 88 and n_ggu_q8 == 8, f"unexpected ggu counts: {n_ggu_nvfp4}, {n_ggu_q8}"
    assert n_gdn_nvfp4 == 88 and n_gdn_q8 == 8, f"unexpected gdn counts: {n_gdn_nvfp4}, {n_gdn_q8}"
    assert "%act_sorted" not in res, "stray %act_sorted reference remains"
    assert "%s_rows" not in res, "stray %s_rows reference remains"
    return res


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_moe_f16_act.py SRC.loom DST.loom")
    src, dst = sys.argv[1], sys.argv[2]
    patched = patch(open(src).read())
    open(dst, "w").write(patched)
    print("patched 4 g_moe_ggu_* -> direct f16 act16, 4 g_moe_gdn_* -> unit scale, removed 96 g_row_scale_f16_tiles + act_sorted/s_rows", file=sys.stderr)


if __name__ == "__main__":
    main()
