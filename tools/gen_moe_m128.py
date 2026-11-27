#!/usr/bin/env python3
"""Rewrite MoE grouped prefill bucketing and all 4 RDNA4 grouped GEMMs from M=64 to M=128 tiles:

  - g_moe_bucket + host %c64m -> 128-row expert tiles (cuts weight DRAM/L2 reads & NVFP4
    dequant VALU per FLOP by up to 2x at prefill_capacity=2048, ~128 tokens/expert mean)
  - g_moe_ggu_nvfp4_fused_sc_k2304_rdna4 : 128x64x64 (36.9 KB LDS, 37.5% occ, 32 WMMAs/step,
    reusing every RHS fragment across rows 0..63 and 64..127)
  - g_moe_gdn_nvfp4_k1152_rdna4          : 128x64x64 (27.6 KB LDS, 50% occ, 16 WMMAs/step)
                                        or 128x128x64 (36.9 KB LDS, 37.5% occ, 32 WMMAs/step, 128-bit loads)
  - g_moe_ggu_q8_0_fused_k2304_rdna4     : 128x64x64 (layers 20..23)
  - g_moe_gdn_q8_0_k1152_rdna4           : 128x64x64 or 128x128x64 (layers 20..23)

All kernels preserve register-level software prefetch pipelining, exact K=64 accumulation order,
and high-byte LUT f16 NVFP4 dequantization (100% bit-identical). Tiles with <=64 live tokens
skip all upper-half (rows 64..127) X loads, WMMAs, and epilogue work via uniform SGPR guards.

usage: gen_moe_m128.py IN.loom OUT.loom [--gdn-n 64|128]
"""
import argparse


def kernel_span(src: str, name: str, target_rdna4: bool = True):
    prefix = f'kernel.def target(@rdna4) export("{name}")' if target_rdna4 else f'kernel.def export("{name}")'
    start = src.index(prefix)
    end = src.index("\n  kernel.return\n}\n", start) + len("\n  kernel.return\n}\n")
    return start, end


def _nvfp4_consts(indent: str = "  ") -> list[str]:
    ind = indent
    return [
        f"{ind}%nv_mask = vector.splati 15 : vector<8xi8>",
        f"{ind}%nv_four = vector.splati 4 : vector<8xi8>",
        f"{ind}%nv_z8 = vector.splati 0 : vector<8xi8>",
        f"{ind}%nv_k0 = scalar.constant 0 : i8",
        f"{ind}%nv_k1 = scalar.constant 60 : i8",
        f"{ind}%nv_k2 = scalar.constant 64 : i8",
        f"{ind}%nv_k3 = scalar.constant 66 : i8",
        f"{ind}%nv_k4 = scalar.constant 68 : i8",
        f"{ind}%nv_k5 = scalar.constant 70 : i8",
        f"{ind}%nv_k6 = scalar.constant 72 : i8",
        f"{ind}%nv_k7 = scalar.constant 74 : i8",
        f"{ind}%nv_k9 = scalar.constant -68 : i8",
        f"{ind}%nv_k10 = scalar.constant -64 : i8",
        f"{ind}%nv_k11 = scalar.constant -62 : i8",
        f"{ind}%nv_k12 = scalar.constant -60 : i8",
        f"{ind}%nv_k13 = scalar.constant -58 : i8",
        f"{ind}%nv_k14 = scalar.constant -56 : i8",
        f"{ind}%nv_k15 = scalar.constant -54 : i8",
        f"{ind}%nv_cb = vector.from_elements %nv_k0, %nv_k1, %nv_k2, %nv_k3, %nv_k4, %nv_k5, %nv_k6, %nv_k7, %nv_k0, %nv_k9, %nv_k10, %nv_k11, %nv_k12, %nv_k13, %nv_k14, %nv_k15 : vector<16xi8>",
    ]


def _nvfp4_decode16(p: str, b_var: str, q8_var: str, out_var: str, indent: str = "    ") -> list[str]:
    ind = indent
    return [
        f"{ind}%{p}_d0 = func.call @g_ue4m3_to_f16({b_var}) : (i8) -> (f16)",
        f"{ind}%{p}_lo = vector.andi {q8_var}, %nv_mask : vector<8xi8>",
        f"{ind}%{p}_hi = vector.shrui {q8_var}, %nv_four : vector<8xi8>",
        f"{ind}%{p}_lohi = vector.table.lookup %nv_cb[%{p}_lo] : vector<16xi8>, vector<8xi8> -> vector<8xi8>",
        f"{ind}%{p}_hihi = vector.table.lookup %nv_cb[%{p}_hi] : vector<16xi8>, vector<8xi8> -> vector<8xi8>",
        f"{ind}%{p}_lo16c = vector.concat<0> %nv_z8, %{p}_lohi : vector<8xi8>, vector<8xi8> -> vector<16xi8>",
        f"{ind}%{p}_hi16c = vector.concat<0> %nv_z8, %{p}_hihi : vector<8xi8>, vector<8xi8> -> vector<16xi8>",
        f"{ind}%{p}_lo16 = vector.shuffle<[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]> %{p}_lo16c : vector<16xi8>",
        f"{ind}%{p}_hi16 = vector.shuffle<[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]> %{p}_hi16c : vector<16xi8>",
        f"{ind}%{p}_loh = vector.bitcast %{p}_lo16 : vector<16xi8> to vector<8xf16>",
        f"{ind}%{p}_hih = vector.bitcast %{p}_hi16 : vector<16xi8> to vector<8xf16>",
        f"{ind}%{p}_ch = vector.concat<0> %{p}_loh, %{p}_hih : vector<8xf16>, vector<8xf16> -> vector<16xf16>",
        f"{ind}%{p}_dv = vector.splat %{p}_d0 : vector<16xf16>",
        f"{ind}{out_var} = vector.mulf %{p}_ch, %{p}_dv : vector<16xf16>",
    ]


def _act_block(p: str, g_var: str, u_var: str, out_var: str, ind: str = "  ") -> list[str]:
    return [
        f"{ind}%{p}_g2 = vector.mulf {g_var}, %oa_invb2 : vector<8xf32>",
        f"{ind}%{p}_gs2 = vector.logisticf<afn> %{p}_g2 : vector<8xf32>",
        f"{ind}%{p}_gt = vector.mulf %{p}_gs2, %oa_two : vector<8xf32>",
        f"{ind}%{p}_gtanh = vector.subf %{p}_gt, %oa_one : vector<8xf32>",
        f"{ind}%{p}_gb = vector.mulf %oa_beta, %{p}_gtanh : vector<8xf32>",
        f"{ind}%{p}_gsig = vector.logisticf<afn> {g_var} : vector<8xf32>",
        f"{ind}%{p}_gact = vector.mulf %{p}_gb, %{p}_gsig : vector<8xf32>",
        f"{ind}%{p}_u2 = vector.mulf {u_var}, %oa_invlb2 : vector<8xf32>",
        f"{ind}%{p}_us2 = vector.logisticf<afn> %{p}_u2 : vector<8xf32>",
        f"{ind}%{p}_ut = vector.mulf %{p}_us2, %oa_two : vector<8xf32>",
        f"{ind}%{p}_utanh = vector.subf %{p}_ut, %oa_one : vector<8xf32>",
        f"{ind}%{p}_uact = vector.mulf %oa_lb, %{p}_utanh : vector<8xf32>",
        f"{ind}{out_var} = vector.mulf %{p}_gact, %{p}_uact : vector<8xf32>",
    ]


def patch_bucket_and_host(src: str) -> str:
    s, e = kernel_span(src, "g_moe_bucket", target_rdna4=False)
    b = src[s:e]
    for old, new in [
        ("  %cT = index.constant 64 : index\n", "  %cT = index.constant 128 : index\n"),
        ("  %cshift = scalar.constant 6 : i32\n", "  %cshift = scalar.constant 7 : i32\n"),
        ("  %cTm1 = scalar.constant 63 : i32\n", "  %cTm1 = scalar.constant 127 : i32\n"),
    ]:
        assert b.count(old) == 1, f"expected 1 occurrence of {old!r} in g_moe_bucket, got {b.count(old)}"
        b = b.replace(old, new, 1)
    out = src[:s] + b + src[e:]

    old_host = "  %c64m = index.constant 64 : index\n"
    new_host = "  %c64m = index.constant 128 : index\n"
    n_host = out.count(old_host)
    assert n_host == 2, f"expected 2 occurrences of %c64m, got {n_host}"
    out = out.replace(old_host, new_host)
    return out


def emit_ggu_nvfp4_m128() -> str:
    """128x64x64 NVFP4 fused gate+up+act -> direct f16 act16, software-pipelined."""
    L = [
        'kernel.def target(@rdna4) export("g_moe_ggu_nvfp4_fused_sc_k2304_rdna4") @g_moe_ggu_nvfp4_fused_sc_k2304_rdna4(%rows: index, %n: index) {',
        "  %c1 = index.constant 1 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        "  %c256 = index.constant 256 : index",
        "  %row_tiles = index.div %rows, %c128 : index",
        "  %out_tiles = index.div %n, %c64 : index",
        "  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index",
        "} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, %w: buffer, %w2: buffer, %gus: buffer, %y: buffer) {",
        "  // MoE grouped GEMM M=128 (128x64x64 NVFP4 gate+up -> f16 act16, pipelined, RHS reuse).",
        "  %rows = index.assume %rows_arg [range(%rows_arg, 128, 131072), multiple_of(%rows_arg, 128)] : index",
        "  %k = index.assume %k_arg [range(%k_arg, 64, 65536), multiple_of(%k_arg, 64)] : index",
        "  %n = index.assume %n_arg [range(%n_arg, 64, 1152), multiple_of(%n_arg, 64)] : index",
        "  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index",
        "  %origin = index.constant 0 : offset",
        "  %one_o = index.constant 1 : offset",
        "  %four_o = index.constant 4 : offset",
        "  %eight_o = index.constant 8 : offset",
        "  %block_bytes = index.constant 36 : offset",
        "  %c0 = index.constant 0 : index",
        "  %c1 = index.constant 1 : index",
        "  %c2 = index.constant 2 : index",
        "  %c4 = index.constant 4 : index",
        "  %c8 = index.constant 8 : index",
        "  %c16 = index.constant 16 : index",
        "  %c32 = index.constant 32 : index",
        "  %c48 = index.constant 48 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        "  %c0_i32 = scalar.constant 0 : i32",
        "  %cEm1 = scalar.constant 127 : i32",
        "  %row_tile_count = index.div %rows, %c128 : index",
        "  %out_tile_count = index.div %n, %c64 : index",
        "  %row_group_id = kernel.workgroup.id<y> : index",
        "  %out_group_id = kernel.workgroup.id<x> : index",
        "  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index",
        "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index",
        "  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>",
        "  %tile_expert_v = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32",
        "  %tile_expert = kernel.subgroup.broadcast.first %tile_expert_v : i32",
        "  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32",
        "  scf.if %tile_used {",
        "  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32",
        "  %e_b = scalar.assume %e_lo [range(%e_lo, 0, 127)] : i32",
        "  %expert = index.cast %e_b : i32 to index",
        "  %expert_bytes = index.constant 2985984 : offset",
        "  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset",
        "  %workitem_id = kernel.workitem.id<x> : index",
        "  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index",
        "  %subgroup = kernel.subgroup.id : index",
        "  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index",
        "  %wave_row_group = index.div %bounded_subgroup, %c2 : index",
        "  %wave_out_group = index.rem %bounded_subgroup, %c2 : index",
        "  %wave_row_lo_v = index.mul %wave_row_group, %c16 : index",
        "  %wave_row_lo = index.assume %wave_row_lo_v [range(%wave_row_lo_v, 0, 48)] : index",
        "  %wave_row_hi_v = index.add %wave_row_lo, %c64 : index",
        "  %wave_row_hi = index.assume %wave_row_hi_v [range(%wave_row_hi_v, 64, 112)] : index",
        "  %wave_out_v = index.mul %wave_out_group, %c32 : index",
        "  %wave_out0 = index.assume %wave_out_v [range(%wave_out_v, 0, 32)] : index",
        "  %wave_out1 = index.add %wave_out0, %c16 : index",
        "  %row_origin = index.mul %row_group, %c128 : index",
        "  %out_origin = index.mul %out_group, %c64 : index",
        "  %blocks = index.div %k, %c64 : index",
        "  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset",
        "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>",
        "  %y16_view = buffer.view %y[%origin] : buffer -> view<[%rows]x[%n]xf16>",
        "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>",
        "  %x_stage_bytes = index.constant 18432 : offset",
        "  %w_stage_bytes = index.constant 9216 : offset",
        "  %x_stage = buffer.alloca<workgroup> align(16) %x_stage_bytes : buffer",
        "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        "  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>",
        "  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        "  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x64xf16, %w_stage_view_tl>",
        "  %w2_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        "  %w2_stage_view = buffer.view %w2_stage[%origin] : buffer -> view<64x72xf16>",
        "  %w2_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        "  %w2_stage_view_t = buffer.view %w2_stage[%origin] : buffer -> view<72x64xf16, %w2_stage_view_tl>",
        "  %wave_rows32_v = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32",
        "  %wave_rows32 = kernel.subgroup.broadcast.first %wave_rows32_v : i32",
        "  %wave_rows_b = scalar.assume %wave_rows32 [range(%wave_rows32, 0, 128)] : i32",
        "  %wave_rows = index.cast %wave_rows_b : i32 to index",
        "  %has_hi64 = index.cmp ugt, %wave_rows, %c64 : index",
        "  %wave_live_lo = index.cmp ult, %wave_row_lo, %wave_rows : index",
        "  %wave_live_hi = index.cmp ult, %wave_row_hi, %wave_rows : index",
        "  // X staging: 256 threads -> lo rows 0..63 and hi rows 64..127",
        "  %x_stage_row_lo = index.div %workitem, %c4 : index",
        "  %x_stage_row_hi = index.add %x_stage_row_lo, %c64 : index",
        "  %x_stage_q = index.rem %workitem, %c4 : index",
        "  %x_stage_col = index.mul %x_stage_q, %c16 : index",
        "  %gx_rm_rows = index.mul %row_tile_count, %c128 : index",
        "  %gx_rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%gx_rm_rows]xi32>",
        "  %x_row_raw_lo = index.add %row_origin, %x_stage_row_lo : index",
        "  %gx_prow_lo = index.assume %x_row_raw_lo [lt(%x_row_raw_lo, %gx_rm_rows)] : index",
        "  %gx_a32_lo = view.load %gx_rm_view[%gx_prow_lo] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_lo = scalar.cmpi sge, %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ac_lo = scalar.maxsi %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ab_lo = scalar.assume %gx_ac_lo [range(%gx_ac_lo, 0, 1048575)] : i32",
        "  %gx_a_lo = index.cast %gx_ab_lo : i32 to index",
        "  %gx_t_lo = index.div %gx_a_lo, %c8 : index",
        "  %bounded_x_row_lo = index.assume %gx_t_lo [lt(%gx_t_lo, %xrows)] : index",
        "  %x_row_raw_hi = index.add %row_origin, %x_stage_row_hi : index",
        "  %gx_prow_hi = index.assume %x_row_raw_hi [lt(%x_row_raw_hi, %gx_rm_rows)] : index",
        "  %gx_a32_hi = view.load %gx_rm_view[%gx_prow_hi] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_hi = scalar.cmpi sge, %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ac_hi = scalar.maxsi %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ab_hi = scalar.assume %gx_ac_hi [range(%gx_ac_hi, 0, 1048575)] : i32",
        "  %gx_a_hi = index.cast %gx_ab_hi : i32 to index",
        "  %gx_t_hi = index.div %gx_a_hi, %c8 : index",
        "  %bounded_x_row_hi = index.assume %gx_t_hi [lt(%gx_t_hi, %xrows)] : index",
        "  // W staging (N=64): 4 threads per 36-byte block",
        "  %up_bytes = index.constant 1492992 : offset",
        "  %w_stage_row = index.div %workitem, %c4 : index",
        "  %w_stage_q = index.rem %workitem, %c4 : index",
        "  %w_stage_col = index.mul %w_stage_q, %c16 : index",
        "  %w_out_row = index.add %out_origin, %w_stage_row : index",
        "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
        "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
        "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
        "  %w_d_add = index.scale %w_stage_q, %one_o : index, offset -> offset",
        "  %g_d_base = index.add %w_row_origin, %w_d_add : offset",
        "  %w_q_add = index.scale %w_stage_q, %eight_o : index, offset -> offset",
        "  %w_q_base0 = index.add %w_row_origin, %four_o : offset",
        "  %g_q_base = index.add %w_q_base0, %w_q_add : offset",
        "  %u_d_base = index.add %g_d_base, %up_bytes : offset",
        "  %u_q_base = index.add %g_q_base, %up_bytes : offset",
        "  %zero8 = vector.splatf 0.0 : vector<8xf32>",
        "  %zero16h = vector.splatf 0.0 : vector<16xf16>",
        "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>",
        "  %k_steps = index.div %k, %c64 : index",
        "  %pf_last = index.sub %k_steps, %c1 : index",
    ]
    L += _nvfp4_consts("  ")
    L += [
        "  %x_lo_p = vector.load %x_view[%bounded_x_row_lo, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "  %x_hi_p = scf.if %has_hi64 -> (vector<16xf16>) {",
        "    %xhp = vector.load %x_view[%bounded_x_row_hi, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    scf.yield %xhp : vector<16xf16>",
        "  } else {",
        "    scf.yield %zero16h : vector<16xf16>",
        "  }",
        "  %gd_view_p = buffer.view %w[%g_d_base] : buffer -> view<1xi8>",
        "  %gd_p = view.load %gd_view_p[0] : view<1xi8> -> i8",
        "  %gq_view_p = buffer.view %w[%g_q_base] : buffer -> view<8xi8>",
        "  %gq_p = vector.load %gq_view_p[0] : view<8xi8> -> vector<8xi8>",
        "  %ud_view_p = buffer.view %w2[%u_d_base] : buffer -> view<1xi8>",
        "  %ud_p = view.load %ud_view_p[0] : view<1xi8> -> i8",
        "  %uq_view_p = buffer.view %w2[%u_q_base] : buffer -> view<8xi8>",
        "  %uq_p = vector.load %uq_view_p[0] : view<8xi8> -> vector<8xi8>",
    ]
    acc_names = ["g0l", "g1l", "u0l", "u1l", "g0h", "g1h", "u0h", "u1h"]
    acc_outs = ", ".join(f"%{a}_raw" for a in acc_names)
    acc_inits = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in acc_names)
    acc_types = ", ".join(["vector<8xf32>"] * 8)
    L += [
        f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3, %pf_u4, %pf_u5 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_gd = %gd_p : i8, %cr_gq = %gq_p : vector<8xi8>, %cr_ud = %ud_p : i8, %cr_uq = %uq_p : vector<8xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, i8, vector<8xi8>, i8, vector<8xi8>) {{",
        "    %x_lo_s = scf.select %gx_valid_lo, %cr_xlo, %zero16h : vector<16xf16>",
        "    vector.store %x_lo_s, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    scf.if %has_hi64 {",
        "      %x_hi_s = scf.select %gx_valid_hi, %cr_xhi, %zero16h : vector<16xf16>",
        "      vector.store %x_hi_s, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    }",
    ]
    L += _nvfp4_decode16("dg", "%cr_gd", "%cr_gq", "%dg_v", "    ")
    L += ["    vector.store %dg_v, %w_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>"]
    L += _nvfp4_decode16("du", "%cr_ud", "%cr_uq", "%du_v", "    ")
    L += [
        "    vector.store %du_v, %w2_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %pf_s1 = index.add %step, %c1 : index",
        "    %pf_m = index.min %pf_s1, %pf_last : index",
        "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
        "    %k_next = index.mul %pf_next, %c64 : index",
        "    %x_col_n = index.add %k_next, %x_stage_col : index",
        "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
        "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "      scf.yield %xhn : vector<16xf16>",
        "    } else {",
        "      scf.yield %zero16h : vector<16xf16>",
        "    }",
        "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
        "    %gd_off_n = index.add %g_d_base, %step_off_n : offset",
        "    %gd_view_n = buffer.view %w[%gd_off_n] : buffer -> view<1xi8>",
        "    %gd_n = view.load %gd_view_n[0] : view<1xi8> -> i8",
        "    %gq_off_n = index.add %g_q_base, %step_off_n : offset",
        "    %gq_view_n = buffer.view %w[%gq_off_n] : buffer -> view<8xi8>",
        "    %gq_n = vector.load %gq_view_n[0] : view<8xi8> -> vector<8xi8>",
        "    %ud_off_n = index.add %u_d_base, %step_off_n : offset",
        "    %ud_view_n = buffer.view %w2[%ud_off_n] : buffer -> view<1xi8>",
        "    %ud_n = view.load %ud_view_n[0] : view<1xi8> -> i8",
        "    %uq_off_n = index.add %u_q_base, %step_off_n : offset",
        "    %uq_view_n = buffer.view %w2[%uq_off_n] : buffer -> view<8xi8>",
        "    %uq_n = vector.load %uq_view_n[0] : view<8xi8> -> vector<8xi8>",
    ]
    live_outs = ", ".join(f"%l_{a}" for a in acc_names)
    L += [f"    {live_outs} = scf.if %wave_live_hi -> ({acc_types}) {{"]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"      %ll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %lh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %rg0_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"      %rg1_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"      %ru0_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
            f"      %ru1_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
        ]
        for nm, lh, rh in [
            ("g0l", f"%ll_{ks}", f"%rg0_{ks}"), ("g0h", f"%lh_{ks}", f"%rg0_{ks}"),
            ("g1l", f"%ll_{ks}", f"%rg1_{ks}"), ("g1h", f"%lh_{ks}", f"%rg1_{ks}"),
            ("u0l", f"%ll_{ks}", f"%ru0_{ks}"), ("u0h", f"%lh_{ks}", f"%ru0_{ks}"),
            ("u1l", f"%ll_{ks}", f"%ru1_{ks}"), ("u1h", f"%lh_{ks}", f"%ru1_{ks}"),
        ]:
            prev = f"%{nm}_acc" if ks == 0 else f"%f_{nm}_{ks-1}"
            L.append(f"      %f_{nm}_{ks} = vector.mma {lh}, {rh}, {prev} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "      scf.yield " + ", ".join(f"%f_{a}_3" for a in acc_names) + f" : {acc_types}",
        "    } else {",
        f"      %lo_g0, %lo_g1, %lo_u0, %lo_u1 = scf.if %wave_live_lo -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {{",
    ]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"        %sl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"        %srg0_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"        %srg1_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"        %sru0_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
            f"        %sru1_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
        ]
        for nm, rh in [("g0l", f"%srg0_{ks}"), ("g1l", f"%srg1_{ks}"), ("u0l", f"%sru0_{ks}"), ("u1l", f"%sru1_{ks}")]:
            prev = f"%{nm}_acc" if ks == 0 else f"%s_{nm}_{ks-1}"
            L.append(f"        %s_{nm}_{ks} = vector.mma %sl_{ks}, {rh}, {prev} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "        scf.yield %s_g0l_3, %s_g1l_3, %s_u0l_3, %s_u1l_3 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>",
        "      } else {",
        "        scf.yield %g0l_acc, %g1l_acc, %u0l_acc, %u1l_acc : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>",
        "      }",
        f"      scf.yield %lo_g0, %lo_g1, %lo_u0, %lo_u1, %g0h_acc, %g1h_acc, %u0h_acc, %u1h_acc : {acc_types}",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        f"    scf.yield {live_outs}, %x_lo_n, %x_hi_n, %gd_n, %gq_n, %ud_n, %uq_n : {acc_types}, vector<16xf16>, vector<16xf16>, i8, vector<8xi8>, i8, vector<8xi8>",
        "  }",
        "  // Epilogue: load inv scales for all 128 rows into x_stage (8192 B <= 18432 B)",
        "  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<128x16xf32>",
        "  %writer = index.cmp ult, %workitem, %c128 : index",
        "  scf.if %writer {",
        "    %inv_prow_v = index.add %row_origin, %workitem : index",
        "    %inv_prow = index.assume %inv_prow_v [lt(%inv_prow_v, %gx_rm_rows)] : index",
        "    %inv_a32 = view.load %gx_rm_view[%inv_prow] : view<[%gx_rm_rows]xi32> -> i32",
        "    %inv_ac = scalar.maxsi %inv_a32, %c0_i32 : i32",
        "    %inv_ab = scalar.assume %inv_ac [range(%inv_ac, 0, 1048575)] : i32",
        "    %inv_a = index.cast %inv_ab : i32 to index",
        "    %inv_t = index.div %inv_a, %c8 : index",
        "    %inv_row = index.assume %inv_t [lt(%inv_t, %xrows)] : index",
        "    %inverse = view.load %s_view[%inv_row, %c1] : view<[%xrows]x2xf32> -> f32",
        "    %inverse16 = vector.splat %inverse : vector<16xf32>",
        "    %bw = index.assume %workitem [range(%workitem, 0, 127)] : index",
        "    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<128x16xf32>",
        "  }",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  %gus_view = buffer.view %gus[%origin] : buffer -> view<128xf32>",
        "  %gus_sv = view.load %gus_view[%expert] : view<128xf32> -> f32",
        "  %gus_s = kernel.subgroup.broadcast.first %gus_sv : f32",
        "  %gus_v = vector.splat %gus_s : vector<8xf32>",
        "  %oa_beta = vector.splatf 4.0 : vector<8xf32>",
        "  %oa_invb2 = vector.splatf 0.5 : vector<8xf32>",
        "  %oa_lb = vector.splatf 25.0 : vector<8xf32>",
        "  %oa_invlb2 = vector.splatf 0.08 : vector<8xf32>",
        "  %oa_two = vector.splatf 2.0 : vector<8xf32>",
        "  %oa_one = vector.splatf 1.0 : vector<8xf32>",
        "  %f16_out_lo = buffer.view %w_stage[%origin] : buffer -> view<32x68xf32>",
        "  %f16_out_hi = buffer.view %w2_stage[%origin] : buffer -> view<32x68xf32>",
        "  %f16_half = index.cmp ult, %wave_row_group, %c2 : index",
        "  %f16_wrow_m = index.rem %wave_row_group, %c2 : index",
        "  %f16_wrow_r = index.mul %f16_wrow_m, %c16 : index",
        "  %f16_wrow = index.assume %f16_wrow_r [range(%f16_wrow_r, 0, 16)] : index",
        "  %f16_st_lo = index.cmp ult, %x_stage_row_lo, %c32 : index",
        "  %f16_gc0 = index.add %out_origin, %x_stage_col : index",
        "  %f16_gc = index.assume %f16_gc0 [lt(%f16_gc0, %n)] : index",
        "  // Pass 1: lower 64 rows (0..63)",
        "  %inv_frag_lo = vector.fragment.load<result> %inv_view[%wave_row_lo, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
        "  %g0l_i = vector.mulf %g0l_raw, %inv_frag_lo : vector<8xf32>",
        "  %g1l_i = vector.mulf %g1l_raw, %inv_frag_lo : vector<8xf32>",
        "  %u0l_i = vector.mulf %u0l_raw, %inv_frag_lo : vector<8xf32>",
        "  %u1l_i = vector.mulf %u1l_raw, %inv_frag_lo : vector<8xf32>",
        "  %g0ls = vector.mulf %g0l_i, %gus_v : vector<8xf32>",
        "  %g1ls = vector.mulf %g1l_i, %gus_v : vector<8xf32>",
        "  %u0ls = vector.mulf %u0l_i, %gus_v : vector<8xf32>",
        "  %u1ls = vector.mulf %u1l_i, %gus_v : vector<8xf32>",
    ]
    L += _act_block("al0", "%g0ls", "%u0ls", "%act0l", "  ")
    L += _act_block("al1", "%g1ls", "%u1ls", "%act1l", "  ")
    L += [
        "  scf.if %f16_half {",
        "    vector.fragment.store<result> %act0l, %f16_out_lo[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    vector.fragment.store<result> %act1l, %f16_out_lo[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "  } else {",
        "    vector.fragment.store<result> %act0l, %f16_out_hi[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    vector.fragment.store<result> %act1l, %f16_out_hi[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "  }",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  %f16_v32_l = scf.if %f16_st_lo -> (vector<16xf32>) {",
        "    %sr_l0 = index.assume %x_stage_row_lo [range(%x_stage_row_lo, 0, 31)] : index",
        "    %ld_l0 = vector.load %f16_out_lo[%sr_l0, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "    scf.yield %ld_l0 : vector<16xf32>",
        "  } else {",
        "    %sr_l1r = index.sub %x_stage_row_lo, %c32 : index",
        "    %sr_l1 = index.assume %sr_l1r [range(%sr_l1r, 0, 31)] : index",
        "    %ld_l1 = vector.load %f16_out_hi[%sr_l1, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "    scf.yield %ld_l1 : vector<16xf32>",
        "  }",
        "  %f16_v16_l = vector.fptrunc %f16_v32_l : vector<16xf32> to vector<16xf16>",
        "  %f16_grl0 = index.add %row_origin, %x_stage_row_lo : index",
        "  %f16_grl = index.assume %f16_grl0 [lt(%f16_grl0, %rows)] : index",
        "  vector.store %f16_v16_l, %y16_view[%f16_grl, %f16_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        "  // Pass 2: upper 64 rows (64..127), skipped entirely when wave_rows <= 64",
        "  scf.if %has_hi64 {",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %inv_frag_hi = vector.fragment.load<result> %inv_view[%wave_row_hi, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
        "    %g0h_i = vector.mulf %g0h_raw, %inv_frag_hi : vector<8xf32>",
        "    %g1h_i = vector.mulf %g1h_raw, %inv_frag_hi : vector<8xf32>",
        "    %u0h_i = vector.mulf %u0h_raw, %inv_frag_hi : vector<8xf32>",
        "    %u1h_i = vector.mulf %u1h_raw, %inv_frag_hi : vector<8xf32>",
        "    %g0hs = vector.mulf %g0h_i, %gus_v : vector<8xf32>",
        "    %g1hs = vector.mulf %g1h_i, %gus_v : vector<8xf32>",
        "    %u0hs = vector.mulf %u0h_i, %gus_v : vector<8xf32>",
        "    %u1hs = vector.mulf %u1h_i, %gus_v : vector<8xf32>",
    ]
    L += _act_block("ah0", "%g0hs", "%u0hs", "%act0h", "    ")
    L += _act_block("ah1", "%g1hs", "%u1hs", "%act1h", "    ")
    L += [
        "    scf.if %f16_half {",
        "      vector.fragment.store<result> %act0h, %f16_out_lo[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "      vector.fragment.store<result> %act1h, %f16_out_lo[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    } else {",
        "      vector.fragment.store<result> %act0h, %f16_out_hi[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "      vector.fragment.store<result> %act1h, %f16_out_hi[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %f16_v32_h = scf.if %f16_st_lo -> (vector<16xf32>) {",
        "      %sr_h0 = index.assume %x_stage_row_lo [range(%x_stage_row_lo, 0, 31)] : index",
        "      %ld_h0 = vector.load %f16_out_lo[%sr_h0, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "      scf.yield %ld_h0 : vector<16xf32>",
        "    } else {",
        "      %sr_h1r = index.sub %x_stage_row_lo, %c32 : index",
        "      %sr_h1 = index.assume %sr_h1r [range(%sr_h1r, 0, 31)] : index",
        "      %ld_h1 = vector.load %f16_out_hi[%sr_h1, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "      scf.yield %ld_h1 : vector<16xf32>",
        "    }",
        "    %f16_v16_h = vector.fptrunc %f16_v32_h : vector<16xf32> to vector<16xf16>",
        "    %f16_grh0 = index.add %row_origin, %x_stage_row_hi : index",
        "    %f16_grh = index.assume %f16_grh0 [lt(%f16_grh0, %rows)] : index",
        "    vector.store %f16_v16_h, %y16_view[%f16_grh, %f16_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        "  }",
        "  }",
        "  kernel.return",
        "}",
        "",
    ]
    return "\n".join(L)


def emit_ggu_nvfp4_m128_n128(shared_w: bool = False) -> str:
    """128x128x64 NVFP4 fused gate+up+act -> direct f16 act16, software-pipelined.
    shared_w=False: 55.3 KB LDS (25% occ, 2 barriers/step, 64 WMMAs/step, matching dense m256).
    shared_w=True : 36.9 KB LDS (37.5% occ, 4 barriers/step, 64 WMMAs/step, shared w_stage).
    """
    L = [
        'kernel.def target(@rdna4) export("g_moe_ggu_nvfp4_fused_sc_k2304_rdna4") @g_moe_ggu_nvfp4_fused_sc_k2304_rdna4(%rows: index, %n: index) {',
        "  %c1 = index.constant 1 : index",
        "  %c128 = index.constant 128 : index",
        "  %c256 = index.constant 256 : index",
        "  %row_tiles = index.div %rows, %c128 : index",
        "  %out_tiles = index.div %n, %c128 : index",
        "  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index",
        "} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, %w: buffer, %w2: buffer, %gus: buffer, %y: buffer) {",
        f"  // MoE grouped GEMM M=128 (128x128x64 NVFP4 gate+up -> f16 act16, shared_w={shared_w}, pipelined).",
        "  %rows = index.assume %rows_arg [range(%rows_arg, 128, 131072), multiple_of(%rows_arg, 128)] : index",
        "  %k = index.assume %k_arg [range(%k_arg, 64, 65536), multiple_of(%k_arg, 64)] : index",
        "  %n = index.assume %n_arg [range(%n_arg, 128, 1152), multiple_of(%n_arg, 128)] : index",
        "  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index",
        "  %origin = index.constant 0 : offset",
        "  %two_o = index.constant 2 : offset",
        "  %four_o = index.constant 4 : offset",
        "  %sixteen_o = index.constant 16 : offset",
        "  %twenty_o = index.constant 20 : offset",
        "  %block_bytes = index.constant 36 : offset",
        "  %c0 = index.constant 0 : index",
        "  %c1 = index.constant 1 : index",
        "  %c2 = index.constant 2 : index",
        "  %c4 = index.constant 4 : index",
        "  %c8 = index.constant 8 : index",
        "  %c16 = index.constant 16 : index",
        "  %c32 = index.constant 32 : index",
        "  %c48 = index.constant 48 : index",
        "  %c64 = index.constant 64 : index",
        "  %c96 = index.constant 96 : index",
        "  %c128 = index.constant 128 : index",
        "  %c0_i32 = scalar.constant 0 : i32",
        "  %cEm1 = scalar.constant 127 : i32",
        "  %row_tile_count = index.div %rows, %c128 : index",
        "  %out_tile_count = index.div %n, %c128 : index",
        "  %row_group_id = kernel.workgroup.id<y> : index",
        "  %out_group_id = kernel.workgroup.id<x> : index",
        "  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index",
        "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index",
        "  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>",
        "  %tile_expert_v = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32",
        "  %tile_expert = kernel.subgroup.broadcast.first %tile_expert_v : i32",
        "  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32",
        "  scf.if %tile_used {",
        "  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32",
        "  %e_b = scalar.assume %e_lo [range(%e_lo, 0, 127)] : i32",
        "  %expert = index.cast %e_b : i32 to index",
        "  %expert_bytes = index.constant 2985984 : offset",
        "  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset",
        "  %workitem_id = kernel.workitem.id<x> : index",
        "  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index",
        "  %subgroup = kernel.subgroup.id : index",
        "  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index",
        "  %wave_row_group = index.div %bounded_subgroup, %c2 : index",
        "  %wave_out_group = index.rem %bounded_subgroup, %c2 : index",
        "  %wave_row_lo_v = index.mul %wave_row_group, %c16 : index",
        "  %wave_row_lo = index.assume %wave_row_lo_v [range(%wave_row_lo_v, 0, 48)] : index",
        "  %wave_row_hi_v = index.add %wave_row_lo, %c64 : index",
        "  %wave_row_hi = index.assume %wave_row_hi_v [range(%wave_row_hi_v, 64, 112)] : index",
        "  %wave_out_v = index.mul %wave_out_group, %c64 : index",
        "  %wave_out0 = index.assume %wave_out_v [range(%wave_out_v, 0, 64)] : index",
        "  %wave_out1 = index.add %wave_out0, %c16 : index",
        "  %wave_out2 = index.add %wave_out0, %c32 : index",
        "  %wave_out3 = index.add %wave_out0, %c48 : index",
        "  %row_origin = index.mul %row_group, %c128 : index",
        "  %out_origin = index.mul %out_group, %c128 : index",
        "  %blocks = index.div %k, %c64 : index",
        "  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset",
        "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>",
        "  %y16_view = buffer.view %y[%origin] : buffer -> view<[%rows]x[%n]xf16>",
        "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>",
        "  %stage_bytes = index.constant 18432 : offset",
        "  %x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer",
        "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer",
        "  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        "  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x128xf16, %w_stage_view_tl>",
    ]
    if not shared_w:
        L += [
            "  %w2_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer",
            "  %w2_stage_view = buffer.view %w2_stage[%origin] : buffer -> view<128x72xf16>",
            "  %w2_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
            "  %w2_stage_view_t = buffer.view %w2_stage[%origin] : buffer -> view<72x128xf16, %w2_stage_view_tl>",
        ]
    L += [
        "  %wave_rows32_v = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32",
        "  %wave_rows32 = kernel.subgroup.broadcast.first %wave_rows32_v : i32",
        "  %wave_rows_b = scalar.assume %wave_rows32 [range(%wave_rows32, 0, 128)] : i32",
        "  %wave_rows = index.cast %wave_rows_b : i32 to index",
        "  %has_hi64 = index.cmp ugt, %wave_rows, %c64 : index",
        "  %wave_live_lo = index.cmp ult, %wave_row_lo, %wave_rows : index",
        "  %wave_live_hi = index.cmp ult, %wave_row_hi, %wave_rows : index",
        "  %x_stage_row_lo = index.div %workitem, %c4 : index",
        "  %x_stage_row_hi = index.add %x_stage_row_lo, %c64 : index",
        "  %x_stage_q = index.rem %workitem, %c4 : index",
        "  %x_stage_col = index.mul %x_stage_q, %c16 : index",
        "  %gx_rm_rows = index.mul %row_tile_count, %c128 : index",
        "  %gx_rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%gx_rm_rows]xi32>",
        "  %x_row_raw_lo = index.add %row_origin, %x_stage_row_lo : index",
        "  %gx_prow_lo = index.assume %x_row_raw_lo [lt(%x_row_raw_lo, %gx_rm_rows)] : index",
        "  %gx_a32_lo = view.load %gx_rm_view[%gx_prow_lo] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_lo = scalar.cmpi sge, %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ac_lo = scalar.maxsi %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ab_lo = scalar.assume %gx_ac_lo [range(%gx_ac_lo, 0, 1048575)] : i32",
        "  %gx_a_lo = index.cast %gx_ab_lo : i32 to index",
        "  %gx_t_lo = index.div %gx_a_lo, %c8 : index",
        "  %bounded_x_row_lo = index.assume %gx_t_lo [lt(%gx_t_lo, %xrows)] : index",
        "  %x_row_raw_hi = index.add %row_origin, %x_stage_row_hi : index",
        "  %gx_prow_hi = index.assume %x_row_raw_hi [lt(%x_row_raw_hi, %gx_rm_rows)] : index",
        "  %gx_a32_hi = view.load %gx_rm_view[%gx_prow_hi] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_hi = scalar.cmpi sge, %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ac_hi = scalar.maxsi %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ab_hi = scalar.assume %gx_ac_hi [range(%gx_ac_hi, 0, 1048575)] : i32",
        "  %gx_a_hi = index.cast %gx_ab_hi : i32 to index",
        "  %gx_t_hi = index.div %gx_a_hi, %c8 : index",
        "  %bounded_x_row_hi = index.assume %gx_t_hi [lt(%gx_t_hi, %xrows)] : index",
        "  %up_bytes = index.constant 1492992 : offset",
        "  %w_stage_row = index.div %workitem, %c2 : index",
        "  %w_stage_half = index.rem %workitem, %c2 : index",
        "  %w_stage_col0 = index.mul %w_stage_half, %c32 : index",
        "  %w_stage_col1 = index.add %w_stage_col0, %c16 : index",
        "  %w_out_row = index.add %out_origin, %w_stage_row : index",
        "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
        "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
        "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
        "  %w_d_add = index.scale %w_stage_half, %two_o : index, offset -> offset",
        "  %g_d_base = index.add %w_row_origin, %w_d_add : offset",
        "  %w_q_add = index.scale %w_stage_half, %sixteen_o : index, offset -> offset",
        "  %w_q_base0 = index.add %w_row_origin, %four_o : offset",
        "  %g_q_base = index.add %w_q_base0, %w_q_add : offset",
        "  %u_d_base = index.add %g_d_base, %up_bytes : offset",
        "  %u_q_base = index.add %g_q_base, %up_bytes : offset",
        "  %zero8 = vector.splatf 0.0 : vector<8xf32>",
        "  %zero16h = vector.splatf 0.0 : vector<16xf16>",
        "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>",
        "  %k_steps = index.div %k, %c64 : index",
        "  %pf_last = index.sub %k_steps, %c1 : index",
    ]
    L += _nvfp4_consts("  ")
    L += [
        "  %x_lo_p = vector.load %x_view[%bounded_x_row_lo, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "  %x_hi_p = scf.if %has_hi64 -> (vector<16xf16>) {",
        "    %xhp = vector.load %x_view[%bounded_x_row_hi, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    scf.yield %xhp : vector<16xf16>",
        "  } else {",
        "    scf.yield %zero16h : vector<16xf16>",
        "  }",
        "  %gd_view_p = buffer.view %w[%g_d_base] : buffer -> view<2xi8>",
        "  %gd_p = vector.load %gd_view_p[0] : view<2xi8> -> vector<2xi8>",
        "  %gq_view_p = buffer.view %w[%g_q_base] : buffer -> view<16xi8>",
        "  %gq_p = vector.load %gq_view_p[0] : view<16xi8> -> vector<16xi8>",
        "  %ud_view_p = buffer.view %w2[%u_d_base] : buffer -> view<2xi8>",
        "  %ud_p = vector.load %ud_view_p[0] : view<2xi8> -> vector<2xi8>",
        "  %uq_view_p = buffer.view %w2[%u_q_base] : buffer -> view<16xi8>",
        "  %uq_p = vector.load %uq_view_p[0] : view<16xi8> -> vector<16xi8>",
    ]
    g_names = ["g0l", "g1l", "g2l", "g3l", "g0h", "g1h", "g2h", "g3h"]
    u_names = ["u0l", "u1l", "u2l", "u3l", "u0h", "u1h", "u2h", "u3h"]
    acc_names = g_names + u_names
    acc_outs = ", ".join(f"%{a}_raw" for a in acc_names)
    acc_inits = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in acc_names)
    acc_types = ", ".join(["vector<8xf32>"] * 16)
    g_types = ", ".join(["vector<8xf32>"] * 8)
    lo4_types = ", ".join(["vector<8xf32>"] * 4)
    lo8_types = ", ".join(["vector<8xf32>"] * 8)

    L += [
        f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3, %pf_u4, %pf_u5 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_gd = %gd_p : vector<2xi8>, %cr_gq = %gq_p : vector<16xi8>, %cr_ud = %ud_p : vector<2xi8>, %cr_uq = %uq_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, vector<2xi8>, vector<16xi8>, vector<2xi8>, vector<16xi8>) {{",
        "    %x_lo_s = scf.select %gx_valid_lo, %cr_xlo, %zero16h : vector<16xf16>",
        "    vector.store %x_lo_s, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    scf.if %has_hi64 {",
        "      %x_hi_s = scf.select %gx_valid_hi, %cr_xhi, %zero16h : vector<16xf16>",
        "      vector.store %x_hi_s, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    }",
        "    %gd0 = vector.extract %cr_gd[0] : vector<2xi8> -> i8",
        "    %gd1 = vector.extract %cr_gd[1] : vector<2xi8> -> i8",
        "    %gq0 = vector.slice %cr_gq[0] : vector<16xi8> -> vector<8xi8>",
        "    %gq1 = vector.slice %cr_gq[8] : vector<16xi8> -> vector<8xi8>",
    ]
    L += _nvfp4_decode16("dg0", "%gd0", "%gq0", "%dg_v0", "    ")
    L += ["    vector.store %dg_v0, %w_stage_view[%w_stage_row, %w_stage_col0] : vector<16xf16>, view<128x72xf16>"]
    L += _nvfp4_decode16("dg1", "%gd1", "%gq1", "%dg_v1", "    ")
    L += ["    vector.store %dg_v1, %w_stage_view[%w_stage_row, %w_stage_col1] : vector<16xf16>, view<128x72xf16>"]

    if not shared_w:
        L += [
            "    %ud0 = vector.extract %cr_ud[0] : vector<2xi8> -> i8",
            "    %ud1 = vector.extract %cr_ud[1] : vector<2xi8> -> i8",
            "    %uq0 = vector.slice %cr_uq[0] : vector<16xi8> -> vector<8xi8>",
            "    %uq1 = vector.slice %cr_uq[8] : vector<16xi8> -> vector<8xi8>",
        ]
        L += _nvfp4_decode16("du0", "%ud0", "%uq0", "%du_v0", "    ")
        L += ["    vector.store %du_v0, %w2_stage_view[%w_stage_row, %w_stage_col0] : vector<16xf16>, view<128x72xf16>"]
        L += _nvfp4_decode16("du1", "%ud1", "%uq1", "%du_v1", "    ")
        L += [
            "    vector.store %du_v1, %w2_stage_view[%w_stage_row, %w_stage_col1] : vector<16xf16>, view<128x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
            "    %gd_off_n = index.add %g_d_base, %step_off_n : offset",
            "    %gd_view_n = buffer.view %w[%gd_off_n] : buffer -> view<2xi8>",
            "    %gd_n = vector.load %gd_view_n[0] : view<2xi8> -> vector<2xi8>",
            "    %gq_off_n = index.add %g_q_base, %step_off_n : offset",
            "    %gq_view_n = buffer.view %w[%gq_off_n] : buffer -> view<16xi8>",
            "    %gq_n = vector.load %gq_view_n[0] : view<16xi8> -> vector<16xi8>",
            "    %ud_off_n = index.add %u_d_base, %step_off_n : offset",
            "    %ud_view_n = buffer.view %w2[%ud_off_n] : buffer -> view<2xi8>",
            "    %ud_n = vector.load %ud_view_n[0] : view<2xi8> -> vector<2xi8>",
            "    %uq_off_n = index.add %u_q_base, %step_off_n : offset",
            "    %uq_view_n = buffer.view %w2[%uq_off_n] : buffer -> view<16xi8>",
            "    %uq_n = vector.load %uq_view_n[0] : view<16xi8> -> vector<16xi8>",
        ]
        live_outs = ", ".join(f"%l_{a}" for a in acc_names)
        L += [f"    {live_outs} = scf.if %wave_live_hi -> ({acc_types}) {{"]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L += [
                f"      %ll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
                f"      %lh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            ]
            for i in range(4):
                L.append(f"      %rg{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%g{i}l_acc" if ks == 0 else f"%f_g{i}l_{ks-1}"
                ph = f"%g{i}h_acc" if ks == 0 else f"%f_g{i}h_{ks-1}"
                L.append(f"      %f_g{i}l_{ks} = vector.mma %ll_{ks}, %rg{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                L.append(f"      %f_g{i}h_{ks} = vector.mma %lh_{ks}, %rg{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            for i in range(4):
                L.append(f"      %ru{i}_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w2_stage_view_tl> -> vector<8xf16>")
                pl = f"%u{i}l_acc" if ks == 0 else f"%f_u{i}l_{ks-1}"
                ph = f"%u{i}h_acc" if ks == 0 else f"%f_u{i}h_{ks-1}"
                L.append(f"      %f_u{i}l_{ks} = vector.mma %ll_{ks}, %ru{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                L.append(f"      %f_u{i}h_{ks} = vector.mma %lh_{ks}, %ru{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            "      scf.yield " + ", ".join(f"%f_{a}_3" for a in acc_names) + f" : {acc_types}",
            "    } else {",
            f"      %lo_g0, %lo_g1, %lo_g2, %lo_g3, %lo_u0, %lo_u1, %lo_u2, %lo_u3 = scf.if %wave_live_lo -> ({lo8_types}) {{",
        ]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L.append(f"        %sl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>")
            for i in range(4):
                L.append(f"        %srg{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%g{i}l_acc" if ks == 0 else f"%s_g{i}l_{ks-1}"
                L.append(f"        %s_g{i}l_{ks} = vector.mma %sl_{ks}, %srg{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            for i in range(4):
                L.append(f"        %sru{i}_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w2_stage_view_tl> -> vector<8xf16>")
                pl = f"%u{i}l_acc" if ks == 0 else f"%s_u{i}l_{ks-1}"
                L.append(f"        %s_u{i}l_{ks} = vector.mma %sl_{ks}, %sru{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            f"        scf.yield %s_g0l_3, %s_g1l_3, %s_g2l_3, %s_g3l_3, %s_u0l_3, %s_u1l_3, %s_u2l_3, %s_u3l_3 : {lo8_types}",
            "      } else {",
            f"        scf.yield %g0l_acc, %g1l_acc, %g2l_acc, %g3l_acc, %u0l_acc, %u1l_acc, %u2l_acc, %u3l_acc : {lo8_types}",
            "      }",
            f"      scf.yield %lo_g0, %lo_g1, %lo_g2, %lo_g3, %g0h_acc, %g1h_acc, %g2h_acc, %g3h_acc, %lo_u0, %lo_u1, %lo_u2, %lo_u3, %u0h_acc, %u1h_acc, %u2h_acc, %u3h_acc : {acc_types}",
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            f"    scf.yield {live_outs}, %x_lo_n, %x_hi_n, %gd_n, %gq_n, %ud_n, %uq_n : {acc_types}, vector<16xf16>, vector<16xf16>, vector<2xi8>, vector<16xi8>, vector<2xi8>, vector<16xi8>",
            "  }",
        ]
    else:
        # shared_w=True: Gate WMMAs use w_stage, then Up dequant overwrites w_stage and runs Up WMMAs.
        lg_outs = ", ".join(f"%l_{a}" for a in g_names)
        lu_outs = ", ".join(f"%l_{a}" for a in u_names)
        L += [
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
            "    %gd_off_n = index.add %g_d_base, %step_off_n : offset",
            "    %gd_view_n = buffer.view %w[%gd_off_n] : buffer -> view<2xi8>",
            "    %gd_n = vector.load %gd_view_n[0] : view<2xi8> -> vector<2xi8>",
            "    %gq_off_n = index.add %g_q_base, %step_off_n : offset",
            "    %gq_view_n = buffer.view %w[%gq_off_n] : buffer -> view<16xi8>",
            "    %gq_n = vector.load %gq_view_n[0] : view<16xi8> -> vector<16xi8>",
            f"    {lg_outs} = scf.if %wave_live_hi -> ({g_types}) {{",
        ]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L += [
                f"      %gll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
                f"      %glh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            ]
            for i in range(4):
                L.append(f"      %rg{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%g{i}l_acc" if ks == 0 else f"%f_g{i}l_{ks-1}"
                ph = f"%g{i}h_acc" if ks == 0 else f"%f_g{i}h_{ks-1}"
                L.append(f"      %f_g{i}l_{ks} = vector.mma %gll_{ks}, %rg{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                L.append(f"      %f_g{i}h_{ks} = vector.mma %glh_{ks}, %rg{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            "      scf.yield " + ", ".join(f"%f_{a}_3" for a in g_names) + f" : {g_types}",
            "    } else {",
            f"      %lo_g0, %lo_g1, %lo_g2, %lo_g3 = scf.if %wave_live_lo -> ({lo4_types}) {{",
        ]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L.append(f"        %gsl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>")
            for i in range(4):
                L.append(f"        %srg{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%g{i}l_acc" if ks == 0 else f"%s_g{i}l_{ks-1}"
                L.append(f"        %s_g{i}l_{ks} = vector.mma %gsl_{ks}, %srg{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            f"        scf.yield %s_g0l_3, %s_g1l_3, %s_g2l_3, %s_g3l_3 : {lo4_types}",
            "      } else {",
            f"        scf.yield %g0l_acc, %g1l_acc, %g2l_acc, %g3l_acc : {lo4_types}",
            "      }",
            f"      scf.yield %lo_g0, %lo_g1, %lo_g2, %lo_g3, %g0h_acc, %g1h_acc, %g2h_acc, %g3h_acc : {g_types}",
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %ud0 = vector.extract %cr_ud[0] : vector<2xi8> -> i8",
            "    %ud1 = vector.extract %cr_ud[1] : vector<2xi8> -> i8",
            "    %uq0 = vector.slice %cr_uq[0] : vector<16xi8> -> vector<8xi8>",
            "    %uq1 = vector.slice %cr_uq[8] : vector<16xi8> -> vector<8xi8>",
        ]
        L += _nvfp4_decode16("du0", "%ud0", "%uq0", "%du_v0", "    ")
        L += ["    vector.store %du_v0, %w_stage_view[%w_stage_row, %w_stage_col0] : vector<16xf16>, view<128x72xf16>"]
        L += _nvfp4_decode16("du1", "%ud1", "%uq1", "%du_v1", "    ")
        L += [
            "    vector.store %du_v1, %w_stage_view[%w_stage_row, %w_stage_col1] : vector<16xf16>, view<128x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %ud_off_n = index.add %u_d_base, %step_off_n : offset",
            "    %ud_view_n = buffer.view %w2[%ud_off_n] : buffer -> view<2xi8>",
            "    %ud_n = vector.load %ud_view_n[0] : view<2xi8> -> vector<2xi8>",
            "    %uq_off_n = index.add %u_q_base, %step_off_n : offset",
            "    %uq_view_n = buffer.view %w2[%uq_off_n] : buffer -> view<16xi8>",
            "    %uq_n = vector.load %uq_view_n[0] : view<16xi8> -> vector<16xi8>",
            f"    {lu_outs} = scf.if %wave_live_hi -> ({g_types}) {{",
        ]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L += [
                f"      %ull_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
                f"      %ulh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            ]
            for i in range(4):
                L.append(f"      %ru{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%u{i}l_acc" if ks == 0 else f"%f_u{i}l_{ks-1}"
                ph = f"%u{i}h_acc" if ks == 0 else f"%f_u{i}h_{ks-1}"
                L.append(f"      %f_u{i}l_{ks} = vector.mma %ull_{ks}, %ru{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
                L.append(f"      %f_u{i}h_{ks} = vector.mma %ulh_{ks}, %ru{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            "      scf.yield " + ", ".join(f"%f_{a}_3" for a in u_names) + f" : {g_types}",
            "    } else {",
            f"      %lo_u0, %lo_u1, %lo_u2, %lo_u3 = scf.if %wave_live_lo -> ({lo4_types}) {{",
        ]
        for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
            L.append(f"        %usl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>")
            for i in range(4):
                L.append(f"        %sru{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x128xf16, %w_stage_view_tl> -> vector<8xf16>")
                pl = f"%u{i}l_acc" if ks == 0 else f"%s_u{i}l_{ks-1}"
                L.append(f"        %s_u{i}l_{ks} = vector.mma %usl_{ks}, %sru{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
        L += [
            f"        scf.yield %s_u0l_3, %s_u1l_3, %s_u2l_3, %s_u3l_3 : {lo4_types}",
            "      } else {",
            f"        scf.yield %u0l_acc, %u1l_acc, %u2l_acc, %u3l_acc : {lo4_types}",
            "      }",
            f"      scf.yield %lo_u0, %lo_u1, %lo_u2, %lo_u3, %u0h_acc, %u1h_acc, %u2h_acc, %u3h_acc : {g_types}",
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            f"    scf.yield {lg_outs}, {lu_outs}, %x_lo_n, %x_hi_n, %gd_n, %gq_n, %ud_n, %uq_n : {acc_types}, vector<16xf16>, vector<16xf16>, vector<2xi8>, vector<16xi8>, vector<2xi8>, vector<16xi8>",
            "  }",
        ]

    # Epilogue (32 rows at a time through w_stage view<32x132xf32> = 16,896 B <= 18,432 B)
    # Using all 256 threads coalesced: ep_row = workitem/8 (0..31), ep_col = (workitem%8)*16 (0..112)
    L += [
        "  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<128x16xf32>",
        "  %writer = index.cmp ult, %workitem, %c128 : index",
        "  scf.if %writer {",
        "    %inv_prow_v = index.add %row_origin, %workitem : index",
        "    %inv_prow = index.assume %inv_prow_v [lt(%inv_prow_v, %gx_rm_rows)] : index",
        "    %inv_a32 = view.load %gx_rm_view[%inv_prow] : view<[%gx_rm_rows]xi32> -> i32",
        "    %inv_ac = scalar.maxsi %inv_a32, %c0_i32 : i32",
        "    %inv_ab = scalar.assume %inv_ac [range(%inv_ac, 0, 1048575)] : i32",
        "    %inv_a = index.cast %inv_ab : i32 to index",
        "    %inv_t = index.div %inv_a, %c8 : index",
        "    %inv_row = index.assume %inv_t [lt(%inv_t, %xrows)] : index",
        "    %inverse = view.load %s_view[%inv_row, %c1] : view<[%xrows]x2xf32> -> f32",
        "    %inverse16 = vector.splat %inverse : vector<16xf32>",
        "    %bw = index.assume %workitem [range(%workitem, 0, 127)] : index",
        "    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<128x16xf32>",
        "  }",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  %gus_view = buffer.view %gus[%origin] : buffer -> view<128xf32>",
        "  %gus_sv = view.load %gus_view[%expert] : view<128xf32> -> f32",
        "  %gus_s = kernel.subgroup.broadcast.first %gus_sv : f32",
        "  %gus_v = vector.splat %gus_s : vector<8xf32>",
        "  %oa_beta = vector.splatf 4.0 : vector<8xf32>",
        "  %oa_invb2 = vector.splatf 0.5 : vector<8xf32>",
        "  %oa_lb = vector.splatf 25.0 : vector<8xf32>",
        "  %oa_invlb2 = vector.splatf 0.08 : vector<8xf32>",
        "  %oa_two = vector.splatf 2.0 : vector<8xf32>",
        "  %oa_one = vector.splatf 1.0 : vector<8xf32>",
        "  %epi_view = buffer.view %w_stage[%origin] : buffer -> view<32x132xf32>",
        "  %ep_row = index.div %workitem, %c8 : index",
        "  %ep_oct = index.rem %workitem, %c8 : index",
        "  %ep_col = index.mul %ep_oct, %c16 : index",
        "  %ep_gc0 = index.add %out_origin, %ep_col : index",
        "  %ep_gc = index.assume %ep_gc0 [lt(%ep_gc0, %n)] : index",
        "  %f16_half = index.cmp ult, %wave_row_group, %c2 : index",
        "  %f16_half_hi = index.cmp uge, %wave_row_group, %c2 : index",
        "  %f16_wrow_m = index.rem %wave_row_group, %c2 : index",
        "  %f16_wrow_r = index.mul %f16_wrow_m, %c16 : index",
        "  %f16_wrow = index.assume %f16_wrow_r [range(%f16_wrow_r, 0, 16)] : index",
        "  // Lower 64 rows",
        "  %inv_frag_lo = vector.fragment.load<result> %inv_view[%wave_row_lo, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
    ]
    for i in range(4):
        L += [
            f"  %g{i}l_i = vector.mulf %g{i}l_raw, %inv_frag_lo : vector<8xf32>",
            f"  %u{i}l_i = vector.mulf %u{i}l_raw, %inv_frag_lo : vector<8xf32>",
            f"  %g{i}ls = vector.mulf %g{i}l_i, %gus_v : vector<8xf32>",
            f"  %u{i}ls = vector.mulf %u{i}l_i, %gus_v : vector<8xf32>",
        ]
        L += _act_block(f"al{i}", f"%g{i}ls", f"%u{i}ls", f"%act{i}l", "  ")

    if not shared_w:
        L += [
            "  %epi_view2 = buffer.view %w2_stage[%origin] : buffer -> view<32x132xf32>",
            "  scf.if %f16_half {",
        ]
        for i in range(4):
            L.append(f"    vector.fragment.store<result> %act{i}l, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L.append("  } else {")
        for i in range(4):
            L.append(f"    vector.fragment.store<result> %act{i}l, %epi_view2[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "  }",
            "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "  %el0 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "  %el0h = vector.fptrunc %el0 : vector<16xf32> to vector<16xf16>",
            "  %gr_l0r = index.add %row_origin, %ep_row : index",
            "  %gr_l0 = index.assume %gr_l0r [lt(%gr_l0r, %rows)] : index",
            "  vector.store %el0h, %y16_view[%gr_l0, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
            "  %el1 = vector.load %epi_view2[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "  %el1h = vector.fptrunc %el1 : vector<16xf32> to vector<16xf16>",
            "  %gr_l1r = index.add %gr_l0, %c32 : index",
            "  %gr_l1 = index.assume %gr_l1r [lt(%gr_l1r, %rows)] : index",
            "  vector.store %el1h, %y16_view[%gr_l1, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        ]
    else:
        L += ["  scf.if %f16_half {"]
        for i in range(4):
            L.append(f"    vector.fragment.store<result> %act{i}l, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "  }",
            "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "  %el0 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "  %el0h = vector.fptrunc %el0 : vector<16xf32> to vector<16xf16>",
            "  %gr_l0r = index.add %row_origin, %ep_row : index",
            "  %gr_l0 = index.assume %gr_l0r [lt(%gr_l0r, %rows)] : index",
            "  vector.store %el0h, %y16_view[%gr_l0, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
            "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "  scf.if %f16_half_hi {",
        ]
        for i in range(4):
            L.append(f"    vector.fragment.store<result> %act{i}l, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "  }",
            "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "  %el1 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "  %el1h = vector.fptrunc %el1 : vector<16xf32> to vector<16xf16>",
            "  %gr_l1r = index.add %gr_l0, %c32 : index",
            "  %gr_l1 = index.assume %gr_l1r [lt(%gr_l1r, %rows)] : index",
            "  vector.store %el1h, %y16_view[%gr_l1, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        ]

    # Upper 64 rows (guarded by %has_hi64)
    L += [
        "  scf.if %has_hi64 {",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %inv_frag_hi = vector.fragment.load<result> %inv_view[%wave_row_hi, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
    ]
    for i in range(4):
        L += [
            f"    %g{i}h_i = vector.mulf %g{i}h_raw, %inv_frag_hi : vector<8xf32>",
            f"    %u{i}h_i = vector.mulf %u{i}h_raw, %inv_frag_hi : vector<8xf32>",
            f"    %g{i}hs = vector.mulf %g{i}h_i, %gus_v : vector<8xf32>",
            f"    %u{i}hs = vector.mulf %u{i}h_i, %gus_v : vector<8xf32>",
        ]
        L += _act_block(f"ah{i}", f"%g{i}hs", f"%u{i}hs", f"%act{i}h", "    ")

    if not shared_w:
        L += ["    scf.if %f16_half {"]
        for i in range(4):
            L.append(f"      vector.fragment.store<result> %act{i}h, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L.append("    } else {")
        for i in range(4):
            L.append(f"      vector.fragment.store<result> %act{i}h, %epi_view2[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %eh0 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "    %eh0h = vector.fptrunc %eh0 : vector<16xf32> to vector<16xf16>",
            "    %gr_h0r = index.add %gr_l0, %c64 : index",
            "    %gr_h0 = index.assume %gr_h0r [lt(%gr_h0r, %rows)] : index",
            "    vector.store %eh0h, %y16_view[%gr_h0, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
            "    %eh1 = vector.load %epi_view2[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "    %eh1h = vector.fptrunc %eh1 : vector<16xf32> to vector<16xf16>",
            "    %gr_h1r = index.add %gr_l0, %c96 : index",
            "    %gr_h1 = index.assume %gr_h1r [lt(%gr_h1r, %rows)] : index",
            "    vector.store %eh1h, %y16_view[%gr_h1, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        ]
    else:
        L += ["    scf.if %f16_half {"]
        for i in range(4):
            L.append(f"      vector.fragment.store<result> %act{i}h, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %eh0 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "    %eh0h = vector.fptrunc %eh0 : vector<16xf32> to vector<16xf16>",
            "    %gr_h0r = index.add %gr_l0, %c64 : index",
            "    %gr_h0 = index.assume %gr_h0r [lt(%gr_h0r, %rows)] : index",
            "    vector.store %eh0h, %y16_view[%gr_h0, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    scf.if %f16_half_hi {",
        ]
        for i in range(4):
            L.append(f"      vector.fragment.store<result> %act{i}h, %epi_view[%f16_wrow, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<32x132xf32>")
        L += [
            "    }",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %eh1 = vector.load %epi_view[%ep_row, %ep_col] : view<32x132xf32> -> vector<16xf32>",
            "    %eh1h = vector.fptrunc %eh1 : vector<16xf32> to vector<16xf16>",
            "    %gr_h1r = index.add %gr_l0, %c96 : index",
            "    %gr_h1 = index.assume %gr_h1r [lt(%gr_h1r, %rows)] : index",
            "    vector.store %eh1h, %y16_view[%gr_h1, %ep_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        ]
    L += [
        "  }",
        "  }",
        "  kernel.return",
        "}",
        "",
    ]
    return "\n".join(L)


def emit_gdn_nvfp4_m128(tile_n: int = 64) -> str:
    """128x{64,128,256}x64 NVFP4 MoE down projection -> f32 ys_sorted, software-pipelined."""
    assert tile_n in (64, 128, 256)
    ntiles = tile_n // 32  # 2, 4, or 8
    w_stage_bytes = tile_n * 72 * 2

    L = [
        'kernel.def target(@rdna4) export("g_moe_gdn_nvfp4_k1152_rdna4") @g_moe_gdn_nvfp4_k1152_rdna4(%rows: index, %n: index) {',
        "  %c1 = index.constant 1 : index",
        f"  %ctn = index.constant {tile_n} : index",
        "  %c128 = index.constant 128 : index",
        "  %c256 = index.constant 256 : index",
        "  %row_tiles = index.div %rows, %c128 : index",
        "  %out_tiles = index.div %n, %ctn : index",
        "  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index",
        "} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, %w: buffer, %y: buffer) {",
        f"  // MoE grouped GEMM M=128 (128x{tile_n}x64 NVFP4 down, pipelined, RHS reuse).",
        "  %rows = index.assume %rows_arg [range(%rows_arg, 128, 131072), multiple_of(%rows_arg, 128)] : index",
        "  %k = index.assume %k_arg [range(%k_arg, 64, 65536), multiple_of(%k_arg, 64)] : index",
        f"  %n = index.assume %n_arg [range(%n_arg, {tile_n}, 2304), multiple_of(%n_arg, {tile_n})] : index",
        "  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index",
        "  %origin = index.constant 0 : offset",
        "  %one_o = index.constant 1 : offset",
        "  %two_o = index.constant 2 : offset",
        "  %four_o = index.constant 4 : offset",
        "  %eight_o = index.constant 8 : offset",
        "  %sixteen_o = index.constant 16 : offset",
        "  %twenty_o = index.constant 20 : offset",
        "  %block_bytes = index.constant 36 : offset",
        "  %c0 = index.constant 0 : index",
        "  %c1 = index.constant 1 : index",
        "  %c2 = index.constant 2 : index",
        "  %c4 = index.constant 4 : index",
        "  %c16 = index.constant 16 : index",
        "  %c32 = index.constant 32 : index",
        "  %c48 = index.constant 48 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        f"  %ctn = index.constant {tile_n} : index",
        "  %c0_i32 = scalar.constant 0 : i32",
        "  %cEm1 = scalar.constant 127 : i32",
        "  %row_tile_count = index.div %rows, %c128 : index",
        "  %out_tile_count = index.div %n, %ctn : index",
        "  %row_group_id = kernel.workgroup.id<y> : index",
        "  %out_group_id = kernel.workgroup.id<x> : index",
        "  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index",
        "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index",
        "  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>",
        "  %tile_expert_v = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32",
        "  %tile_expert = kernel.subgroup.broadcast.first %tile_expert_v : i32",
        "  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32",
        "  scf.if %tile_used {",
        "  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32",
        "  %e_b = scalar.assume %e_lo [range(%e_lo, 0, 127)] : i32",
        "  %expert = index.cast %e_b : i32 to index",
        "  %expert_bytes = index.constant 1492992 : offset",
        "  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset",
        "  %workitem_id = kernel.workitem.id<x> : index",
        "  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index",
        "  %subgroup = kernel.subgroup.id : index",
        "  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index",
        "  %wave_row_group = index.div %bounded_subgroup, %c2 : index",
        "  %wave_out_group = index.rem %bounded_subgroup, %c2 : index",
        "  %wave_row_lo_v = index.mul %wave_row_group, %c16 : index",
        "  %wave_row_lo = index.assume %wave_row_lo_v [range(%wave_row_lo_v, 0, 48)] : index",
        "  %wave_row_hi_v = index.add %wave_row_lo, %c64 : index",
        "  %wave_row_hi = index.assume %wave_row_hi_v [range(%wave_row_hi_v, 64, 112)] : index",
        f"  %c_wout = index.constant {tile_n // 2} : index",
        "  %wave_out_v = index.mul %wave_out_group, %c_wout : index",
        f"  %wave_out0 = index.assume %wave_out_v [range(%wave_out_v, 0, {tile_n // 2})] : index",
    ]
    for i in range(1, ntiles):
        L.append(f"  %c_wo{i} = index.constant {i * 16} : index")
        L.append(f"  %wave_out{i} = index.add %wave_out0, %c_wo{i} : index")
    L += [
        "  %row_origin = index.mul %row_group, %c128 : index",
        "  %out_origin = index.mul %out_group, %ctn : index",
        "  %blocks = index.div %k, %c64 : index",
        "  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset",
        "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>",
        "  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>",
        "  %tile_first = index.madd %row_origin, %n, %out_origin : index",
        "  %tile_offset = index.scale %tile_first, %four_o : index, offset -> offset",
        f"  %y_view = buffer.view %y[%tile_offset] : buffer -> view<128x{tile_n}xf32, %y_layout>",
        "  %x_stage_bytes = index.constant 18432 : offset",
        f"  %w_stage_bytes = index.constant {w_stage_bytes} : offset",
        "  %x_stage = buffer.alloca<workgroup> align(16) %x_stage_bytes : buffer",
        "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        f"  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<{tile_n}x72xf16>",
        "  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        f"  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x{tile_n}xf16, %w_stage_view_tl>",
        "  %wave_rows32_v = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32",
        "  %wave_rows32 = kernel.subgroup.broadcast.first %wave_rows32_v : i32",
        "  %wave_rows_b = scalar.assume %wave_rows32 [range(%wave_rows32, 0, 128)] : i32",
        "  %wave_rows = index.cast %wave_rows_b : i32 to index",
        "  %has_hi64 = index.cmp ugt, %wave_rows, %c64 : index",
        "  %wave_live_lo = index.cmp ult, %wave_row_lo, %wave_rows : index",
        "  %wave_live_hi = index.cmp ult, %wave_row_hi, %wave_rows : index",
        "  %x_stage_row_lo = index.div %workitem, %c4 : index",
        "  %x_stage_row_hi = index.add %x_stage_row_lo, %c64 : index",
        "  %x_stage_q = index.rem %workitem, %c4 : index",
        "  %x_stage_col = index.mul %x_stage_q, %c16 : index",
        "  %x_row_lo = index.add %row_origin, %x_stage_row_lo : index",
        "  %bounded_x_row_lo = index.assume %x_row_lo [lt(%x_row_lo, %xrows)] : index",
        "  %x_row_hi = index.add %row_origin, %x_stage_row_hi : index",
        "  %bounded_x_row_hi = index.assume %x_row_hi [lt(%x_row_hi, %xrows)] : index",
    ]
    if tile_n == 64:
        L += [
            "  %w_stage_row = index.div %workitem, %c4 : index",
            "  %w_stage_q = index.rem %workitem, %c4 : index",
            "  %w_stage_col = index.mul %w_stage_q, %c16 : index",
            "  %w_out_row = index.add %out_origin, %w_stage_row : index",
            "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
            "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
            "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
            "  %w_d_add = index.scale %w_stage_q, %one_o : index, offset -> offset",
            "  %w_d_base = index.add %w_row_origin, %w_d_add : offset",
            "  %w_q_add = index.scale %w_stage_q, %eight_o : index, offset -> offset",
            "  %w_q_base0 = index.add %w_row_origin, %four_o : offset",
            "  %w_q_base = index.add %w_q_base0, %w_q_add : offset",
        ]
    elif tile_n == 128:
        L += [
            "  %w_stage_row = index.div %workitem, %c2 : index",
            "  %w_stage_half = index.rem %workitem, %c2 : index",
            "  %w_stage_col0 = index.mul %w_stage_half, %c32 : index",
            "  %w_stage_col1 = index.add %w_stage_col0, %c16 : index",
            "  %w_out_row = index.add %out_origin, %w_stage_row : index",
            "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
            "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
            "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
            "  %w_d_add = index.scale %w_stage_half, %two_o : index, offset -> offset",
            "  %w_d_base = index.add %w_row_origin, %w_d_add : offset",
            "  %w_q_add = index.scale %w_stage_half, %sixteen_o : index, offset -> offset",
            "  %w_q_base0 = index.add %w_row_origin, %four_o : offset",
            "  %w_q_base = index.add %w_q_base0, %w_q_add : offset",
        ]
    else:
        L += [
            "  %w_out_row = index.add %out_origin, %workitem : index",
            "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
            "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
            "  %w_d_base = index.add %w_row_origin0, %expert_off : offset",
            "  %w_q0_base = index.add %w_d_base, %four_o : offset",
            "  %w_q1_base = index.add %w_d_base, %twenty_o : offset",
        ]
    L += [
        "  %zero8 = vector.splatf 0.0 : vector<8xf32>",
        "  %zero16h = vector.splatf 0.0 : vector<16xf16>",
        "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>",
        "  %k_steps = index.div %k, %c64 : index",
        "  %pf_last = index.sub %k_steps, %c1 : index",
    ]
    L += _nvfp4_consts("  ")
    L += [
        "  %x_lo_p = vector.load %x_view[%bounded_x_row_lo, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "  %x_hi_p = scf.if %has_hi64 -> (vector<16xf16>) {",
        "    %xhp = vector.load %x_view[%bounded_x_row_hi, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    scf.yield %xhp : vector<16xf16>",
        "  } else {",
        "    scf.yield %zero16h : vector<16xf16>",
        "  }",
    ]
    acc_names = [f"a{i}l" for i in range(ntiles)] + [f"a{i}h" for i in range(ntiles)]
    acc_outs = ", ".join(f"%{a}_raw" for a in acc_names)
    acc_inits = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in acc_names)
    acc_types = ", ".join(["vector<8xf32>"] * (2 * ntiles))

    if tile_n == 64:
        L += [
            "  %wd_view_p = buffer.view %w[%w_d_base] : buffer -> view<1xi8>",
            "  %wd_p = view.load %wd_view_p[0] : view<1xi8> -> i8",
            "  %wq_view_p = buffer.view %w[%w_q_base] : buffer -> view<8xi8>",
            "  %wq_p = vector.load %wq_view_p[0] : view<8xi8> -> vector<8xi8>",
            f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_wd = %wd_p : i8, %cr_wq = %wq_p : vector<8xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, i8, vector<8xi8>) {{",
            "    vector.store %cr_xlo, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    scf.if %has_hi64 {",
            "      vector.store %cr_xhi, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    }",
        ]
        L += _nvfp4_decode16("dw", "%cr_wd", "%cr_wq", "%dw_v", "    ")
        L += [
            "    vector.store %dw_v, %w_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
            "    %wd_off_n = index.add %w_d_base, %step_off_n : offset",
            "    %wd_view_n = buffer.view %w[%wd_off_n] : buffer -> view<1xi8>",
            "    %wd_n = view.load %wd_view_n[0] : view<1xi8> -> i8",
            "    %wq_off_n = index.add %w_q_base, %step_off_n : offset",
            "    %wq_view_n = buffer.view %w[%wq_off_n] : buffer -> view<8xi8>",
            "    %wq_n = vector.load %wq_view_n[0] : view<8xi8> -> vector<8xi8>",
        ]
        pf_yield = "%x_lo_n, %x_hi_n, %wd_n, %wq_n"
        pf_types = "vector<16xf16>, vector<16xf16>, i8, vector<8xi8>"
    elif tile_n == 128:
        L += [
            "  %wd_view_p = buffer.view %w[%w_d_base] : buffer -> view<2xi8>",
            "  %wd_p = vector.load %wd_view_p[0] : view<2xi8> -> vector<2xi8>",
            "  %wq_view_p = buffer.view %w[%w_q_base] : buffer -> view<16xi8>",
            "  %wq_p = vector.load %wq_view_p[0] : view<16xi8> -> vector<16xi8>",
            f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_wd = %wd_p : vector<2xi8>, %cr_wq = %wq_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, vector<2xi8>, vector<16xi8>) {{",
            "    vector.store %cr_xlo, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    scf.if %has_hi64 {",
            "      vector.store %cr_xhi, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    }",
            "    %wd0 = vector.extract %cr_wd[0] : vector<2xi8> -> i8",
            "    %wd1 = vector.extract %cr_wd[1] : vector<2xi8> -> i8",
            "    %wq0 = vector.slice %cr_wq[0] : vector<16xi8> -> vector<8xi8>",
            "    %wq1 = vector.slice %cr_wq[8] : vector<16xi8> -> vector<8xi8>",
        ]
        L += _nvfp4_decode16("dw0", "%wd0", "%wq0", "%dw_v0", "    ")
        L += ["    vector.store %dw_v0, %w_stage_view[%w_stage_row, %w_stage_col0] : vector<16xf16>, view<128x72xf16>"]
        L += _nvfp4_decode16("dw1", "%wd1", "%wq1", "%dw_v1", "    ")
        L += [
            "    vector.store %dw_v1, %w_stage_view[%w_stage_row, %w_stage_col1] : vector<16xf16>, view<128x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
            "    %wd_off_n = index.add %w_d_base, %step_off_n : offset",
            "    %wd_view_n = buffer.view %w[%wd_off_n] : buffer -> view<2xi8>",
            "    %wd_n = vector.load %wd_view_n[0] : view<2xi8> -> vector<2xi8>",
            "    %wq_off_n = index.add %w_q_base, %step_off_n : offset",
            "    %wq_view_n = buffer.view %w[%wq_off_n] : buffer -> view<16xi8>",
            "    %wq_n = vector.load %wq_view_n[0] : view<16xi8> -> vector<16xi8>",
        ]
        pf_yield = "%x_lo_n, %x_hi_n, %wd_n, %wq_n"
        pf_types = "vector<16xf16>, vector<16xf16>, vector<2xi8>, vector<16xi8>"
    else:
        L += [
            "  %wd_view_p = buffer.view %w[%w_d_base] : buffer -> view<4xi8>",
            "  %wd_p = vector.load %wd_view_p[0] : view<4xi8> -> vector<4xi8>",
            "  %wq0_view_p = buffer.view %w[%w_q0_base] : buffer -> view<16xi8>",
            "  %wq0_p = vector.load %wq0_view_p[0] : view<16xi8> -> vector<16xi8>",
            "  %wq1_view_p = buffer.view %w[%w_q1_base] : buffer -> view<16xi8>",
            "  %wq1_p = vector.load %wq1_view_p[0] : view<16xi8> -> vector<16xi8>",
            f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3, %pf_u4 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_wd = %wd_p : vector<4xi8>, %cr_wq0 = %wq0_p : vector<16xi8>, %cr_wq1 = %wq1_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, vector<4xi8>, vector<16xi8>, vector<16xi8>) {{",
            "    vector.store %cr_xlo, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    scf.if %has_hi64 {",
            "      vector.store %cr_xhi, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    }",
            "    %wd0 = vector.extract %cr_wd[0] : vector<4xi8> -> i8",
            "    %wd1 = vector.extract %cr_wd[1] : vector<4xi8> -> i8",
            "    %wd2 = vector.extract %cr_wd[2] : vector<4xi8> -> i8",
            "    %wd3 = vector.extract %cr_wd[3] : vector<4xi8> -> i8",
            "    %wq0 = vector.slice %cr_wq0[0] : vector<16xi8> -> vector<8xi8>",
            "    %wq1 = vector.slice %cr_wq0[8] : vector<16xi8> -> vector<8xi8>",
            "    %wq2 = vector.slice %cr_wq1[0] : vector<16xi8> -> vector<8xi8>",
            "    %wq3 = vector.slice %cr_wq1[8] : vector<16xi8> -> vector<8xi8>",
        ]
        L += _nvfp4_decode16("dw0", "%wd0", "%wq0", "%dw_v0", "    ")
        L += ["    vector.store %dw_v0, %w_stage_view[%workitem, %c0] : vector<16xf16>, view<256x72xf16>"]
        L += _nvfp4_decode16("dw1", "%wd1", "%wq1", "%dw_v1", "    ")
        L += ["    vector.store %dw_v1, %w_stage_view[%workitem, %c16] : vector<16xf16>, view<256x72xf16>"]
        L += _nvfp4_decode16("dw2", "%wd2", "%wq2", "%dw_v2", "    ")
        L += ["    vector.store %dw_v2, %w_stage_view[%workitem, %c32] : vector<16xf16>, view<256x72xf16>"]
        L += _nvfp4_decode16("dw3", "%wd3", "%wq3", "%dw_v3", "    ")
        L += [
            "    vector.store %dw_v3, %w_stage_view[%workitem, %c48] : vector<16xf16>, view<256x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %block_bytes : index, offset -> offset",
            "    %wd_off_n = index.add %w_d_base, %step_off_n : offset",
            "    %wd_view_n = buffer.view %w[%wd_off_n] : buffer -> view<4xi8>",
            "    %wd_n = vector.load %wd_view_n[0] : view<4xi8> -> vector<4xi8>",
            "    %wq0_off_n = index.add %w_q0_base, %step_off_n : offset",
            "    %wq0_view_n = buffer.view %w[%wq0_off_n] : buffer -> view<16xi8>",
            "    %wq0_n = vector.load %wq0_view_n[0] : view<16xi8> -> vector<16xi8>",
            "    %wq1_off_n = index.add %w_q1_base, %step_off_n : offset",
            "    %wq1_view_n = buffer.view %w[%wq1_off_n] : buffer -> view<16xi8>",
            "    %wq1_n = vector.load %wq1_view_n[0] : view<16xi8> -> vector<16xi8>",
        ]
        pf_yield = "%x_lo_n, %x_hi_n, %wd_n, %wq0_n, %wq1_n"
        pf_types = "vector<16xf16>, vector<16xf16>, vector<4xi8>, vector<16xi8>, vector<16xi8>"

    live_outs = ", ".join(f"%l_{a}" for a in acc_names)
    lo_types = ", ".join(["vector<8xf32>"] * ntiles)
    lo_vars = ", ".join(f"%lo_a{i}" for i in range(ntiles))
    L += [f"    {live_outs} = scf.if %wave_live_hi -> ({acc_types}) {{"]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"      %ll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %lh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
        ]
        for i in range(ntiles):
            L.append(f"      %rw{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x{tile_n}xf16, %w_stage_view_tl> -> vector<8xf16>")
            pl = f"%a{i}l_acc" if ks == 0 else f"%f_a{i}l_{ks-1}"
            ph = f"%a{i}h_acc" if ks == 0 else f"%f_a{i}h_{ks-1}"
            L.append(f"      %f_a{i}l_{ks} = vector.mma %ll_{ks}, %rw{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            L.append(f"      %f_a{i}h_{ks} = vector.mma %lh_{ks}, %rw{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "      scf.yield " + ", ".join(f"%f_{a}_3" for a in acc_names) + f" : {acc_types}",
        "    } else {",
        f"      {lo_vars} = scf.if %wave_live_lo -> ({lo_types}) {{",
    ]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L.append(f"        %sl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>")
        for i in range(ntiles):
            L.append(f"        %srw{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x{tile_n}xf16, %w_stage_view_tl> -> vector<8xf16>")
            pl = f"%a{i}l_acc" if ks == 0 else f"%s_a{i}l_{ks-1}"
            L.append(f"        %s_a{i}l_{ks} = vector.mma %sl_{ks}, %srw{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "        scf.yield " + ", ".join(f"%s_a{i}l_3" for i in range(ntiles)) + f" : {lo_types}",
        "      } else {",
        "        scf.yield " + ", ".join(f"%a{i}l_acc" for i in range(ntiles)) + f" : {lo_types}",
        "      }",
        f"      scf.yield {lo_vars}, " + ", ".join(f"%a{i}h_acc" for i in range(ntiles)) + f" : {acc_types}",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        f"    scf.yield {live_outs}, {pf_yield} : {acc_types}, {pf_types}",
        "  }",
        "  scf.if %wave_live_lo {",
    ]
    for i in range(ntiles):
        L.append(f"    vector.fragment.store<result> %a{i}l_raw, %y_view[%wave_row_lo, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<128x{tile_n}xf32, %y_layout>")
    L += [
        "  }",
        "  scf.if %wave_live_hi {",
    ]
    for i in range(ntiles):
        L.append(f"    vector.fragment.store<result> %a{i}h_raw, %y_view[%wave_row_hi, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<128x{tile_n}xf32, %y_layout>")
    L += [
        "  }",
        "  }",
        "  kernel.return",
        "}",
        "",
    ]
    return "\n".join(L)


def emit_ggu_q8_0_m128() -> str:
    """128x64x64 Q8_0 fused gate+up+act -> direct f16 act16, software-pipelined."""
    L = [
        'kernel.def target(@rdna4) export("g_moe_ggu_q8_0_fused_k2304_rdna4") @g_moe_ggu_q8_0_fused_k2304_rdna4(%rows: index, %n: index) {',
        "  %c1 = index.constant 1 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        "  %c256 = index.constant 256 : index",
        "  %row_tiles = index.div %rows, %c128 : index",
        "  %out_tiles = index.div %n, %c64 : index",
        "  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index",
        "} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, %w: buffer, %w2: buffer, %y: buffer) {",
        "  // MoE grouped GEMM M=128 (128x64x64 Q8_0 gate+up -> f16 act16, pipelined, RHS reuse).",
        "  %rows = index.assume %rows_arg [range(%rows_arg, 128, 131072), multiple_of(%rows_arg, 128)] : index",
        "  %k = index.assume %k_arg [range(%k_arg, 64, 65536), multiple_of(%k_arg, 64)] : index",
        "  %n = index.assume %n_arg [range(%n_arg, 64, 1152), multiple_of(%n_arg, 64)] : index",
        "  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index",
        "  %origin = index.constant 0 : offset",
        "  %two_o = index.constant 2 : offset",
        "  %sixteen_o = index.constant 16 : offset",
        "  %block_bytes = index.constant 34 : offset",
        "  %step_bytes = index.constant 68 : offset",
        "  %c0 = index.constant 0 : index",
        "  %c1 = index.constant 1 : index",
        "  %c2 = index.constant 2 : index",
        "  %c4 = index.constant 4 : index",
        "  %c8 = index.constant 8 : index",
        "  %c16 = index.constant 16 : index",
        "  %c32 = index.constant 32 : index",
        "  %c48 = index.constant 48 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        "  %c0_i32 = scalar.constant 0 : i32",
        "  %cEm1 = scalar.constant 127 : i32",
        "  %row_tile_count = index.div %rows, %c128 : index",
        "  %out_tile_count = index.div %n, %c64 : index",
        "  %row_group_id = kernel.workgroup.id<y> : index",
        "  %out_group_id = kernel.workgroup.id<x> : index",
        "  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index",
        "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index",
        "  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>",
        "  %tile_expert_v = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32",
        "  %tile_expert = kernel.subgroup.broadcast.first %tile_expert_v : i32",
        "  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32",
        "  scf.if %tile_used {",
        "  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32",
        "  %e_b = scalar.assume %e_lo [range(%e_lo, 0, 127)] : i32",
        "  %expert = index.cast %e_b : i32 to index",
        "  %expert_bytes = index.constant 5640192 : offset",
        "  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset",
        "  %workitem_id = kernel.workitem.id<x> : index",
        "  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index",
        "  %subgroup = kernel.subgroup.id : index",
        "  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index",
        "  %wave_row_group = index.div %bounded_subgroup, %c2 : index",
        "  %wave_out_group = index.rem %bounded_subgroup, %c2 : index",
        "  %wave_row_lo_v = index.mul %wave_row_group, %c16 : index",
        "  %wave_row_lo = index.assume %wave_row_lo_v [range(%wave_row_lo_v, 0, 48)] : index",
        "  %wave_row_hi_v = index.add %wave_row_lo, %c64 : index",
        "  %wave_row_hi = index.assume %wave_row_hi_v [range(%wave_row_hi_v, 64, 112)] : index",
        "  %wave_out_v = index.mul %wave_out_group, %c32 : index",
        "  %wave_out0 = index.assume %wave_out_v [range(%wave_out_v, 0, 32)] : index",
        "  %wave_out1 = index.add %wave_out0, %c16 : index",
        "  %row_origin = index.mul %row_group, %c128 : index",
        "  %out_origin = index.mul %out_group, %c64 : index",
        "  %blocks = index.div %k, %c32 : index",
        "  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset",
        "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>",
        "  %y16_view = buffer.view %y[%origin] : buffer -> view<[%rows]x[%n]xf16>",
        "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%xrows]x2xf32>",
        "  %x_stage_bytes = index.constant 18432 : offset",
        "  %w_stage_bytes = index.constant 9216 : offset",
        "  %x_stage = buffer.alloca<workgroup> align(16) %x_stage_bytes : buffer",
        "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        "  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>",
        "  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        "  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x64xf16, %w_stage_view_tl>",
        "  %w2_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        "  %w2_stage_view = buffer.view %w2_stage[%origin] : buffer -> view<64x72xf16>",
        "  %w2_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        "  %w2_stage_view_t = buffer.view %w2_stage[%origin] : buffer -> view<72x64xf16, %w2_stage_view_tl>",
        "  %wave_rows32_v = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32",
        "  %wave_rows32 = kernel.subgroup.broadcast.first %wave_rows32_v : i32",
        "  %wave_rows_b = scalar.assume %wave_rows32 [range(%wave_rows32, 0, 128)] : i32",
        "  %wave_rows = index.cast %wave_rows_b : i32 to index",
        "  %has_hi64 = index.cmp ugt, %wave_rows, %c64 : index",
        "  %wave_live_lo = index.cmp ult, %wave_row_lo, %wave_rows : index",
        "  %wave_live_hi = index.cmp ult, %wave_row_hi, %wave_rows : index",
        "  %x_stage_row_lo = index.div %workitem, %c4 : index",
        "  %x_stage_row_hi = index.add %x_stage_row_lo, %c64 : index",
        "  %x_stage_q = index.rem %workitem, %c4 : index",
        "  %x_stage_col = index.mul %x_stage_q, %c16 : index",
        "  %gx_rm_rows = index.mul %row_tile_count, %c128 : index",
        "  %gx_rm_view = buffer.view %rowmap[%origin] : buffer -> view<[%gx_rm_rows]xi32>",
        "  %x_row_raw_lo = index.add %row_origin, %x_stage_row_lo : index",
        "  %gx_prow_lo = index.assume %x_row_raw_lo [lt(%x_row_raw_lo, %gx_rm_rows)] : index",
        "  %gx_a32_lo = view.load %gx_rm_view[%gx_prow_lo] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_lo = scalar.cmpi sge, %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ac_lo = scalar.maxsi %gx_a32_lo, %c0_i32 : i32",
        "  %gx_ab_lo = scalar.assume %gx_ac_lo [range(%gx_ac_lo, 0, 1048575)] : i32",
        "  %gx_a_lo = index.cast %gx_ab_lo : i32 to index",
        "  %gx_t_lo = index.div %gx_a_lo, %c8 : index",
        "  %bounded_x_row_lo = index.assume %gx_t_lo [lt(%gx_t_lo, %xrows)] : index",
        "  %x_row_raw_hi = index.add %row_origin, %x_stage_row_hi : index",
        "  %gx_prow_hi = index.assume %x_row_raw_hi [lt(%x_row_raw_hi, %gx_rm_rows)] : index",
        "  %gx_a32_hi = view.load %gx_rm_view[%gx_prow_hi] : view<[%gx_rm_rows]xi32> -> i32",
        "  %gx_valid_hi = scalar.cmpi sge, %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ac_hi = scalar.maxsi %gx_a32_hi, %c0_i32 : i32",
        "  %gx_ab_hi = scalar.assume %gx_ac_hi [range(%gx_ac_hi, 0, 1048575)] : i32",
        "  %gx_a_hi = index.cast %gx_ab_hi : i32 to index",
        "  %gx_t_hi = index.div %gx_a_hi, %c8 : index",
        "  %bounded_x_row_hi = index.assume %gx_t_hi [lt(%gx_t_hi, %xrows)] : index",
        "  %up_bytes = index.constant 2820096 : offset",
        "  %w_stage_row = index.div %workitem, %c4 : index",
        "  %w_stage_q = index.rem %workitem, %c4 : index",
        "  %w_stage_col = index.mul %w_stage_q, %c16 : index",
        "  %w_blk = index.div %w_stage_q, %c2 : index",
        "  %w_sub = index.rem %w_stage_q, %c2 : index",
        "  %w_out_row = index.add %out_origin, %w_stage_row : index",
        "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
        "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
        "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
        "  %w_blk_off = index.scale %w_blk, %block_bytes : index, offset -> offset",
        "  %g_d_base = index.add %w_row_origin, %w_blk_off : offset",
        "  %w_sub_off = index.scale %w_sub, %sixteen_o : index, offset -> offset",
        "  %g_q_base0 = index.add %g_d_base, %two_o : offset",
        "  %g_q_base = index.add %g_q_base0, %w_sub_off : offset",
        "  %u_d_base = index.add %g_d_base, %up_bytes : offset",
        "  %u_q_base = index.add %g_q_base, %up_bytes : offset",
        "  %zero8 = vector.splatf 0.0 : vector<8xf32>",
        "  %zero16h = vector.splatf 0.0 : vector<16xf16>",
        "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>",
        "  %k_steps = index.div %k, %c64 : index",
        "  %pf_last = index.sub %k_steps, %c1 : index",
        "  %x_lo_p = vector.load %x_view[%bounded_x_row_lo, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "  %x_hi_p = scf.if %has_hi64 -> (vector<16xf16>) {",
        "    %xhp = vector.load %x_view[%bounded_x_row_hi, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    scf.yield %xhp : vector<16xf16>",
        "  } else {",
        "    scf.yield %zero16h : vector<16xf16>",
        "  }",
        "  %gd_view_p = buffer.view %w[%g_d_base] : buffer -> view<1xf16>",
        "  %gd_p = view.load %gd_view_p[0] : view<1xf16> -> f16",
        "  %gq_view_p = buffer.view %w[%g_q_base] : buffer -> view<16xi8>",
        "  %gq_p = vector.load %gq_view_p[0] : view<16xi8> -> vector<16xi8>",
        "  %ud_view_p = buffer.view %w2[%u_d_base] : buffer -> view<1xf16>",
        "  %ud_p = view.load %ud_view_p[0] : view<1xf16> -> f16",
        "  %uq_view_p = buffer.view %w2[%u_q_base] : buffer -> view<16xi8>",
        "  %uq_p = vector.load %uq_view_p[0] : view<16xi8> -> vector<16xi8>",
    ]
    acc_names = ["g0l", "g1l", "u0l", "u1l", "g0h", "g1h", "u0h", "u1h"]
    acc_outs = ", ".join(f"%{a}_raw" for a in acc_names)
    acc_inits = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in acc_names)
    acc_types = ", ".join(["vector<8xf32>"] * 8)
    live_outs = ", ".join(f"%l_{a}" for a in acc_names)
    L += [
        f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3, %pf_u4, %pf_u5 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_gd = %gd_p : f16, %cr_gq = %gq_p : vector<16xi8>, %cr_ud = %ud_p : f16, %cr_uq = %uq_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, f16, vector<16xi8>, f16, vector<16xi8>) {{",
        "    %x_lo_s = scf.select %gx_valid_lo, %cr_xlo, %zero16h : vector<16xf16>",
        "    vector.store %x_lo_s, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    scf.if %has_hi64 {",
        "      %x_hi_s = scf.select %gx_valid_hi, %cr_xhi, %zero16h : vector<16xf16>",
        "      vector.store %x_hi_s, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
        "    }",
        "    %dg_f = vector.sitofp %cr_gq : vector<16xi8> to vector<16xf16>",
        "    %dg_s = vector.splat %cr_gd : vector<16xf16>",
        "    %dg_v = vector.mulf %dg_f, %dg_s : vector<16xf16>",
        "    vector.store %dg_v, %w_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>",
        "    %du_f = vector.sitofp %cr_uq : vector<16xi8> to vector<16xf16>",
        "    %du_s = vector.splat %cr_ud : vector<16xf16>",
        "    %du_v = vector.mulf %du_f, %du_s : vector<16xf16>",
        "    vector.store %du_v, %w2_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %pf_s1 = index.add %step, %c1 : index",
        "    %pf_m = index.min %pf_s1, %pf_last : index",
        "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
        "    %k_next = index.mul %pf_next, %c64 : index",
        "    %x_col_n = index.add %k_next, %x_stage_col : index",
        "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
        "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "      scf.yield %xhn : vector<16xf16>",
        "    } else {",
        "      scf.yield %zero16h : vector<16xf16>",
        "    }",
        "    %step_off_n = index.scale %pf_next, %step_bytes : index, offset -> offset",
        "    %gd_off_n = index.add %g_d_base, %step_off_n : offset",
        "    %gd_view_n = buffer.view %w[%gd_off_n] : buffer -> view<1xf16>",
        "    %gd_n = view.load %gd_view_n[0] : view<1xf16> -> f16",
        "    %gq_off_n = index.add %g_q_base, %step_off_n : offset",
        "    %gq_view_n = buffer.view %w[%gq_off_n] : buffer -> view<16xi8>",
        "    %gq_n = vector.load %gq_view_n[0] : view<16xi8> -> vector<16xi8>",
        "    %ud_off_n = index.add %u_d_base, %step_off_n : offset",
        "    %ud_view_n = buffer.view %w2[%ud_off_n] : buffer -> view<1xf16>",
        "    %ud_n = view.load %ud_view_n[0] : view<1xf16> -> f16",
        "    %uq_off_n = index.add %u_q_base, %step_off_n : offset",
        "    %uq_view_n = buffer.view %w2[%uq_off_n] : buffer -> view<16xi8>",
        "    %uq_n = vector.load %uq_view_n[0] : view<16xi8> -> vector<16xi8>",
        f"    {live_outs} = scf.if %wave_live_hi -> ({acc_types}) {{",
    ]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"      %ll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %lh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %rg0_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"      %rg1_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"      %ru0_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
            f"      %ru1_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
        ]
        for nm, lh, rh in [
            ("g0l", f"%ll_{ks}", f"%rg0_{ks}"), ("g0h", f"%lh_{ks}", f"%rg0_{ks}"),
            ("g1l", f"%ll_{ks}", f"%rg1_{ks}"), ("g1h", f"%lh_{ks}", f"%rg1_{ks}"),
            ("u0l", f"%ll_{ks}", f"%ru0_{ks}"), ("u0h", f"%lh_{ks}", f"%ru0_{ks}"),
            ("u1l", f"%ll_{ks}", f"%ru1_{ks}"), ("u1h", f"%lh_{ks}", f"%ru1_{ks}"),
        ]:
            prev = f"%{nm}_acc" if ks == 0 else f"%f_{nm}_{ks-1}"
            L.append(f"      %f_{nm}_{ks} = vector.mma {lh}, {rh}, {prev} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "      scf.yield " + ", ".join(f"%f_{a}_3" for a in acc_names) + f" : {acc_types}",
        "    } else {",
        "      %lo_g0, %lo_g1, %lo_u0, %lo_u1 = scf.if %wave_live_lo -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {",
    ]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"        %sl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"        %srg0_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"        %srg1_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w_stage_view_tl> -> vector<8xf16>",
            f"        %sru0_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out0] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
            f"        %sru1_{ks} = vector.fragment.load<rhs> %w2_stage_view_t[{fc}, %wave_out1] shape [%c16, %c16] : view<72x64xf16, %w2_stage_view_tl> -> vector<8xf16>",
        ]
        for nm, rh in [("g0l", f"%srg0_{ks}"), ("g1l", f"%srg1_{ks}"), ("u0l", f"%sru0_{ks}"), ("u1l", f"%sru1_{ks}")]:
            prev = f"%{nm}_acc" if ks == 0 else f"%s_{nm}_{ks-1}"
            L.append(f"        %s_{nm}_{ks} = vector.mma %sl_{ks}, {rh}, {prev} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "        scf.yield %s_g0l_3, %s_g1l_3, %s_u0l_3, %s_u1l_3 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>",
        "      } else {",
        "        scf.yield %g0l_acc, %g1l_acc, %u0l_acc, %u1l_acc : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>",
        "      }",
        f"      scf.yield %lo_g0, %lo_g1, %lo_u0, %lo_u1, %g0h_acc, %g1h_acc, %u0h_acc, %u1h_acc : {acc_types}",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        f"    scf.yield {live_outs}, %x_lo_n, %x_hi_n, %gd_n, %gq_n, %ud_n, %uq_n : {acc_types}, vector<16xf16>, vector<16xf16>, f16, vector<16xi8>, f16, vector<16xi8>",
        "  }",
        "  %inv_view = buffer.view %x_stage[%origin] : buffer -> view<128x16xf32>",
        "  %writer = index.cmp ult, %workitem, %c128 : index",
        "  scf.if %writer {",
        "    %inv_prow_v = index.add %row_origin, %workitem : index",
        "    %inv_prow = index.assume %inv_prow_v [lt(%inv_prow_v, %gx_rm_rows)] : index",
        "    %inv_a32 = view.load %gx_rm_view[%inv_prow] : view<[%gx_rm_rows]xi32> -> i32",
        "    %inv_ac = scalar.maxsi %inv_a32, %c0_i32 : i32",
        "    %inv_ab = scalar.assume %inv_ac [range(%inv_ac, 0, 1048575)] : i32",
        "    %inv_a = index.cast %inv_ab : i32 to index",
        "    %inv_t = index.div %inv_a, %c8 : index",
        "    %inv_row = index.assume %inv_t [lt(%inv_t, %xrows)] : index",
        "    %inverse = view.load %s_view[%inv_row, %c1] : view<[%xrows]x2xf32> -> f32",
        "    %inverse16 = vector.splat %inverse : vector<16xf32>",
        "    %bw = index.assume %workitem [range(%workitem, 0, 127)] : index",
        "    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<128x16xf32>",
        "  }",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  %oa_beta = vector.splatf 4.0 : vector<8xf32>",
        "  %oa_invb2 = vector.splatf 0.5 : vector<8xf32>",
        "  %oa_lb = vector.splatf 25.0 : vector<8xf32>",
        "  %oa_invlb2 = vector.splatf 0.08 : vector<8xf32>",
        "  %oa_two = vector.splatf 2.0 : vector<8xf32>",
        "  %oa_one = vector.splatf 1.0 : vector<8xf32>",
        "  %f16_out_lo = buffer.view %w_stage[%origin] : buffer -> view<32x68xf32>",
        "  %f16_out_hi = buffer.view %w2_stage[%origin] : buffer -> view<32x68xf32>",
        "  %f16_half = index.cmp ult, %wave_row_group, %c2 : index",
        "  %f16_wrow_m = index.rem %wave_row_group, %c2 : index",
        "  %f16_wrow_r = index.mul %f16_wrow_m, %c16 : index",
        "  %f16_wrow = index.assume %f16_wrow_r [range(%f16_wrow_r, 0, 16)] : index",
        "  %f16_st_lo = index.cmp ult, %x_stage_row_lo, %c32 : index",
        "  %f16_gc0 = index.add %out_origin, %x_stage_col : index",
        "  %f16_gc = index.assume %f16_gc0 [lt(%f16_gc0, %n)] : index",
        "  %inv_frag_lo = vector.fragment.load<result> %inv_view[%wave_row_lo, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
        "  %g0ls = vector.mulf %g0l_raw, %inv_frag_lo : vector<8xf32>",
        "  %g1ls = vector.mulf %g1l_raw, %inv_frag_lo : vector<8xf32>",
        "  %u0ls = vector.mulf %u0l_raw, %inv_frag_lo : vector<8xf32>",
        "  %u1ls = vector.mulf %u1l_raw, %inv_frag_lo : vector<8xf32>",
    ]
    L += _act_block("al0", "%g0ls", "%u0ls", "%act0l", "  ")
    L += _act_block("al1", "%g1ls", "%u1ls", "%act1l", "  ")
    L += [
        "  scf.if %f16_half {",
        "    vector.fragment.store<result> %act0l, %f16_out_lo[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    vector.fragment.store<result> %act1l, %f16_out_lo[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "  } else {",
        "    vector.fragment.store<result> %act0l, %f16_out_hi[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    vector.fragment.store<result> %act1l, %f16_out_hi[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "  }",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  %f16_v32_l = scf.if %f16_st_lo -> (vector<16xf32>) {",
        "    %sr_l0 = index.assume %x_stage_row_lo [range(%x_stage_row_lo, 0, 31)] : index",
        "    %ld_l0 = vector.load %f16_out_lo[%sr_l0, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "    scf.yield %ld_l0 : vector<16xf32>",
        "  } else {",
        "    %sr_l1r = index.sub %x_stage_row_lo, %c32 : index",
        "    %sr_l1 = index.assume %sr_l1r [range(%sr_l1r, 0, 31)] : index",
        "    %ld_l1 = vector.load %f16_out_hi[%sr_l1, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "    scf.yield %ld_l1 : vector<16xf32>",
        "  }",
        "  %f16_v16_l = vector.fptrunc %f16_v32_l : vector<16xf32> to vector<16xf16>",
        "  %f16_grl0 = index.add %row_origin, %x_stage_row_lo : index",
        "  %f16_grl = index.assume %f16_grl0 [lt(%f16_grl0, %rows)] : index",
        "  vector.store %f16_v16_l, %y16_view[%f16_grl, %f16_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        "  scf.if %has_hi64 {",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %inv_frag_hi = vector.fragment.load<result> %inv_view[%wave_row_hi, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>",
        "    %g0hs = vector.mulf %g0h_raw, %inv_frag_hi : vector<8xf32>",
        "    %g1hs = vector.mulf %g1h_raw, %inv_frag_hi : vector<8xf32>",
        "    %u0hs = vector.mulf %u0h_raw, %inv_frag_hi : vector<8xf32>",
        "    %u1hs = vector.mulf %u1h_raw, %inv_frag_hi : vector<8xf32>",
    ]
    L += _act_block("ah0", "%g0hs", "%u0hs", "%act0h", "    ")
    L += _act_block("ah1", "%g1hs", "%u1hs", "%act1h", "    ")
    L += [
        "    scf.if %f16_half {",
        "      vector.fragment.store<result> %act0h, %f16_out_lo[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "      vector.fragment.store<result> %act1h, %f16_out_lo[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    } else {",
        "      vector.fragment.store<result> %act0h, %f16_out_hi[%f16_wrow, %wave_out0] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "      vector.fragment.store<result> %act1h, %f16_out_hi[%f16_wrow, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<32x68xf32>",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "    %f16_v32_h = scf.if %f16_st_lo -> (vector<16xf32>) {",
        "      %sr_h0 = index.assume %x_stage_row_lo [range(%x_stage_row_lo, 0, 31)] : index",
        "      %ld_h0 = vector.load %f16_out_lo[%sr_h0, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "      scf.yield %ld_h0 : vector<16xf32>",
        "    } else {",
        "      %sr_h1r = index.sub %x_stage_row_lo, %c32 : index",
        "      %sr_h1 = index.assume %sr_h1r [range(%sr_h1r, 0, 31)] : index",
        "      %ld_h1 = vector.load %f16_out_hi[%sr_h1, %x_stage_col] : view<32x68xf32> -> vector<16xf32>",
        "      scf.yield %ld_h1 : vector<16xf32>",
        "    }",
        "    %f16_v16_h = vector.fptrunc %f16_v32_h : vector<16xf32> to vector<16xf16>",
        "    %f16_grh0 = index.add %row_origin, %x_stage_row_hi : index",
        "    %f16_grh = index.assume %f16_grh0 [lt(%f16_grh0, %rows)] : index",
        "    vector.store %f16_v16_h, %y16_view[%f16_grh, %f16_gc] : vector<16xf16>, view<[%rows]x[%n]xf16>",
        "  }",
        "  }",
        "  kernel.return",
        "}",
        "",
    ]
    return "\n".join(L)


def emit_gdn_q8_0_m128(tile_n: int = 64) -> str:
    """128x{64,128}x64 Q8_0 MoE down projection -> f32 ys_sorted, software-pipelined."""
    assert tile_n in (64, 128)
    ntiles = tile_n // 32
    w_stage_bytes = tile_n * 72 * 2

    L = [
        'kernel.def target(@rdna4) export("g_moe_gdn_q8_0_k1152_rdna4") @g_moe_gdn_q8_0_k1152_rdna4(%rows: index, %n: index) {',
        "  %c1 = index.constant 1 : index",
        f"  %ctn = index.constant {tile_n} : index",
        "  %c128 = index.constant 128 : index",
        "  %c256 = index.constant 256 : index",
        "  %row_tiles = index.div %rows, %c128 : index",
        "  %out_tiles = index.div %n, %ctn : index",
        "  kernel.launch.config workgroups(%out_tiles, %row_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index",
        "} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %xrows_arg: index, %tile_info: buffer, %rowmap: buffer, %x: buffer, %scales: buffer, %w: buffer, %y: buffer) {",
        f"  // MoE grouped GEMM M=128 (128x{tile_n}x64 Q8_0 down, pipelined, RHS reuse).",
        "  %rows = index.assume %rows_arg [range(%rows_arg, 128, 131072), multiple_of(%rows_arg, 128)] : index",
        "  %k = index.assume %k_arg [range(%k_arg, 64, 65536), multiple_of(%k_arg, 64)] : index",
        f"  %n = index.assume %n_arg [range(%n_arg, {tile_n}, 2304), multiple_of(%n_arg, {tile_n})] : index",
        "  %xrows = index.assume %xrows_arg [range(%xrows_arg, 1, 1048576)] : index",
        "  %origin = index.constant 0 : offset",
        "  %two_o = index.constant 2 : offset",
        "  %four_o = index.constant 4 : offset",
        "  %sixteen_o = index.constant 16 : offset",
        "  %eighteen_o = index.constant 18 : offset",
        "  %block_bytes = index.constant 34 : offset",
        "  %step_bytes = index.constant 68 : offset",
        "  %c0 = index.constant 0 : index",
        "  %c1 = index.constant 1 : index",
        "  %c2 = index.constant 2 : index",
        "  %c4 = index.constant 4 : index",
        "  %c16 = index.constant 16 : index",
        "  %c32 = index.constant 32 : index",
        "  %c48 = index.constant 48 : index",
        "  %c64 = index.constant 64 : index",
        "  %c128 = index.constant 128 : index",
        f"  %ctn = index.constant {tile_n} : index",
        "  %c0_i32 = scalar.constant 0 : i32",
        "  %cEm1 = scalar.constant 127 : i32",
        "  %row_tile_count = index.div %rows, %c128 : index",
        "  %out_tile_count = index.div %n, %ctn : index",
        "  %row_group_id = kernel.workgroup.id<y> : index",
        "  %out_group_id = kernel.workgroup.id<x> : index",
        "  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index",
        "  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index",
        "  %ti_view = buffer.view %tile_info[%origin] : buffer -> view<[%row_tile_count]x2xi32>",
        "  %tile_expert_v = view.load %ti_view[%row_group, 0] : view<[%row_tile_count]x2xi32> -> i32",
        "  %tile_expert = kernel.subgroup.broadcast.first %tile_expert_v : i32",
        "  %tile_used = scalar.cmpi sge, %tile_expert, %c0_i32 : i32",
        "  scf.if %tile_used {",
        "  %e_lo = scalar.minsi %tile_expert, %cEm1 : i32",
        "  %e_b = scalar.assume %e_lo [range(%e_lo, 0, 127)] : i32",
        "  %expert = index.cast %e_b : i32 to index",
        "  %expert_bytes = index.constant 2820096 : offset",
        "  %expert_off = index.scale %expert, %expert_bytes : index, offset -> offset",
        "  %workitem_id = kernel.workitem.id<x> : index",
        "  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index",
        "  %subgroup = kernel.subgroup.id : index",
        "  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index",
        "  %wave_row_group = index.div %bounded_subgroup, %c2 : index",
        "  %wave_out_group = index.rem %bounded_subgroup, %c2 : index",
        "  %wave_row_lo_v = index.mul %wave_row_group, %c16 : index",
        "  %wave_row_lo = index.assume %wave_row_lo_v [range(%wave_row_lo_v, 0, 48)] : index",
        "  %wave_row_hi_v = index.add %wave_row_lo, %c64 : index",
        "  %wave_row_hi = index.assume %wave_row_hi_v [range(%wave_row_hi_v, 64, 112)] : index",
        f"  %c_wout = index.constant {tile_n // 2} : index",
        "  %wave_out_v = index.mul %wave_out_group, %c_wout : index",
        f"  %wave_out0 = index.assume %wave_out_v [range(%wave_out_v, 0, {tile_n // 2})] : index",
    ]
    for i in range(1, ntiles):
        L.append(f"  %c_wo{i} = index.constant {i * 16} : index")
        L.append(f"  %wave_out{i} = index.add %wave_out0, %c_wo{i} : index")
    L += [
        "  %row_origin = index.mul %row_group, %c128 : index",
        "  %out_origin = index.mul %out_group, %ctn : index",
        "  %blocks = index.div %k, %c32 : index",
        "  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset",
        "  %x_view = buffer.view %x[%origin] : buffer -> view<[%xrows]x[%k]xf16>",
        "  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>",
        "  %tile_first = index.madd %row_origin, %n, %out_origin : index",
        "  %tile_offset = index.scale %tile_first, %four_o : index, offset -> offset",
        f"  %y_view = buffer.view %y[%tile_offset] : buffer -> view<128x{tile_n}xf32, %y_layout>",
        "  %x_stage_bytes = index.constant 18432 : offset",
        f"  %w_stage_bytes = index.constant {w_stage_bytes} : offset",
        "  %x_stage = buffer.alloca<workgroup> align(16) %x_stage_bytes : buffer",
        "  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>",
        "  %w_stage = buffer.alloca<workgroup> align(16) %w_stage_bytes : buffer",
        f"  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<{tile_n}x72xf16>",
        "  %w_stage_view_tl = encoding.layout.strided [1, 72] : encoding<layout>",
        f"  %w_stage_view_t = buffer.view %w_stage[%origin] : buffer -> view<72x{tile_n}xf16, %w_stage_view_tl>",
        "  %wave_rows32_v = view.load %ti_view[%row_group, 1] : view<[%row_tile_count]x2xi32> -> i32",
        "  %wave_rows32 = kernel.subgroup.broadcast.first %wave_rows32_v : i32",
        "  %wave_rows_b = scalar.assume %wave_rows32 [range(%wave_rows32, 0, 128)] : i32",
        "  %wave_rows = index.cast %wave_rows_b : i32 to index",
        "  %has_hi64 = index.cmp ugt, %wave_rows, %c64 : index",
        "  %wave_live_lo = index.cmp ult, %wave_row_lo, %wave_rows : index",
        "  %wave_live_hi = index.cmp ult, %wave_row_hi, %wave_rows : index",
        "  %x_stage_row_lo = index.div %workitem, %c4 : index",
        "  %x_stage_row_hi = index.add %x_stage_row_lo, %c64 : index",
        "  %x_stage_q = index.rem %workitem, %c4 : index",
        "  %x_stage_col = index.mul %x_stage_q, %c16 : index",
        "  %x_row_lo = index.add %row_origin, %x_stage_row_lo : index",
        "  %bounded_x_row_lo = index.assume %x_row_lo [lt(%x_row_lo, %xrows)] : index",
        "  %x_row_hi = index.add %row_origin, %x_stage_row_hi : index",
        "  %bounded_x_row_hi = index.assume %x_row_hi [lt(%x_row_hi, %xrows)] : index",
        "  %zero8 = vector.splatf 0.0 : vector<8xf32>",
        "  %zero16h = vector.splatf 0.0 : vector<16xf16>",
        "  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>",
        "  %k_steps = index.div %k, %c64 : index",
        "  %pf_last = index.sub %k_steps, %c1 : index",
        "  %x_lo_p = vector.load %x_view[%bounded_x_row_lo, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "  %x_hi_p = scf.if %has_hi64 -> (vector<16xf16>) {",
        "    %xhp = vector.load %x_view[%bounded_x_row_hi, %x_stage_col] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
        "    scf.yield %xhp : vector<16xf16>",
        "  } else {",
        "    scf.yield %zero16h : vector<16xf16>",
        "  }",
    ]
    acc_names = [f"a{i}l" for i in range(ntiles)] + [f"a{i}h" for i in range(ntiles)]
    acc_outs = ", ".join(f"%{a}_raw" for a in acc_names)
    acc_inits = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in acc_names)
    acc_types = ", ".join(["vector<8xf32>"] * (2 * ntiles))

    if tile_n == 64:
        L += [
            "  %w_stage_row = index.div %workitem, %c4 : index",
            "  %w_stage_q = index.rem %workitem, %c4 : index",
            "  %w_stage_col = index.mul %w_stage_q, %c16 : index",
            "  %w_blk = index.div %w_stage_q, %c2 : index",
            "  %w_sub = index.rem %w_stage_q, %c2 : index",
            "  %w_out_row = index.add %out_origin, %w_stage_row : index",
            "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
            "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
            "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
            "  %w_blk_off = index.scale %w_blk, %block_bytes : index, offset -> offset",
            "  %w_d_base = index.add %w_row_origin, %w_blk_off : offset",
            "  %w_sub_off = index.scale %w_sub, %sixteen_o : index, offset -> offset",
            "  %w_q_base0 = index.add %w_d_base, %two_o : offset",
            "  %w_q_base = index.add %w_q_base0, %w_sub_off : offset",
            "  %wd_view_p = buffer.view %w[%w_d_base] : buffer -> view<1xf16>",
            "  %wd_p = view.load %wd_view_p[0] : view<1xf16> -> f16",
            "  %wq_view_p = buffer.view %w[%w_q_base] : buffer -> view<16xi8>",
            "  %wq_p = vector.load %wq_view_p[0] : view<16xi8> -> vector<16xi8>",
            f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_wd = %wd_p : f16, %cr_wq = %wq_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, f16, vector<16xi8>) {{",
            "    vector.store %cr_xlo, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    scf.if %has_hi64 {",
            "      vector.store %cr_xhi, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    }",
            "    %dw_f = vector.sitofp %cr_wq : vector<16xi8> to vector<16xf16>",
            "    %dw_s = vector.splat %cr_wd : vector<16xf16>",
            "    %dw_v = vector.mulf %dw_f, %dw_s : vector<16xf16>",
            "    vector.store %dw_v, %w_stage_view[%w_stage_row, %w_stage_col] : vector<16xf16>, view<64x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %step_bytes : index, offset -> offset",
            "    %wd_off_n = index.add %w_d_base, %step_off_n : offset",
            "    %wd_view_n = buffer.view %w[%wd_off_n] : buffer -> view<1xf16>",
            "    %wd_n = view.load %wd_view_n[0] : view<1xf16> -> f16",
            "    %wq_off_n = index.add %w_q_base, %step_off_n : offset",
            "    %wq_view_n = buffer.view %w[%wq_off_n] : buffer -> view<16xi8>",
            "    %wq_n = vector.load %wq_view_n[0] : view<16xi8> -> vector<16xi8>",
        ]
        pf_yield = "%x_lo_n, %x_hi_n, %wd_n, %wq_n"
        pf_types = "vector<16xf16>, vector<16xf16>, f16, vector<16xi8>"
    else:
        L += [
            "  %w_stage_row = index.div %workitem, %c2 : index",
            "  %w_stage_half = index.rem %workitem, %c2 : index",
            "  %w_stage_col0 = index.mul %w_stage_half, %c32 : index",
            "  %w_stage_col1 = index.add %w_stage_col0, %c16 : index",
            "  %w_out_row = index.add %out_origin, %w_stage_row : index",
            "  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index",
            "  %w_row_origin0 = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset",
            "  %w_row_origin = index.add %w_row_origin0, %expert_off : offset",
            "  %w_half_off = index.scale %w_stage_half, %block_bytes : index, offset -> offset",
            "  %w_d_base = index.add %w_row_origin, %w_half_off : offset",
            "  %w_q0_base = index.add %w_d_base, %two_o : offset",
            "  %w_q1_base = index.add %w_d_base, %eighteen_o : offset",
            "  %wd_view_p = buffer.view %w[%w_d_base] : buffer -> view<1xf16>",
            "  %wd_p = view.load %wd_view_p[0] : view<1xf16> -> f16",
            "  %wq0_view_p = buffer.view %w[%w_q0_base] : buffer -> view<16xi8>",
            "  %wq0_p = vector.load %wq0_view_p[0] : view<16xi8> -> vector<16xi8>",
            "  %wq1_view_p = buffer.view %w[%w_q1_base] : buffer -> view<16xi8>",
            "  %wq1_p = vector.load %wq1_view_p[0] : view<16xi8> -> vector<16xi8>",
            f"  {acc_outs}, %pf_u0, %pf_u1, %pf_u2, %pf_u3, %pf_u4 = scf.for %step = [%c0 to %k_steps step %c1]({acc_inits}, %cr_xlo = %x_lo_p : vector<16xf16>, %cr_xhi = %x_hi_p : vector<16xf16>, %cr_wd = %wd_p : f16, %cr_wq0 = %wq0_p : vector<16xi8>, %cr_wq1 = %wq1_p : vector<16xi8>) -> ({acc_types}, vector<16xf16>, vector<16xf16>, f16, vector<16xi8>, vector<16xi8>) {{",
            "    vector.store %cr_xlo, %x_stage_view[%x_stage_row_lo, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    scf.if %has_hi64 {",
            "      vector.store %cr_xhi, %x_stage_view[%x_stage_row_hi, %x_stage_col] : vector<16xf16>, view<128x72xf16>",
            "    }",
            "    %dw_s = vector.splat %cr_wd : vector<16xf16>",
            "    %dw0_f = vector.sitofp %cr_wq0 : vector<16xi8> to vector<16xf16>",
            "    %dw0_v = vector.mulf %dw0_f, %dw_s : vector<16xf16>",
            "    vector.store %dw0_v, %w_stage_view[%w_stage_row, %w_stage_col0] : vector<16xf16>, view<128x72xf16>",
            "    %dw1_f = vector.sitofp %cr_wq1 : vector<16xi8> to vector<16xf16>",
            "    %dw1_v = vector.mulf %dw1_f, %dw_s : vector<16xf16>",
            "    vector.store %dw1_v, %w_stage_view[%w_stage_row, %w_stage_col1] : vector<16xf16>, view<128x72xf16>",
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
            "    %pf_s1 = index.add %step, %c1 : index",
            "    %pf_m = index.min %pf_s1, %pf_last : index",
            "    %pf_next = index.assume %pf_m [lt(%pf_m, %k_steps)] : index",
            "    %k_next = index.mul %pf_next, %c64 : index",
            "    %x_col_n = index.add %k_next, %x_stage_col : index",
            "    %x_lo_n = vector.load %x_view[%bounded_x_row_lo, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "    %x_hi_n = scf.if %has_hi64 -> (vector<16xf16>) {",
            "      %xhn = vector.load %x_view[%bounded_x_row_hi, %x_col_n] : view<[%xrows]x[%k]xf16> -> vector<16xf16>",
            "      scf.yield %xhn : vector<16xf16>",
            "    } else {",
            "      scf.yield %zero16h : vector<16xf16>",
            "    }",
            "    %step_off_n = index.scale %pf_next, %step_bytes : index, offset -> offset",
            "    %wd_off_n = index.add %w_d_base, %step_off_n : offset",
            "    %wd_view_n = buffer.view %w[%wd_off_n] : buffer -> view<1xf16>",
            "    %wd_n = view.load %wd_view_n[0] : view<1xf16> -> f16",
            "    %wq0_off_n = index.add %w_q0_base, %step_off_n : offset",
            "    %wq0_view_n = buffer.view %w[%wq0_off_n] : buffer -> view<16xi8>",
            "    %wq0_n = vector.load %wq0_view_n[0] : view<16xi8> -> vector<16xi8>",
            "    %wq1_off_n = index.add %w_q1_base, %step_off_n : offset",
            "    %wq1_view_n = buffer.view %w[%wq1_off_n] : buffer -> view<16xi8>",
            "    %wq1_n = vector.load %wq1_view_n[0] : view<16xi8> -> vector<16xi8>",
        ]
        pf_yield = "%x_lo_n, %x_hi_n, %wd_n, %wq0_n, %wq1_n"
        pf_types = "vector<16xf16>, vector<16xf16>, f16, vector<16xi8>, vector<16xi8>"

    live_outs = ", ".join(f"%l_{a}" for a in acc_names)
    lo_types = ", ".join(["vector<8xf32>"] * ntiles)
    lo_vars = ", ".join(f"%lo_a{i}" for i in range(ntiles))
    L += [f"    {live_outs} = scf.if %wave_live_hi -> ({acc_types}) {{"]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L += [
            f"      %ll_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
            f"      %lh_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_hi, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>",
        ]
        for i in range(ntiles):
            L.append(f"      %rw{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x{tile_n}xf16, %w_stage_view_tl> -> vector<8xf16>")
            pl = f"%a{i}l_acc" if ks == 0 else f"%f_a{i}l_{ks-1}"
            ph = f"%a{i}h_acc" if ks == 0 else f"%f_a{i}h_{ks-1}"
            L.append(f"      %f_a{i}l_{ks} = vector.mma %ll_{ks}, %rw{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
            L.append(f"      %f_a{i}h_{ks} = vector.mma %lh_{ks}, %rw{i}_{ks}, {ph} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "      scf.yield " + ", ".join(f"%f_{a}_3" for a in acc_names) + f" : {acc_types}",
        "    } else {",
        f"      {lo_vars} = scf.if %wave_live_lo -> ({lo_types}) {{",
    ]
    for ks, fc in enumerate(["%c0", "%c16", "%c32", "%c48"]):
        L.append(f"        %sl_{ks} = vector.fragment.load<lhs> %x_stage_view[%wave_row_lo, {fc}] shape [%c16, %c16] : view<128x72xf16> -> vector<8xf16>")
        for i in range(ntiles):
            L.append(f"        %srw{i}_{ks} = vector.fragment.load<rhs> %w_stage_view_t[{fc}, %wave_out{i}] shape [%c16, %c16] : view<72x{tile_n}xf16, %w_stage_view_tl> -> vector<8xf16>")
            pl = f"%a{i}l_acc" if ks == 0 else f"%s_a{i}l_{ks-1}"
            L.append(f"        %s_a{i}l_{ks} = vector.mma %sl_{ks}, %srw{i}_{ks}, {pl} : vector<8xf16>, vector<8xf16>, vector<8xf32>")
    L += [
        "        scf.yield " + ", ".join(f"%s_a{i}l_3" for i in range(ntiles)) + f" : {lo_types}",
        "      } else {",
        "        scf.yield " + ", ".join(f"%a{i}l_acc" for i in range(ntiles)) + f" : {lo_types}",
        "      }",
        f"      scf.yield {lo_vars}, " + ", ".join(f"%a{i}h_acc" for i in range(ntiles)) + f" : {acc_types}",
        "    }",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        f"    scf.yield {live_outs}, {pf_yield} : {acc_types}, {pf_types}",
        "  }",
        "  scf.if %wave_live_lo {",
    ]
    for i in range(ntiles):
        L.append(f"    vector.fragment.store<result> %a{i}l_raw, %y_view[%wave_row_lo, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<128x{tile_n}xf32, %y_layout>")
    L += [
        "  }",
        "  scf.if %wave_live_hi {",
    ]
    for i in range(ntiles):
        L.append(f"    vector.fragment.store<result> %a{i}h_raw, %y_view[%wave_row_hi, %wave_out{i}] shape [%c16, %c16] : vector<8xf32>, view<128x{tile_n}xf32, %y_layout>")
    L += [
        "  }",
        "  }",
        "  kernel.return",
        "}",
        "",
    ]
    return "\n".join(L)


def transform(src: str, ggu_n: int = 128, ggu_shared_w: bool = False, gdn_n: int = 128) -> str:
    out = patch_bucket_and_host(src)
    ggu_nv = emit_ggu_nvfp4_m128() if ggu_n == 64 else emit_ggu_nvfp4_m128_n128(shared_w=ggu_shared_w)
    for name, body in [
        ("g_moe_ggu_nvfp4_fused_sc_k2304_rdna4", ggu_nv),
        ("g_moe_gdn_nvfp4_k1152_rdna4", emit_gdn_nvfp4_m128(gdn_n)),
        ("g_moe_ggu_q8_0_fused_k2304_rdna4", emit_ggu_q8_0_m128()),
        ("g_moe_gdn_q8_0_k1152_rdna4", emit_gdn_q8_0_m128(min(gdn_n, 128))),
    ]:
        s, e = kernel_span(out, name, target_rdna4=True)
        out = out[:s] + body + out[e:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--ggu-n", type=int, choices=[64, 128], default=128)
    ap.add_argument("--ggu-shared-w", action="store_true")
    ap.add_argument("--gdn-n", type=int, choices=[64, 128, 256], default=128)
    args = ap.parse_args()

    with open(args.in_path, "r") as f:
        src = f.read()
    out = transform(src, ggu_n=args.ggu_n, ggu_shared_w=args.ggu_shared_w, gdn_n=args.gdn_n)
    with open(args.out_path, "w") as f:
        f.write(out)
    sw = "+sharedW" if (args.ggu_n == 128 and args.ggu_shared_w) else ""
    print(f"rewrote MoE M=128: g_moe_bucket(128) + ggu(128x{args.ggu_n}{sw}) + gdn(128x{args.gdn_n}) [NVFP4 + Q8_0]")


if __name__ == "__main__":
    main()
