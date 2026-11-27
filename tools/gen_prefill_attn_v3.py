#!/usr/bin/env python3
"""Bit-identical prefill attention optimizations for gfx12 (RDNA4):

  5A (g_ink_attention_flash2_ts2_rdna4 — full-attention prefill):
    - Native f16 int8-KV dequant (vector.sitofp i8->f16 + vector.mulf f16) and
      f16 scale carry across the prefetch loop (halves dequant VALU).
    - Hoist the KV page-table base out of the key-tile loop (broadcast to SGPRs)
      and replace 256 per-lane @g_kv_offset8 calls per 32-key tile with 2
      wave-uniform scalar page-table lookups (all full-attention call sites pass
      sliding=0 so tile_first is always a multiple of 32).
    - Broadcast wave-uniform staging dim / liveness and delete dead st_live_p1.
    - Compiles to 215 VGPRs / 7 waves/SIMD / 18,944 B LDS / 0 spills.

  5B (g_ink_attention_flash2_swa_rdna4 — sliding-window prefill):
    - Key-major 3-pass K/V staging with single per-lane address and ring-mod
      (index.rem 1040) calculation per 32-key tile (zero V^T LDS bank conflicts,
      3x fewer integer mod divisions).
    - Drops VGPRs from 226 -> 216 (jumps to 7 waves/SIMD) with unchanged
      18,944 B LDS and 0 spills.

  5C (g_ink_attention_flash2_partial_rdna4 — non-16-aligned tail/continuation tiles):
    - Key-major 3-pass staging + hoisted 2-page scalar KV lookup per 32-key tile
      + native f16 int8-KV dequant in both the hot loop and peeled last tile.

usage: gen_prefill_attn_v3.py IN.loom OUT.loom [--no-ts2] [--no-swa] [--no-partial]
"""
import argparse
import sys


def kernel_span(src: str, name: str):
    start = src.index(f'kernel.def target(@rdna4) export("{name}")')
    end = src.index("\n  kernel.return\n}\n", start) + len("\n  kernel.return\n}\n")
    return start, end


def sub1(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise ValueError(f"expected {count} x {old[:80]!r}, found {n}")
    return text.replace(old, new)


def _uniform_map_setup() -> str:
    """Load %origins[%row * 40][4] (i64) and broadcast lo/hi i32 halves so %v3_map is SGPR."""
    return (
        "  %v3_c40 = index.constant 40 : offset\n"
        "  %v3_rec_off = index.scale %row, %v3_c40 : index, offset -> offset\n"
        "  %v3_rec32 = buffer.view %origins[%v3_rec_off] : buffer -> view<10xi32>\n"
        "  %v3_mlo_r = view.load %v3_rec32[8] : view<10xi32> -> i32\n"
        "  %v3_mhi_r = view.load %v3_rec32[9] : view<10xi32> -> i32\n"
        "  %v3_mlo = kernel.subgroup.broadcast %v3_mlo_r from %c0_i32 : i32, i32\n"
        "  %v3_mhi = kernel.subgroup.broadcast %v3_mhi_r from %c0_i32 : i32, i32\n"
        "  %v3_mlo64 = scalar.extui %v3_mlo : i32 to i64\n"
        "  %v3_mhi64 = scalar.extui %v3_mhi : i32 to i64\n"
        "  %v3_sh32 = scalar.constant 32 : i64\n"
        "  %v3_mhi_s = scalar.shli %v3_mhi64, %v3_sh32 : i64\n"
        "  %v3_map64 = scalar.ori %v3_mhi_s, %v3_mlo64 : i64\n"
        "  %v3_map_b = scalar.assume %v3_map64 [range(%v3_map64, 0, 9223372036854775807)] : i64\n"
        "  %v3_map_off = index.cast %v3_map_b : i64 to offset\n"
        "  %v3_map = buffer.view %origins[%v3_map_off] : buffer -> view<[%blocks]xi32>\n"
        "  %v3_last_blk = index.sub %blocks, %c1 : index\n"
        "  %v3_plane_base = index.mul %k_plane, %pool : index\n"
        "  %v3_c1024 = index.constant 1024 : offset\n"
        "  %v3_c32 = index.constant 32 : index\n"
    )


# ---------------------------------------------------------------------------
# 5A: g_ink_attention_flash2_ts2_rdna4
# ---------------------------------------------------------------------------
def _ts2_load_block(p: str, tile_first: str, indent: str, extra: str | None = None) -> str:
    """Hoisted 2-page scalar KV lookup + raw int8/f16-scale prefetch load."""
    ind = indent
    L = []
    if extra:
        L.append(f"{ind}%{p}_tile_ok = scalar.andi %v3_live, {extra} : i1")
        tile_ok = f"%{p}_tile_ok"
    else:
        tile_ok = "%v3_live"
    L += [
        f"{ind}%{p}_k, %{p}_v, %{p}_sk, %{p}_sv = scf.if {tile_ok} -> (vector<16xi8>, vector<16xi8>, f16, f16) {{",
        f"{ind}  %{p}_p0_raw = index.div {tile_first}, %c16 : index",
        f"{ind}  %{p}_p0_cl = index.min %{p}_p0_raw, %v3_last_blk : index",
        f"{ind}  %{p}_p0 = index.assume %{p}_p0_cl [range(%{p}_p0_cl, 0, %v3_last_blk), lt(%{p}_p0_cl, %blocks)] : index",
        f"{ind}  %{p}_p1_raw = index.add %{p}_p0, %c1 : index",
        f"{ind}  %{p}_p1_cl = index.min %{p}_p1_raw, %v3_last_blk : index",
        f"{ind}  %{p}_p1 = index.assume %{p}_p1_cl [range(%{p}_p1_cl, 0, %v3_last_blk), lt(%{p}_p1_cl, %blocks)] : index",
        f"{ind}  %{p}_phys0_r = view.load %v3_map[%{p}_p0] : view<[%blocks]xi32> -> i32",
        f"{ind}  %{p}_phys1_r = view.load %v3_map[%{p}_p1] : view<[%blocks]xi32> -> i32",
        f"{ind}  %{p}_phys0 = kernel.subgroup.broadcast %{p}_phys0_r from %c0_i32 : i32, i32",
        f"{ind}  %{p}_phys1 = kernel.subgroup.broadcast %{p}_phys1_r from %c0_i32 : i32, i32",
        f"{ind}  %{p}_kp_raw = index.add {tile_first}, %st_key_p0 : index",
        f"{ind}  %{p}_ok = index.cmp ule, %{p}_kp_raw, %pos_last : index",
        f"{ind}  %{p}_ki, %{p}_vi, %{p}_ski, %{p}_svi = scf.if %{p}_ok -> (vector<16xi8>, vector<16xi8>, f16, f16) {{",
        f"{ind}    %{p}_phys32 = scf.select %hi_half, %{p}_phys1, %{p}_phys0 : i32",
        f"{ind}    %{p}_phys_b = scalar.assume %{p}_phys32 [range(%{p}_phys32, 0, 134217727)] : i32",
        f"{ind}    %{p}_phys = index.cast %{p}_phys_b : i32 to index",
        f"{ind}    %{p}_pbase = index.mul %{p}_phys, %c16 : index",
        f"{ind}    %{p}_slot = index.add %{p}_pbase, %lane_col : index",
        f"{ind}    %{p}_pslot = index.add %v3_plane_base, %{p}_slot : index",
        f"{ind}    %{p}_koff = index.scale %{p}_pslot, %pos_bytes : index, offset -> offset",
        f"{ind}    %{p}_voff = index.add %{p}_koff, %plane_stride : offset",
        f"{ind}    %{p}_kc = buffer.view %arena[%{p}_koff] : buffer -> view<8x128xi8>",
        f"{ind}    %{p}_kso = index.add %{p}_koff, %v3_c1024 : offset",
        f"{ind}    %{p}_ks = buffer.view %arena[%{p}_kso] : buffer -> view<8x4xf16>",
        f"{ind}    %{p}_vc = buffer.view %arena[%{p}_voff] : buffer -> view<8x128xi8>",
        f"{ind}    %{p}_vso = index.add %{p}_voff, %v3_c1024 : offset",
        f"{ind}    %{p}_vs = buffer.view %arena[%{p}_vso] : buffer -> view<8x4xf16>",
        f"{ind}    %{p}_kq = vector.load %{p}_kc[%kv, %v3_dim] : view<8x128xi8> -> vector<16xi8>",
        f"{ind}    %{p}_ks16 = view.load %{p}_ks[%kv, %v3_bb] : view<8x4xf16> -> f16",
        f"{ind}    %{p}_vq = vector.load %{p}_vc[%kv, %v3_dim] : view<8x128xi8> -> vector<16xi8>",
        f"{ind}    %{p}_vs16 = view.load %{p}_vs[%kv, %v3_bb] : view<8x4xf16> -> f16",
        f"{ind}    scf.yield %{p}_kq, %{p}_vq, %{p}_ks16, %{p}_vs16 : vector<16xi8>, vector<16xi8>, f16, f16",
        f"{ind}  }} else {{",
        f"{ind}    scf.yield %pf_zq, %pf_zq, %v3_zero_h, %v3_zero_h : vector<16xi8>, vector<16xi8>, f16, f16",
        f"{ind}  }}",
        f"{ind}  scf.yield %{p}_ki, %{p}_vi, %{p}_ski, %{p}_svi : vector<16xi8>, vector<16xi8>, f16, f16",
        f"{ind}}} else {{",
        f"{ind}  scf.yield %pf_zq, %pf_zq, %v3_zero_h, %v3_zero_h : vector<16xi8>, vector<16xi8>, f16, f16",
        f"{ind}}}",
    ]
    return "\n".join(L) + "\n"


def rewrite_ts2(k: str) -> str:
    dead_p1_defs = (
        "  %st_add_p1 = index.constant 192 : index\n"
        "  %st_slot_p1 = index.add %workitem, %st_add_p1 : index\n"
        "  %st_live_p1 = scalar.constant false : i1\n"
        "  %st_slot_cl_p1 = index.min %st_slot_p1, %c_slots_less : index\n"
        "  %st_key_val_p1 = index.div %st_slot_cl_p1, %ctpk : index\n"
        "  %st_key_p1 = index.assume %st_key_val_p1 [range(%st_key_val_p1, 0, 31)] : index\n"
        "  %st_seg_p1 = index.rem %st_slot_cl_p1, %ctpk : index\n"
        "  %st_dim_val_p1 = index.mul %st_seg_p1, %cdpt : index\n"
        "  %st_dim_p1 = index.assume %st_dim_val_p1 [range(%st_dim_val_p1, 0, 112)] : index\n"
    )
    hoisted = (
        _uniform_map_setup()
        + "  %v3_zero_h = scalar.constant 0.0 : f16\n"
        + "  %v3_dim_i = index.cast %st_dim_p0 : index to i32\n"
        + "  %v3_dim_b = kernel.subgroup.broadcast %v3_dim_i from %c0_i32 : i32, i32\n"
        + "  %v3_dim_u = index.cast %v3_dim_b : i32 to index\n"
        + "  %v3_dim = index.assume %v3_dim_u [range(%v3_dim_u, 0, 112)] : index\n"
        + "  %v3_bv = index.div %v3_dim, %v3_c32 : index\n"
        + "  %v3_bb = index.assume %v3_bv [range(%v3_bv, 0, 3)] : index\n"
        + "  %v3_live_i = scf.select %st_live_p0, %c1_i32m, %c0_i32 : i32\n"
        + "  %v3_live_b = kernel.subgroup.broadcast %v3_live_i from %c0_i32 : i32, i32\n"
        + "  %v3_live = scalar.cmpi ne, %v3_live_b, %c0_i32 : i32\n"
    )
    k = sub1(k, dead_p1_defs, hoisted)

    pf0_start = k.index("  %pf0_kp_raw = index.add %key_low, %st_key_p0 : index\n")
    pf0_end = k.index("  %m_final, %l_final, ", pf0_start)
    k = k[:pf0_start] + _ts2_load_block("pf0", "%key_low", "  ") + k[pf0_end:]

    k = sub1(
        k,
        "%psk = %pf0_sk : f32, %psv = %pf0_sv : f32) -> (f32, f32, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xi8>, vector<16xi8>, f32, f32) {",
        "%psk = %pf0_sk : f16, %psv = %pf0_sv : f16) -> (f32, f32, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xi8>, vector<16xi8>, f16, f16) {",
    )
    k = sub1(
        k,
        "    scf.yield %m_new, %l_new, %on0, %on1, %on2, %on3, %on4, %on5, %on6, %on7, %pfn_k, %pfn_v, %pfn_sk, %pfn_sv : f32, f32, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xi8>, vector<16xi8>, f32, f32\n",
        "    scf.yield %m_new, %l_new, %on0, %on1, %on2, %on3, %on4, %on5, %on6, %on7, %pfn_k, %pfn_v, %pfn_sk, %pfn_sv : f32, f32, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<16xi8>, vector<16xi8>, f16, f16\n",
    )

    old_stage = (
        "    scf.if %st_live_p0 {\n"
        "    %pf_kfl = vector.sitofp %pk : vector<16xi8> to vector<16xf32>\n"
        "    %pf_ksp = vector.splat %psk : vector<16xf32>\n"
        "    %pf_kd = vector.mulf %pf_kfl, %pf_ksp : vector<16xf32>\n"
        "    %kvals_p0 = vector.fptrunc %pf_kd : vector<16xf32> to vector<16xf16>\n"
        "    %pf_vfl = vector.sitofp %pv : vector<16xi8> to vector<16xf32>\n"
        "    %pf_vsp = vector.splat %psv : vector<16xf32>\n"
        "    %pf_vd = vector.mulf %pf_vfl, %pf_vsp : vector<16xf32>\n"
        "    %vvals_p0 = vector.fptrunc %pf_vd : vector<16xf32> to vector<16xf16>\n"
        "    vector.store %kvals_p0, %k_view[%st_key_p0, %st_dim_p0] : vector<16xf16>, view<32x136xf16>\n"
        "    scf.for %j_p0 = [%c0 to %cdpt step %c1] unroll {\n"
        "      %vj_p0 = vector.extract %vvals_p0[%j_p0] : vector<16xf16> -> f16\n"
        "      %vt_row_val_p0 = index.add %st_dim_p0, %j_p0 : index\n"
        "      %vt_row_p0 = index.assume %vt_row_val_p0 [range(%vt_row_val_p0, 0, 127)] : index\n"
        "      view.store %vj_p0, %vt_view[%vt_row_p0, %st_key_p0] : f16, view<128x40xf16>\n"
        "    }\n"
        "    }\n"
    )
    new_stage = (
        "    scf.if %v3_live {\n"
        "      %pf_kh = vector.sitofp %pk : vector<16xi8> to vector<16xf16>\n"
        "      %pf_ksp = vector.splat %psk : vector<16xf16>\n"
        "      %kvals_p0 = vector.mulf %pf_kh, %pf_ksp : vector<16xf16>\n"
        "      %pf_vh = vector.sitofp %pv : vector<16xi8> to vector<16xf16>\n"
        "      %pf_vsp = vector.splat %psv : vector<16xf16>\n"
        "      %vvals_p0 = vector.mulf %pf_vh, %pf_vsp : vector<16xf16>\n"
        "      vector.store %kvals_p0, %k_view[%st_key_p0, %v3_dim] : vector<16xf16>, view<32x136xf16>\n"
        "      scf.for %j_p0 = [%c0 to %cdpt step %c1] unroll {\n"
        "        %vj_p0 = vector.extract %vvals_p0[%j_p0] : vector<16xf16> -> f16\n"
        "        %vt_row_val_p0 = index.add %v3_dim, %j_p0 : index\n"
        "        %vt_row_p0 = index.assume %vt_row_val_p0 [range(%vt_row_val_p0, 0, 127)] : index\n"
        "        view.store %vj_p0, %vt_view[%vt_row_p0, %st_key_p0] : f16, view<128x40xf16>\n"
        "      }\n"
        "    }\n"
    )
    k = sub1(k, old_stage, new_stage)

    p1_start = k.index("    scf.if %st_live_p1 {\n")
    p1_end = k.index("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n", p1_start)
    k = k[:p1_start] + k[p1_end:]

    pfn_start = k.index("    %pfn_kp_raw = index.add %pf_tfn, %st_key_p0 : index\n")
    pfn_end = k.index("    %tf32 = index.cast %tile_first : index to i32\n", pfn_start)
    k = k[:pfn_start] + _ts2_load_block("pfn", "%pf_tfn", "    ", "%pf_has") + k[pfn_end:]
    return k


# ---------------------------------------------------------------------------
# 5B: g_ink_attention_flash2_swa_rdna4
# ---------------------------------------------------------------------------
def _swa_km_stage_block(suffix: str, tile_first: str) -> str:
    """Key-major 3-pass SWA K/V staging with single per-lane address / rem-1040 calculation."""
    s = suffix
    L = [
        f"    %km_kp{s} = index.add {tile_first}, %lane : index",
        f"    %km_valid{s} = index.cmp ule, %km_kp{s}, %pos_last : index",
        f"    scf.if %km_valid{s} {{",
        f"      %km_kpb{s} = index.assume %km_kp{s} [range(%km_kp{s}, 0, 1048575)] : index",
        f"      %km_in{s} = index.cmp uge, %km_kpb{s}, %sp_start : index",
        f"      scf.if %km_in{s} {{",
        f"        %km_rel{s} = index.sub %km_kpb{s}, %sp_start : index",
        f"        %km_tkv{s} = index.add %sp_first, %km_rel{s} : index",
        f"        %km_tk{s} = index.assume %km_tkv{s} [range(%km_tkv{s}, 0, 8191), lt(%km_tkv{s}, %tokens)] : index",
    ]
    for p, dim in ((0, "%km_dim0"), (1, "%km_dim1"), (2, "%km_dim2")):
        ind = "          " if p == 2 else "        "
        if p == 2:
            L.append("        scf.if %km_live2 {")
        L += [
            f"{ind}%km_kc{p}{s} = vector.load %swa_cur_v[%km_tk{s}, %c0, %kv, {dim}] : view<[%tokens]x2x8x128xf16> -> vector<16xf16>",
            f"{ind}%km_vc{p}{s} = vector.load %swa_cur_v[%km_tk{s}, %c1, %kv, {dim}] : view<[%tokens]x2x8x128xf16> -> vector<16xf16>",
            f"{ind}vector.store %km_kc{p}{s}, %k_view[%lane, {dim}] : vector<16xf16>, view<32x136xf16>",
            f"{ind}scf.for %km_jc{p}{s} = [%c0 to %cdpt step %c1] unroll {{",
            f"{ind}  %km_vjc{p}{s} = vector.extract %km_vc{p}{s}[%km_jc{p}{s}] : vector<16xf16> -> f16",
            f"{ind}  %km_vrc{p}{s} = index.add {dim}, %km_jc{p}{s} : index",
            f"{ind}  %km_vrbc{p}{s} = index.assume %km_vrc{p}{s} [range(%km_vrc{p}{s}, 0, 127)] : index",
            f"{ind}  view.store %km_vjc{p}{s}, %vt_view[%km_vrbc{p}{s}, %lane] : f16, view<128x40xf16>",
            f"{ind}}}",
        ]
        if p == 2:
            L.append("        }")
    L += [
        f"      }} else {{",
        f"        %km_rp{s} = index.rem %km_kpb{s}, %cWIN_mod : index",
        f"        %km_rpo{s} = index.scale %km_rp{s}, %pos_bytes : index, offset -> offset",
        f"        %km_ko{s} = index.add %ring_rd, %km_rpo{s} : offset",
        f"        %km_vo{s} = index.add %km_ko{s}, %ring_v_stride : offset",
        f"        %km_kv{s} = buffer.view %arena[%km_ko{s}] : buffer -> view<8x128xf16>",
        f"        %km_vv{s} = buffer.view %arena[%km_vo{s}] : buffer -> view<8x128xf16>",
    ]
    for p, dim in ((0, "%km_dim0"), (1, "%km_dim1"), (2, "%km_dim2")):
        ind = "          " if p == 2 else "        "
        if p == 2:
            L.append("        scf.if %km_live2 {")
        L += [
            f"{ind}%km_kr{p}{s} = vector.load %km_kv{s}[%kv, {dim}] : view<8x128xf16> -> vector<16xf16>",
            f"{ind}%km_vr{p}{s} = vector.load %km_vv{s}[%kv, {dim}] : view<8x128xf16> -> vector<16xf16>",
            f"{ind}vector.store %km_kr{p}{s}, %k_view[%lane, {dim}] : vector<16xf16>, view<32x136xf16>",
            f"{ind}scf.for %km_jr{p}{s} = [%c0 to %cdpt step %c1] unroll {{",
            f"{ind}  %km_vjr{p}{s} = vector.extract %km_vr{p}{s}[%km_jr{p}{s}] : vector<16xf16> -> f16",
            f"{ind}  %km_vrr{p}{s} = index.add {dim}, %km_jr{p}{s} : index",
            f"{ind}  %km_vrbr{p}{s} = index.assume %km_vrr{p}{s} [range(%km_vrr{p}{s}, 0, 127)] : index",
            f"{ind}  view.store %km_vjr{p}{s}, %vt_view[%km_vrbr{p}{s}, %lane] : f16, view<128x40xf16>",
            f"{ind}}}",
        ]
        if p == 2:
            L.append("        }")
    L += [
        f"      }}",
        f"    }} else {{",
        f"      vector.store %kz, %k_view[%lane, %km_dim0] : vector<16xf16>, view<32x136xf16>",
        f"      vector.store %kz, %k_view[%lane, %km_dim1] : vector<16xf16>, view<32x136xf16>",
        f"      scf.for %km_jz0{s} = [%c0 to %cdpt step %c1] unroll {{",
        f"        %km_z0{s} = index.add %km_dim0, %km_jz0{s} : index",
        f"        %km_zb0{s} = index.assume %km_z0{s} [range(%km_z0{s}, 0, 127)] : index",
        f"        view.store %km_zh, %vt_view[%km_zb0{s}, %lane] : f16, view<128x40xf16>",
        f"        %km_z1{s} = index.add %km_dim1, %km_jz0{s} : index",
        f"        %km_zb1{s} = index.assume %km_z1{s} [range(%km_z1{s}, 0, 127)] : index",
        f"        view.store %km_zh, %vt_view[%km_zb1{s}, %lane] : f16, view<128x40xf16>",
        f"      }}",
        f"      scf.if %km_live2 {{",
        f"        vector.store %kz, %k_view[%lane, %km_dim2] : vector<16xf16>, view<32x136xf16>",
        f"        scf.for %km_jz2{s} = [%c0 to %cdpt step %c1] unroll {{",
        f"          %km_z2{s} = index.add %km_dim2, %km_jz2{s} : index",
        f"          %km_zb2{s} = index.assume %km_z2{s} [range(%km_z2{s}, 0, 127)] : index",
        f"          view.store %km_zh, %vt_view[%km_zb2{s}, %lane] : f16, view<128x40xf16>",
        f"        }}",
        f"      }}",
        f"    }}\n",
    ]
    return "\n".join(L)


def rewrite_swa(k: str) -> str:
    pre = (
        "  %km_zh = scalar.constant 0.0 : f16\n"
        "  %km_wu_i = index.cast %wave : index to i32\n"
        "  %km_wu_b = kernel.subgroup.broadcast %km_wu_i from %c0_i32 : i32, i32\n"
        "  %km_wu = index.cast %km_wu_b : i32 to index\n"
        "  %km_d0_v = index.mul %km_wu, %c16 : index\n"
        "  %km_dim0 = index.assume %km_d0_v [range(%km_d0_v, 0, 112)] : index\n"
        "  %km_c48 = index.constant 48 : index\n"
        "  %km_d1_v = index.add %km_dim0, %km_c48 : index\n"
        "  %km_dim1 = index.assume %km_d1_v [range(%km_d1_v, 0, 112)] : index\n"
        "  %km_live2 = index.cmp ult, %km_wu, %c2 : index\n"
        "  %km_w2 = index.min %km_wu, %c1 : index\n"
        "  %km_d2_a = index.mul %km_w2, %c16 : index\n"
        "  %km_c96 = index.constant 96 : index\n"
        "  %km_d2_v = index.add %km_d2_a, %km_c96 : index\n"
        "  %km_dim2 = index.assume %km_d2_v [range(%km_d2_v, 0, 112)] : index\n"
    )
    k = sub1(k, "  %kt_hot = index.sub %key_tiles, %c1 : index\n", pre + "  %kt_hot = index.sub %key_tiles, %c1 : index\n")
    for suffix, tf in (("", "%tile_first"), ("_pt", "%tile_first_pt")):
        st_start = k.index(f"    %key_pos_p0{suffix} = index.add {tf}, %st_key_p0 : index\n")
        st_end = k.index("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n", st_start)
        k = k[:st_start] + _swa_km_stage_block(suffix, tf) + k[st_end:]
    return k


# ---------------------------------------------------------------------------
# 5C: g_ink_attention_flash2_partial_rdna4
# ---------------------------------------------------------------------------
def _partial_km_stage_block(suffix: str, tile_first: str) -> str:
    """Key-major 3-pass staging + 2 scalar page lookups + native f16 dequant."""
    s = suffix
    L = [
        f"    %pk_p0_raw{s} = index.div {tile_first}, %c16 : index",
        f"    %pk_p0_cl{s} = index.min %pk_p0_raw{s}, %v3_last_blk : index",
        f"    %pk_p0{s} = index.assume %pk_p0_cl{s} [range(%pk_p0_cl{s}, 0, %v3_last_blk), lt(%pk_p0_cl{s}, %blocks)] : index",
        f"    %pk_p1_raw{s} = index.add %pk_p0{s}, %c1 : index",
        f"    %pk_p1_cl{s} = index.min %pk_p1_raw{s}, %v3_last_blk : index",
        f"    %pk_p1{s} = index.assume %pk_p1_cl{s} [range(%pk_p1_cl{s}, 0, %v3_last_blk), lt(%pk_p1_cl{s}, %blocks)] : index",
        f"    %pk_phys0_r{s} = view.load %v3_map[%pk_p0{s}] : view<[%blocks]xi32> -> i32",
        f"    %pk_phys1_r{s} = view.load %v3_map[%pk_p1{s}] : view<[%blocks]xi32> -> i32",
        f"    %pk_phys0{s} = kernel.subgroup.broadcast %pk_phys0_r{s} from %c0_i32 : i32, i32",
        f"    %pk_phys1{s} = kernel.subgroup.broadcast %pk_phys1_r{s} from %c0_i32 : i32, i32",
        f"    %pk_kp{s} = index.add {tile_first}, %lane : index",
        f"    %pk_valid{s} = index.cmp ule, %pk_kp{s}, %pos_hi : index",
        f"    scf.if %pk_valid{s} {{",
        f"      %pk_phys32{s} = scf.select %hi_half, %pk_phys1{s}, %pk_phys0{s} : i32",
        f"      %pk_phys_b{s} = scalar.assume %pk_phys32{s} [range(%pk_phys32{s}, 0, 134217727)] : i32",
        f"      %pk_phys{s} = index.cast %pk_phys_b{s} : i32 to index",
        f"      %pk_pbase{s} = index.mul %pk_phys{s}, %c16 : index",
        f"      %pk_slot{s} = index.add %pk_pbase{s}, %lane_col : index",
        f"      %pk_pslot{s} = index.add %v3_plane_base, %pk_slot{s} : index",
        f"      %pk_koff{s} = index.scale %pk_pslot{s}, %pos_bytes : index, offset -> offset",
        f"      %pk_voff{s} = index.add %pk_koff{s}, %plane_stride : offset",
        f"      %pk_kc{s} = buffer.view %arena[%pk_koff{s}] : buffer -> view<8x128xi8>",
        f"      %pk_kso{s} = index.add %pk_koff{s}, %v3_c1024 : offset",
        f"      %pk_ks{s} = buffer.view %arena[%pk_kso{s}] : buffer -> view<8x4xf16>",
        f"      %pk_vc{s} = buffer.view %arena[%pk_voff{s}] : buffer -> view<8x128xi8>",
        f"      %pk_vso{s} = index.add %pk_voff{s}, %v3_c1024 : offset",
        f"      %pk_vs{s} = buffer.view %arena[%pk_vso{s}] : buffer -> view<8x4xf16>",
    ]
    for p, dim, bb in ((0, "%km_dim0", "%km_bb0"), (1, "%km_dim1", "%km_bb1"), (2, "%km_dim2", "%km_bb2")):
        ind = "        " if p == 2 else "      "
        if p == 2:
            L.append("      scf.if %km_live2 {")
        L += [
            f"{ind}%pk_kq{p}{s} = vector.load %pk_kc{s}[%kv, {dim}] : view<8x128xi8> -> vector<16xi8>",
            f"{ind}%pk_ks16_{p}{s} = view.load %pk_ks{s}[%kv, {bb}] : view<8x4xf16> -> f16",
            f"{ind}%pk_kh{p}{s} = vector.sitofp %pk_kq{p}{s} : vector<16xi8> to vector<16xf16>",
            f"{ind}%pk_ksp{p}{s} = vector.splat %pk_ks16_{p}{s} : vector<16xf16>",
            f"{ind}%pk_kd{p}{s} = vector.mulf %pk_kh{p}{s}, %pk_ksp{p}{s} : vector<16xf16>",
            f"{ind}vector.store %pk_kd{p}{s}, %k_view[%lane, {dim}] : vector<16xf16>, view<32x136xf16>",
            f"{ind}%pk_vq{p}{s} = vector.load %pk_vc{s}[%kv, {dim}] : view<8x128xi8> -> vector<16xi8>",
            f"{ind}%pk_vs16_{p}{s} = view.load %pk_vs{s}[%kv, {bb}] : view<8x4xf16> -> f16",
            f"{ind}%pk_vh{p}{s} = vector.sitofp %pk_vq{p}{s} : vector<16xi8> to vector<16xf16>",
            f"{ind}%pk_vsp{p}{s} = vector.splat %pk_vs16_{p}{s} : vector<16xf16>",
            f"{ind}%pk_vd{p}{s} = vector.mulf %pk_vh{p}{s}, %pk_vsp{p}{s} : vector<16xf16>",
            f"{ind}scf.for %pk_j{p}{s} = [%c0 to %cdpt step %c1] unroll {{",
            f"{ind}  %pk_vj{p}{s} = vector.extract %pk_vd{p}{s}[%pk_j{p}{s}] : vector<16xf16> -> f16",
            f"{ind}  %pk_vr{p}{s} = index.add {dim}, %pk_j{p}{s} : index",
            f"{ind}  %pk_vrb{p}{s} = index.assume %pk_vr{p}{s} [range(%pk_vr{p}{s}, 0, 127)] : index",
            f"{ind}  view.store %pk_vj{p}{s}, %vt_view[%pk_vrb{p}{s}, %lane] : f16, view<128x40xf16>",
            f"{ind}}}",
        ]
        if p == 2:
            L.append("      }")
    L += [
        f"    }} else {{",
        f"      vector.store %kz, %k_view[%lane, %km_dim0] : vector<16xf16>, view<32x136xf16>",
        f"      vector.store %kz, %k_view[%lane, %km_dim1] : vector<16xf16>, view<32x136xf16>",
        f"      scf.for %pk_jz0{s} = [%c0 to %cdpt step %c1] unroll {{",
        f"        %pk_z0{s} = index.add %km_dim0, %pk_jz0{s} : index",
        f"        %pk_zb0{s} = index.assume %pk_z0{s} [range(%pk_z0{s}, 0, 127)] : index",
        f"        view.store %km_zh, %vt_view[%pk_zb0{s}, %lane] : f16, view<128x40xf16>",
        f"        %pk_z1{s} = index.add %km_dim1, %pk_jz0{s} : index",
        f"        %pk_zb1{s} = index.assume %pk_z1{s} [range(%pk_z1{s}, 0, 127)] : index",
        f"        view.store %km_zh, %vt_view[%pk_zb1{s}, %lane] : f16, view<128x40xf16>",
        f"      }}",
        f"      scf.if %km_live2 {{",
        f"        vector.store %kz, %k_view[%lane, %km_dim2] : vector<16xf16>, view<32x136xf16>",
        f"        scf.for %pk_jz2{s} = [%c0 to %cdpt step %c1] unroll {{",
        f"          %pk_z2{s} = index.add %km_dim2, %pk_jz2{s} : index",
        f"          %pk_zb2{s} = index.assume %pk_z2{s} [range(%pk_z2{s}, 0, 127)] : index",
        f"          view.store %km_zh, %vt_view[%pk_zb2{s}, %lane] : f16, view<128x40xf16>",
        f"        }}",
        f"      }}",
        f"    }}\n",
    ]
    return "\n".join(L)


def rewrite_partial(k: str) -> str:
    hoisted = (
        _uniform_map_setup()
        + "  %km_zh = scalar.constant 0.0 : f16\n"
        + "  %km_wu_i = index.cast %wave : index to i32\n"
        + "  %km_wu_b = kernel.subgroup.broadcast %km_wu_i from %c0_i32 : i32, i32\n"
        + "  %km_wu = index.cast %km_wu_b : i32 to index\n"
        + "  %km_d0_v = index.mul %km_wu, %c16 : index\n"
        + "  %km_dim0 = index.assume %km_d0_v [range(%km_d0_v, 0, 112)] : index\n"
        + "  %km_bv0 = index.div %km_dim0, %v3_c32 : index\n"
        + "  %km_bb0 = index.assume %km_bv0 [range(%km_bv0, 0, 3)] : index\n"
        + "  %km_c48 = index.constant 48 : index\n"
        + "  %km_d1_v = index.add %km_dim0, %km_c48 : index\n"
        + "  %km_dim1 = index.assume %km_d1_v [range(%km_d1_v, 0, 112)] : index\n"
        + "  %km_bv1 = index.div %km_dim1, %v3_c32 : index\n"
        + "  %km_bb1 = index.assume %km_bv1 [range(%km_bv1, 0, 3)] : index\n"
        + "  %km_live2 = index.cmp ult, %km_wu, %c2 : index\n"
        + "  %km_w2 = index.min %km_wu, %c1 : index\n"
        + "  %km_d2_a = index.mul %km_w2, %c16 : index\n"
        + "  %km_c96 = index.constant 96 : index\n"
        + "  %km_d2_v = index.add %km_d2_a, %km_c96 : index\n"
        + "  %km_dim2 = index.assume %km_d2_v [range(%km_d2_v, 0, 112)] : index\n"
        + "  %km_bv2 = index.div %km_dim2, %v3_c32 : index\n"
        + "  %km_bb2 = index.assume %km_bv2 [range(%km_bv2, 0, 3)] : index\n"
    )
    k = sub1(
        k,
        "  %kt_hot = index.sub %key_tiles, %c1 : index\n",
        hoisted + "  %kt_hot = index.sub %key_tiles, %c1 : index\n",
    )

    for suffix, tf in (("", "%tile_first"), ("_pt", "%tile_first_pt")):
        st_start = k.index(f"    %key_pos_p0{suffix} = index.add {tf}, %st_key_p0 : index\n")
        st_end = k.index(
            "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n",
            st_start,
        )
        k = k[:st_start] + _partial_km_stage_block(suffix, tf) + k[st_end:]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--no-ts2", action="store_true", help="skip flash2_ts2_rdna4")
    ap.add_argument("--no-swa", action="store_true", help="skip flash2_swa_rdna4")
    ap.add_argument("--no-partial", action="store_true", help="skip flash2_partial_rdna4")
    a = ap.parse_args()

    src = open(a.inp).read()
    did = []
    if not a.no_ts2:
        s, e = kernel_span(src, "g_ink_attention_flash2_ts2_rdna4")
        src = src[:s] + rewrite_ts2(src[s:e]) + src[e:]
        did.append("ts2(f16+2page)")
    if not a.no_swa:
        s, e = kernel_span(src, "g_ink_attention_flash2_swa_rdna4")
        src = src[:s] + rewrite_swa(src[s:e]) + src[e:]
        did.append("swa(km)")
    if not a.no_partial:
        s, e = kernel_span(src, "g_ink_attention_flash2_partial_rdna4")
        src = src[:s] + rewrite_partial(src[s:e]) + src[e:]
        did.append("partial(km+f16+2page)")

    open(a.out, "w").write(src)
    print("rewrote flash2 v3:", ", ".join(did), file=sys.stderr)


if __name__ == "__main__":
    main()
