#!/usr/bin/env python3
"""Rewrite g_ink_attention_flash2_ts2_rdna4 (prefill attention, gfx12) so one workgroup covers NSUB 16-token query tiles.

Stock: a workgroup covers 2 adjacent tiles (32 queries) x 3 query heads of one KV head = 6 waves. Each 32-key K/V tile it
stages into LDS (int8 dequant, V transposed) is therefore used by only 32 queries. With NSUB=4 (12 waves, 384 lanes) or
NSUB=8 (24 waves) the same staging serves 64 / 128 queries.

Per query nothing changes: same key tiles in the same order, same online-softmax steps. Waves whose tile is earlier in
the group iterate a few more fully masked key tiles (alpha = exp(0) = 1, p = 0), which the stock kernel already does for
its first sub-tile. So the output is expected to be bit-identical to the stock kernel.

Tile ownership generalises the stock pair rule: the workgroup of a listed tile works if the tile is at slot 0 of its
aligned group of NSUB tiles (position / 16 mod NSUB) or starts a run; it covers the following tiles while each one
continues the previous (g_tile_adjacent) and stays inside the aligned group.

usage: gen_flash2_multi.py IN.loom OUT.loom [--nsub 4|8] [--key-major] [--uniform] [--prefetch]
"""
import argparse, re, sys

NAME = "g_ink_attention_flash2_ts2_rdna4"


def kernel_span(src):
    start = src.index(f'kernel.def target(@rdna4) export("{NAME}")')
    end = src.index("\n  kernel.return\n}\n", start) + len("\n  kernel.return\n}\n")
    return start, end


def sub1(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise ValueError(f"expected {count} x {old[:70]!r}, found {n}")
    return text.replace(old, new)


def rewrite(k, nsub, key_major=False):
    lanes = 96 * nsub
    # launch config + lane / wave ranges
    k = sub1(k, "%clanes = index.constant 192 : index", f"%clanes = index.constant {lanes} : index")
    k = sub1(k, "[range(%workitem_id, 0, 191)]", f"[range(%workitem_id, 0, {lanes - 1})]")
    k = sub1(k, "[range(%wave_id, 0, 5)]", f"[range(%wave_id, 0, {3 * nsub - 1})]")
    # aligned group of NSUB tiles
    k = sub1(k, "  %group_slot = index.rem %pos_tile, %c2 : index\n",
             f"  %ms_nsub = index.constant {nsub} : index\n  %group_slot = index.rem %pos_tile, %ms_nsub : index\n")

    # ownership chain: replaces the stock pair logic (adj1 .. pos_last)
    old_start = "  %adj1 = func.call @g_tile_adjacent(%info, %tokens, %anchor, %false_top) : (buffer, index, index, i1) -> (i1)\n"
    old_end = "  %pos_last = scf.select %live1, %pos_last1, %pos_last0 : index\n"
    a = k.index(old_start); b = k.index(old_end) + len(old_end)
    old = k[a:b]
    for tok in ("%live1 = scalar.andi %first_slot, %adj1", "%use_sub1", "%first_token = index.assume", "%pos0 = scf.select"):
        if tok not in old:
            raise ValueError(f"pair block changed: no {tok!r}")
    L = []
    L.append("  %ms_true = scalar.constant true : i1")
    L.append("  %live0c = scalar.constant true : i1")
    prev_live, prev_first = "%live0c", "%anchor"
    for j in range(1, nsub):
        L.append(f"  %ms_lim{j} = index.constant {nsub - j} : index")
        L.append(f"  %ms_room{j} = index.cmp ult, %group_slot, %ms_lim{j} : index")
        L.append(f"  %ms_adj{j} = func.call @g_tile_adjacent(%info, %tokens, {prev_first}, %false_top) : (buffer, index, index, i1) -> (i1)")
        L.append(f"  %ms_a{j} = scalar.andi {prev_live}, %ms_adj{j} : i1")
        L.append(f"  %live{j} = scalar.andi %ms_a{j}, %ms_room{j} : i1")
        if j < nsub - 1:   # anchor of tile j, kept in range when not live (only feeds the next adjacency test)
            L.append(f"  %ms_off{j} = index.constant {16 * j} : index")
            L.append(f"  %ms_fv{j} = index.add %anchor, %ms_off{j} : index")
            L.append(f"  %ms_fs{j} = scf.select %live{j}, %ms_fv{j}, %anchor : index")
            L.append(f"  %ms_f{j} = index.assume %ms_fs{j} [range(%ms_fs{j}, 0, 8191), lt(%ms_fs{j}, %tokens)] : index")
            prev_first = f"%ms_f{j}"
        prev_live = f"%live{j}"
    # this wave's tile
    L.append("  %mine_s0 = scalar.constant true : i1")
    cur = "%mine_s0"
    for j in range(1, nsub):
        L.append(f"  %ms_c{j} = index.constant {j} : index")
        L.append(f"  %ms_is{j} = index.cmp eq, %sub, %ms_c{j} : index")
        L.append(f"  %mine_s{j} = scf.select %ms_is{j}, %live{j}, {cur} : i1")
        cur = f"%mine_s{j}"
    L.append(f"  %mine = scalar.andi {cur}, %ms_true : i1")
    L.append("  %ms_not0 = index.cmp ne, %sub, %c0 : index")
    L.append("  %use_sub = scalar.andi %ms_not0, %mine : i1")
    L.append("  %ms_sub16 = index.mul %sub, %c16 : index")
    L.append("  %ms_firstv = index.add %anchor, %ms_sub16 : index")
    L.append("  %first_sel = scf.select %use_sub, %ms_firstv, %anchor : index")
    L.append("  %first_token = index.assume %first_sel [range(%first_sel, 0, 8191), lt(%first_sel, %tokens)] : index")
    L.append("  %ms_possub = index.add %pos_a, %ms_sub16 : index")
    L.append("  %pos0 = scf.select %use_sub, %ms_possub, %pos_a : index")
    # last query position of the whole group: pos_a + 16 * (covered tiles) - 1
    L.append("  %pos_last_0 = index.add %pos_a, %c15 : index")
    cur = "%pos_last_0"
    for j in range(1, nsub):
        L.append(f"  %ms_pl{j} = index.constant {16 * j + 15} : index")
        L.append(f"  %ms_plv{j} = index.add %pos_a, %ms_pl{j} : index")
        nxt = "%pos_last" if j == nsub - 1 else f"%pos_last_{j}"
        L.append(f"  {nxt} = scf.select %live{j}, %ms_plv{j}, {cur} : index")
        cur = nxt
    k = k[:a] + "\n".join(L) + "\n" + k[b:]

    # later uses of the stock names
    for gone in ("%use_sub1", "%pos_last0", "%pos_last1", "%first1_v", "%is_sub1"):
        if re.search(re.escape(gone) + r"\b", k):
            raise ValueError(f"{gone} still used after the pair block")

    # staging: 256 slots (32 keys x 8 chunks) now fit one pass; guard lanes >= 256, drop pass 1
    k = sub1(k, "  %st_slot_p0 = index.add %workitem, %st_add_p0 : index\n  %st_key_val_p0 = index.div %st_slot_p0, %ctpk : index\n",
             "  %st_slot_p0 = index.add %workitem, %st_add_p0 : index\n"
             "  %st_live_p0 = index.cmp ult, %st_slot_p0, %c_slots : index\n"
             "  %st_slot_cl_p0 = index.min %st_slot_p0, %c_slots_less : index\n"
             "  %st_key_val_p0 = index.div %st_slot_cl_p0, %ctpk : index\n")
    k = sub1(k, "  %st_seg_p0 = index.rem %st_slot_p0, %ctpk : index\n", "  %st_seg_p0 = index.rem %st_slot_cl_p0, %ctpk : index\n")
    k = sub1(k, "    %key_pos_p0 = index.add %tile_first, %st_key_p0 : index\n",
             "    scf.if %st_live_p0 {\n    %key_pos_p0 = index.add %tile_first, %st_key_p0 : index\n")
    k = sub1(k, "    scf.if %st_live_p1 {\n", "    }\n    scf.if %st_live_p1 {\n")
    k = sub1(k, "  %st_live_p1 = index.cmp ult, %st_slot_p1, %c_slots : index\n", "  %st_live_p1 = scalar.constant false : i1\n")
    if key_major:
        # Stock slot order is chunk-major (8 consecutive lanes = 8 dim chunks of one key). The transposed V stores then
        # put a wave's lanes on V^T rows 16 dims apart, which all fall in the same LDS bank. Key-major order (32
        # consecutive lanes = 32 keys of one chunk) writes 32 adjacent keys of one V^T row instead. Same LDS contents.
        k = sub1(k, "  %st_key_val_p0 = index.div %st_slot_cl_p0, %ctpk : index\n",
                 "  %st_keys32 = index.constant 32 : index\n  %st_key_val_p0 = index.rem %st_slot_cl_p0, %st_keys32 : index\n")
        k = sub1(k, "  %st_seg_p0 = index.rem %st_slot_cl_p0, %ctpk : index\n", "  %st_seg_p0 = index.div %st_slot_cl_p0, %st_keys32 : index\n")
    return k


def load_block(p, tile_first, extra=None):
    """Global loads (raw int8 K/V chunk + f32 scales) of this lane's staging slot for the key tile starting at TILE_FIRST.
    Invalid slots (past the last query position, or lanes >= 256) give zeros, which dequantize to +0.0 like the stock kz."""
    L = [f"    %{p}_kp_raw = index.add {tile_first}, %st_key_p0 : index",
         f"    %{p}_ok0 = index.cmp ule, %{p}_kp_raw, %pos_last : index",
         f"    %{p}_ok1 = scalar.andi %{p}_ok0, %st_live_p0 : i1"]
    ok = f"%{p}_ok1"
    if extra:
        L.append(f"    %{p}_ok = scalar.andi %{p}_ok1, {extra} : i1"); ok = f"%{p}_ok"
    L += [f"    %{p}_k, %{p}_v, %{p}_sk, %{p}_sv = scf.if {ok} -> (vector<16xi8>, vector<16xi8>, f32, f32) {{",
          f"      %{p}_kp = index.assume %{p}_kp_raw [range(%{p}_kp_raw, 0, 1048575)] : index",
          f"      %{p}_koff = func.call @g_kv_offset8(%origins, %row, %{p}_kp, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)",
          f"      %{p}_voff = index.add %{p}_koff, %plane_stride : offset",
          f"      %{p}_c1024 = index.constant 1024 : offset",
          f"      %{p}_kc = buffer.view %arena[%{p}_koff] : buffer -> view<8x128xi8>",
          f"      %{p}_kso = index.add %{p}_koff, %{p}_c1024 : offset",
          f"      %{p}_ks = buffer.view %arena[%{p}_kso] : buffer -> view<8x4xf16>",
          f"      %{p}_vc = buffer.view %arena[%{p}_voff] : buffer -> view<8x128xi8>",
          f"      %{p}_vso = index.add %{p}_voff, %{p}_c1024 : offset",
          f"      %{p}_vs = buffer.view %arena[%{p}_vso] : buffer -> view<8x4xf16>",
          f"      %{p}_c32 = index.constant 32 : index",
          f"      %{p}_bv = index.div %st_dim_p0, %{p}_c32 : index",
          f"      %{p}_bb = index.assume %{p}_bv [range(%{p}_bv, 0, 3)] : index",
          f"      %{p}_kq = vector.load %{p}_kc[%kv, %st_dim_p0] : view<8x128xi8> -> vector<16xi8>",
          f"      %{p}_ks16 = view.load %{p}_ks[%kv, %{p}_bb] : view<8x4xf16> -> f16",
          f"      %{p}_ksf = scalar.extf %{p}_ks16 : f16 to f32",
          f"      %{p}_vq = vector.load %{p}_vc[%kv, %st_dim_p0] : view<8x128xi8> -> vector<16xi8>",
          f"      %{p}_vs16 = view.load %{p}_vs[%kv, %{p}_bb] : view<8x4xf16> -> f16",
          f"      %{p}_vsf = scalar.extf %{p}_vs16 : f16 to f32",
          f"      scf.yield %{p}_kq, %{p}_vq, %{p}_ksf, %{p}_vsf : vector<16xi8>, vector<16xi8>, f32, f32",
          "    } else {",
          "      scf.yield %pf_zq, %pf_zq, %zero_f, %zero_f : vector<16xi8>, vector<16xi8>, f32, f32",
          "    }"]
    return "\n".join(L) + "\n"


def prefetch(k):
    """Software-pipeline the K/V staging: tile kt+1's global loads are issued right after the barrier that publishes tile kt,
    so they are in flight during tile kt's MMAs. Tile kt is dequantized and stored from registers at the top of iteration kt.
    Same values reach LDS (same dequant arithmetic, zeros for invalid slots), so the output stays bit-identical."""
    head_old = "  %m_final, %l_final, %o0, %o1, %o2, %o3, %o4, %o5, %o6, %o7 = scf.for %kt = [%c0 to %key_tiles step %c1]("
    a = k.index(head_old); eol = k.index("\n", a); head = k[a:eol]
    assert head.endswith(") {") and "%a7 = %init : vector<8xf32>)" in head, head[-120:]
    head = head.replace("%o7 = scf.for", "%o7, %pf_kf, %pf_vf, %pf_skf, %pf_svf = scf.for", 1)
    head = head.replace("%a7 = %init : vector<8xf32>)",
                        "%a7 = %init : vector<8xf32>, %pk = %pf0_k : vector<16xi8>, %pv = %pf0_v : vector<16xi8>, %psk = %pf0_sk : f32, %psv = %pf0_sv : f32)", 1)
    head = head[:-len(") {")] + ", vector<16xi8>, vector<16xi8>, f32, f32) {"
    pro = "  %pf_zq = vector.splati 0 : vector<16xi8>\n" + load_block("pf0", "%key_low").replace("\n    ", "\n  ").replace("    %pf0_kp_raw", "  %pf0_kp_raw", 1)
    k = k[:a] + pro + head + k[eol:]
    # staging from the carried registers
    s0 = k.index("    %key_pos_p0 = index.add %tile_first, %st_key_p0 : index\n")
    e_mark = "    } else {\n      scf.yield %kz, %kz : vector<16xf16>, vector<16xf16>\n    }\n"
    e0 = k.index(e_mark, s0) + len(e_mark)
    k = k[:s0] + ("    %pf_kfl = vector.sitofp %pk : vector<16xi8> to vector<16xf32>\n"
                  "    %pf_ksp = vector.splat %psk : vector<16xf32>\n"
                  "    %pf_kd = vector.mulf %pf_kfl, %pf_ksp : vector<16xf32>\n"
                  "    %kvals_p0 = vector.fptrunc %pf_kd : vector<16xf32> to vector<16xf16>\n"
                  "    %pf_vfl = vector.sitofp %pv : vector<16xi8> to vector<16xf32>\n"
                  "    %pf_vsp = vector.splat %psv : vector<16xf32>\n"
                  "    %pf_vd = vector.mulf %pf_vfl, %pf_vsp : vector<16xf32>\n"
                  "    %vvals_p0 = vector.fptrunc %pf_vd : vector<16xf32> to vector<16xf16>\n") + k[e0:]
    # issue tile kt+1's loads after the publishing barrier
    bar = "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    %tf32 = index.cast %tile_first : index to i32\n"
    k = sub1(k, bar, "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
             "    %pf_ktn = index.add %kt, %c1 : index\n"
             "    %pf_has = index.cmp ult, %pf_ktn, %key_tiles : index\n"
             "    %pf_tfn = index.madd %pf_ktn, %ckt, %key_low : index\n"
             + load_block("pfn", "%pf_tfn", "%pf_has") +
             "    %tf32 = index.cast %tile_first : index to i32\n")
    # carry them
    y = k.index("    scf.yield %m_new, %l_new, %on0, %on1, %on2, %on3, %on4, %on5, %on6, %on7 : f32, f32, ")
    ye = k.index("\n", y); yl = k[y:ye]
    yl = yl.replace("%on7 : f32, f32,", "%on7, %pfn_k, %pfn_v, %pfn_sk, %pfn_sv : f32, f32,", 1) + ", vector<16xi8>, vector<16xi8>, f32, f32"
    return k[:y] + yl + k[ye:]


def uniform_chain(k, nsub):
    """The tile-ownership values (adjacency tests, this wave's sub-tile, first token, positions, key-tile count) are the
    same in every lane of a wave, but the adjacency tests are per-lane loads, so everything derived from them, including the
    loop bound, lives in VGPRs. Broadcasting the adjacency bits and the sub-tile index from lane 0 makes them scalar.
    Same values: bit-identical."""
    k = sub1(k, "  %ms_true = scalar.constant true : i1\n",
             "  %ms_true = scalar.constant true : i1\n"
             "  %ms_sub_i = index.cast %sub : index to i32\n"
             "  %ms_sub_b = kernel.subgroup.broadcast %ms_sub_i from %c0_i32 : i32, i32\n"
             "  %ms_subu = index.cast %ms_sub_b : i32 to index\n")
    for j in range(1, nsub):
        call = f"  %ms_adj{j} = func.call @g_tile_adjacent("
        a = k.index(call); e = k.index("\n", a) + 1
        line = k[a:e].replace(f"%ms_adj{j} = ", f"%ms_adjr{j} = ", 1)
        k = k[:a] + line + (f"  %ms_adji{j} = scf.select %ms_adjr{j}, %c1_i32m, %c0_i32 : i32\n"
                            f"  %ms_adjb{j} = kernel.subgroup.broadcast %ms_adji{j} from %c0_i32 : i32, i32\n"
                            f"  %ms_adj{j} = scalar.cmpi ne, %ms_adjb{j}, %c0_i32 : i32\n") + k[e:]
        k = sub1(k, f"  %ms_is{j} = index.cmp eq, %sub, %ms_c{j} : index\n", f"  %ms_is{j} = index.cmp eq, %ms_subu, %ms_c{j} : index\n")
    k = sub1(k, "  %ms_not0 = index.cmp ne, %sub, %c0 : index\n", "  %ms_not0 = index.cmp ne, %ms_subu, %c0 : index\n")
    k = sub1(k, "  %ms_sub16 = index.mul %sub, %c16 : index\n", "  %ms_sub16 = index.mul %ms_subu, %c16 : index\n")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--nsub", type=int, default=4, choices=(4, 8))
    ap.add_argument("--key-major", action="store_true", help="key-major K/V staging lanes (fewer LDS bank conflicts)")
    ap.add_argument("--prefetch", action="store_true", help="load the next K/V tile while the current one is computed")
    ap.add_argument("--uniform", action="store_true", help="broadcast the wave-uniform tile-ownership values (scalar registers)")
    a = ap.parse_args()
    src = open(a.inp).read()
    s, e = kernel_span(src)
    new = rewrite(src[s:e], a.nsub, a.key_major)
    if a.uniform:
        new = uniform_chain(new, a.nsub)
    if a.prefetch:
        new = prefetch(new)
    open(a.out, "w").write(src[:s] + new + src[e:])
    print(f"rewrote {NAME}: {a.nsub} query tiles per workgroup ({96 * a.nsub} lanes)", file=sys.stderr)


if __name__ == "__main__":
    main()
