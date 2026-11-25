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

usage: gen_flash2_multi.py IN.loom OUT.loom [--nsub 4|8] [--key-major]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--nsub", type=int, default=4, choices=(4, 8))
    ap.add_argument("--key-major", action="store_true", help="key-major K/V staging lanes (fewer LDS bank conflicts)")
    a = ap.parse_args()
    src = open(a.inp).read()
    s, e = kernel_span(src)
    new = rewrite(src[s:e], a.nsub, a.key_major)
    open(a.out, "w").write(src[:s] + new + src[e:])
    print(f"rewrote {NAME}: {a.nsub} query tiles per workgroup ({96 * a.nsub} lanes)", file=sys.stderr)


if __name__ == "__main__":
    main()
