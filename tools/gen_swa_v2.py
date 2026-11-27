#!/usr/bin/env python3
"""gen_swa_v2.py SRC.loom DST.loom [--splits 16|32|64] : port SWA decode attention
(g_ink_attn_split_swa + g_ink_attn_split_swa_span_v2, 40 of 48 passes/token) to the v2
1-wave / 3-head / 16-byte-load layout with lane-distributed rel_bias and reduced split count.

Stock g_ink_attn_split_swa / _span:
  - 3 waves (96 lanes = 1 wave per query head), reloading the same f16 K/V 3x.
  - 1 key per iteration, 8-byte vector<4xf16> loads, 32-lane subgroup reduction.
  - Every lane of every wave loads all 16 floats of rel_bias[dist] and computes the same 16-wide
    dot product (96x redundant reads/math per key).
  - Always uses up to 64 splits, writing/re-reading ~31.5 MB/token through pm/pl/po and running
    64 combine iterations per pass.

v2:
  - 1 wave (32 lanes) serving all 3 query heads of a KV-head group.
  - Lane = (key slot ks in 0..1, dim chunk dc in 0..15): each lane loads 8 f16 (16 B) of K and
    8 f16 (16 B) of V (256 contiguous bytes per key across lanes 0..15 / 16..31) and exactly 1
    float of rel_bias[dist, dc] (64 contiguous bytes, zero redundant loads).
  - Lane folds r[h, dc] * rel_bias[dist, dc] into its 8-wide Q*K dot product, then a 4-step XOR
    butterfly (x1, x2, x4, x8) yields exact scores for both keys across all 3 heads.
  - Half the VGPRs of full-attention v2 (~96 vs ~180) and only 2 key slots to merge (XOR 16).
  - Caps SWA splits at --splits (default 16) in split_swa, split_swa_span(_v2), and combine.
"""
import argparse
import re
import sys

H = 3  # query heads per KV head


def swa_single_kernel(splits):
    L = []
    a = L.append
    a(f'// SWA decode split-K attention v2: 1 wave x 3 query heads, 2 keys/step, 16-byte f16 loads, lane-distributed rel_bias, max {splits} splits.')
    a('kernel.def export("g_ink_attn_split_swa") @g_ink_attn_split_swa(%tokens: index) {')
    a('  %c1 = index.constant 1 : index')
    a('  %cKH = index.constant 8 : index')
    a(f'  %cS = index.constant {splits} : index')
    a('  %lanes = index.constant 32 : index')
    a('  kernel.launch.config workgroups(%tokens, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index')
    a('} launch(%tokens_arg: index, %spans_arg: index, %swa_off_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, '
      '%q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {')
    a(f'''  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %swa_off_b = index.assume %swa_off_arg [range(%swa_off_arg, 0, 1073741824)] : index
  %swa_off = index.cast %swa_off_b : index to offset
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %kh_id = kernel.workgroup.id<y> : index
  %kh = index.assume %kh_id [range(%kh_id, 0, 7)] : index
  %sp_id = kernel.workgroup.id<z> : index
  %sp = index.assume %sp_id [range(%sp_id, 0, {splits - 1})] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %cG = index.constant 3 : index
  %cS = index.constant {splits} : index
  %cWIN1 = index.constant 1023 : index
  %cRING = index.constant 1040 : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %x1 = scalar.constant 1 : i32
  %x2 = scalar.constant 2 : i32
  %x4 = scalar.constant 4 : i32
  %x8 = scalar.constant 8 : i32
  %x16 = scalar.constant 16 : i32
  %w32 = scalar.constant 32 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero8 = vector.splatf 0.0 : vector<8xf32>
  %scale = scalar.constant 0.08838834764831843 : f32
  %scale4 = vector.splat %scale : vector<4xf32>
  %ks_v = index.div %lane, %c16 : index
  %ks = index.assume %ks_v [range(%ks_v, 0, 1)] : index
  %dc_v = index.rem %lane, %c16 : index
  %dc = index.assume %dc_v [range(%dc_v, 0, 15)] : index
  %dc2_v = index.mul %dc, %c2 : index
  %dc2 = index.assume %dc2_v [range(%dc2_v, 0, 30)] : index
  %dc2_1_v = index.add %dc2, %c1 : index
  %dc2_1 = index.assume %dc2_1_v [range(%dc2_1_v, 1, 31)] : index
  %hd_base = index.mul %kh, %cG : index
  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  scf.if %live {{
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %span32 = view.load %info_v[%tok, 3] : view<[%tokens]x4xi32> -> i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 1048575)] : i32
    %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %pos = index.cast %pos_b : i32 to index
    %span = index.cast %span_b : i32 to index
    %gt = index.cmp ugt, %pos, %cWIN1 : index
    %sub = index.sub %pos, %cWIN1 : index
    %w_low = scf.select %gt, %sub, %c0 : index
    %n_keys_0 = index.sub %pos, %w_low : index
    %n_keys_raw = index.add %n_keys_0, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 1024)] : index
    %nk7 = index.add %n_keys, %c7 : index
    %nk7_b = index.assume %nk7 [range(%nk7, 8, 1031)] : index
    %s_need = index.div %nk7_b, %c8 : index
    %s_eff0 = index.min %s_need, %cS : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, {splits})] : index
    %sp_active = index.cmp ult, %sp, %s_eff : index
    scf.if %sp_active {{
      %chunk0 = index.add %n_keys, %s_eff : index
      %chunk1_raw = index.sub %chunk0, %c1 : index
      %chunk1 = index.assume %chunk1_raw [range(%chunk1_raw, 1, {1024 + splits})] : index
      %chunk = index.div %chunk1, %s_eff : index
      %st0 = index.mul %sp, %chunk : index
      %st = index.min %st0, %n_keys : index
      %en0 = index.add %st, %chunk : index
      %en = index.min %en0, %n_keys : index
      %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
      %ring_rd, %ring_wr, %ring_fr = func.call @g_gdn_slots(%origins, %row, %start32, %swa_off) : (buffer, index, i32, offset) -> (offset, offset, i1)
      %plane_sz = index.constant 2129920 : offset
      %ring_v = index.add %ring_wr, %plane_sz : offset
      %kv_ring = buffer.view %arena[%ring_wr] : buffer -> view<1040x8x16x8xf16>
      %vv_ring = buffer.view %arena[%ring_v] : buffer -> view<1040x8x16x8xf16>
      %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
      %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
      %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>''')
    a('      %hd0_v = index.add %hd_base, %c0 : index')
    a('      %hd1_v = index.add %hd_base, %c1 : index')
    a('      %hd2_v = index.add %hd_base, %c2 : index')
    for h in range(H):
        a(f'      %hd{h} = index.assume %hd{h}_v [range(%hd{h}_v, 0, 23)] : index')
        a(f'      %qr{h}_0 = vector.load %qv[%tok, %hd{h}, %dc2, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>')
        a(f'      %qr{h}_1 = vector.load %qv[%tok, %hd{h}, %dc2_1, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>')
        a(f'      %qs{h}_0 = vector.mulf %qr{h}_0, %scale4 : vector<4xf32>')
        a(f'      %qs{h}_1 = vector.mulf %qr{h}_1, %scale4 : vector<4xf32>')
        a(f'      %q{h} = vector.concat<0> %qs{h}_0, %qs{h}_1 : vector<4xf32>, vector<4xf32> -> vector<8xf32>')
        a(f'      %r{h} = view.load %rv[%tok, %hd{h}, %dc] : view<[%tokens]x24x16xf32> -> f32')

    ty = ", ".join(["f32, f32"] * H) + ", " + ", ".join(["vector<8xf32>"] * H)
    it = ", ".join(f"%km{h} = %neg_inf : f32, %kl{h} = %zero_f : f32" for h in range(H)) + ", " + \
         ", ".join(f"%ka{h} = %zero8 : vector<8xf32>" for h in range(H))
    rf = ", ".join(f"%m{h}f, %l{h}f" for h in range(H)) + ", " + ", ".join(f"%a{h}f" for h in range(H))
    a(f'      {rf} = scf.for %k = [%st to %en step %c2]({it}) -> ({ty}) {{')
    a('''        %j = index.add %k, %ks : index
        %valid = index.cmp ult, %j, %en : index
        %k8, %v8, %rb_s = scf.if %valid -> (vector<8xf32>, vector<8xf32>, f32) {
          %kp_v = index.add %w_low, %j : index
          %kp = index.assume %kp_v [range(%kp_v, 0, 1048575)] : index
          %rp_v = index.rem %kp, %cRING : index
          %rp = index.assume %rp_v [range(%rp_v, 0, 1039)] : index
          %kh8 = vector.load %kv_ring[%rp, %kh, %dc, %c0] : view<1040x8x16x8xf16> -> vector<8xf16>
          %kf8 = vector.extf %kh8 : vector<8xf16> to vector<8xf32>
          %vh8 = vector.load %vv_ring[%rp, %kh, %dc, %c0] : view<1040x8x16x8xf16> -> vector<8xf16>
          %vf8 = vector.extf %vh8 : vector<8xf16> to vector<8xf32>
          %dist = index.sub %pos, %kp : index
          %db = index.assume %dist [range(%dist, 0, 1023)] : index
          %bs = view.load %rbv[%db, %dc] : view<1024x16xf32> -> f32
          scf.yield %kf8, %vf8, %bs : vector<8xf32>, vector<8xf32>, f32
        } else {
          scf.yield %zero8, %zero8, %zero_f : vector<8xf32>, vector<8xf32>, f32
        }''')
    for h in range(H):
        a(f'''        %pq{h} = vector.dotf<reassoc|contract> %q{h}, %k8, %zero_f : vector<8xf32>, vector<8xf32>, f32
        %pp{h} = scalar.fmaf %r{h}, %rb_s, %pq{h} : f32
        %b{h}_1r, %b{h}_1u = kernel.subgroup.shuffle<xor> %pp{h}, %x1, %w32 : f32, i32, i32
        %b{h}_1 = scalar.addf %pp{h}, %b{h}_1r : f32
        %b{h}_2r, %b{h}_2u = kernel.subgroup.shuffle<xor> %b{h}_1, %x2, %w32 : f32, i32, i32
        %b{h}_2 = scalar.addf %b{h}_1, %b{h}_2r : f32
        %b{h}_4r, %b{h}_4u = kernel.subgroup.shuffle<xor> %b{h}_2, %x4, %w32 : f32, i32, i32
        %b{h}_4 = scalar.addf %b{h}_2, %b{h}_4r : f32
        %b{h}_8r, %b{h}_8u = kernel.subgroup.shuffle<xor> %b{h}_4, %x8, %w32 : f32, i32, i32
        %sraw{h} = scalar.addf %b{h}_4, %b{h}_8r : f32
        %s{h} = scf.select %valid, %sraw{h}, %neg_inf : f32
        %nm{h} = scalar.maxnumf %km{h}, %s{h} : f32
        %ms{h} = scalar.subf %km{h}, %nm{h} : f32
        %al{h} = scalar.expf<afn> %ms{h} : f32
        %ds{h} = scalar.subf %s{h}, %nm{h} : f32
        %pe{h} = scalar.expf<afn> %ds{h} : f32
        %pv{h} = scf.select %valid, %pe{h}, %zero_f : f32
        %ls{h} = scalar.mulf %kl{h}, %al{h} : f32
        %nl{h} = scalar.addf %ls{h}, %pv{h} : f32
        %alv{h} = vector.splat %al{h} : vector<8xf32>
        %as{h} = vector.mulf %ka{h}, %alv{h} : vector<8xf32>
        %pvv{h} = vector.splat %pv{h} : vector<8xf32>
        %na{h} = vector.fmaf %pvv{h}, %v8, %as{h} : vector<8xf32>''')
    yk = ", ".join(f"%nm{h}, %nl{h}" for h in range(H)) + ", " + ", ".join(f"%na{h}" for h in range(H))
    a(f'        scf.yield {yk} : {ty}')
    a('      }')

    # Merge the 2 key slots (ks=0 and ks=1 differ in lane bit 4 -> XOR 16).
    for h in range(H):
        a(f'''      %M{h}_16r, %M{h}_16u = kernel.subgroup.shuffle<xor> %m{h}f, %x16, %w32 : f32, i32, i32
      %M{h} = scalar.maxnumf %m{h}f, %M{h}_16r : f32
      %e{h}_d = scalar.subf %m{h}f, %M{h} : f32
      %e{h} = scalar.expf<afn> %e{h}_d : f32
      %lw{h} = scalar.mulf %l{h}f, %e{h} : f32
      %L{h}_16r, %L{h}_16u = kernel.subgroup.shuffle<xor> %lw{h}, %x16, %w32 : f32, i32, i32
      %L{h} = scalar.addf %lw{h}, %L{h}_16r : f32
      %ev{h} = vector.splat %e{h} : vector<8xf32>
      %aw{h} = vector.mulf %a{h}f, %ev{h} : vector<8xf32>''')
        for i in range(8):
            a(f'''      %aw{h}_{i} = vector.extract %aw{h}[{i}] : vector<8xf32> -> f32
      %t{h}_{i}_16r, %t{h}_{i}_16u = kernel.subgroup.shuffle<xor> %aw{h}_{i}, %x16, %w32 : f32, i32, i32
      %t{h}_{i} = scalar.addf %aw{h}_{i}, %t{h}_{i}_16r : f32''')
        a(f'      %A{h}_lo = vector.from_elements %t{h}_0, %t{h}_1, %t{h}_2, %t{h}_3 : vector<4xf32>')
        a(f'      %A{h}_hi = vector.from_elements %t{h}_4, %t{h}_5, %t{h}_6, %t{h}_7 : vector<4xf32>')

    a('      scf.if %is_l0 {')
    for h in range(H):
        a(f'        view.store %M{h}, %pm_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>')
        a(f'        view.store %L{h}, %pl_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>')
    a('      }')
    a('      %ol_v = index.add %dc2, %ks : index')
    a('      %ol = index.assume %ol_v [range(%ol_v, 0, 31)] : index')
    a('      %is_ks0 = index.cmp eq, %ks, %c0 : index')
    a('      scf.if %is_ks0 {')
    for h in range(H):
        a(f'        vector.store %A{h}_lo, %po_v[%tok, %hd{h}, %sp, %ol, %c0] : vector<4xf32>, view<[%tokens]x24x64x32x4xf32>')
    a('      } else {')
    for h in range(H):
        a(f'        vector.store %A{h}_hi, %po_v[%tok, %hd{h}, %sp, %ol, %c0] : vector<4xf32>, view<[%tokens]x24x64x32x4xf32>')
    a('      }')
    a('    }')
    a('  }')
    a('  kernel.return')
    a('}')
    return "\n".join(L) + "\n"


def swa_span_v2_kernel(splits):
    """Single-token-span variant (pure-decode multi-chat epochs where t == spans)."""
    k = swa_single_kernel(splits)
    # Replace header through %kh_id
    head = f'''// SWA decode span v2 (pure-decode epochs where every span has length 1).
kernel.def export("g_ink_attn_split_swa_span_v2") @g_ink_attn_split_swa_span_v2(%spans: index) {{
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant {splits} : index
  %lanes = index.constant 32 : index
  kernel.launch.config workgroups(%spans, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %swa_off_arg: index, %metadata: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %swa_off_b = index.assume %swa_off_arg [range(%swa_off_arg, 0, 1073741824)] : index
  %swa_off = index.cast %swa_off_b : index to offset
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
'''
    h_end = k.index("  %kh_id = kernel.workgroup.id<y>")
    k = head + k[h_end:]

    # Replace %info_v .. %span_b / %span with g_gdn_span prologue
    old_pro = '''  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  scf.if %live {
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %span32 = view.load %info_v[%tok, 3] : view<[%tokens]x4xi32> -> i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 1048575)] : i32
    %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %pos = index.cast %pos_b : i32 to index
    %span = index.cast %span_b : i32 to index'''

    new_pro = '''  %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {
    %tok = index.assume %first [range(%first, 0, 8191), lt(%first, %tokens)] : index
    %start_b = scalar.assume %start32 [range(%start32, 0, 1048575)] : i32
    %pos = index.cast %start_b : i32 to index'''

    assert old_pro in k
    k = k.replace(old_pro, new_pro, 1)

    # Remove the now-duplicate g_gdn_span call inside scf.if %sp_active
    dup_span = "      %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)\n"
    assert dup_span in k
    k = k.replace(dup_span, "", 1)
    assert "%info" not in k and "%live" not in k
    return k


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


def patch(text, splits_c1, splits_cn):
    if 'export("g_ink_attn_split_swa_span_v2")' in text:
        raise ValueError("already patched by gen_swa_v2.py")

    # 1. Replace g_ink_attn_split_swa (t == 1) with splits_c1 and append g_ink_attn_split_swa_span_v2 (t > 1) with splits_cn.
    lo, hi = find_kernel_span(text, "g_ink_attn_split_swa")
    prev_nl = text.rfind("\n", 0, lo - 1)
    if prev_nl >= 0 and text[prev_nl + 1:lo].startswith("// SWA decode split-K"):
        lo = prev_nl + 1
    text = text[:lo] + swa_single_kernel(splits_c1) + "\n" + swa_span_v2_kernel(splits_cn) + text[hi:]

    # 2. Keep stock g_ink_attn_split_swa_span (drafter fallback, t > 1) in sync with splits_cn.
    if splits_cn != 64:
        lo, hi = find_kernel_span(text, "g_ink_attn_split_swa_span")
        body = text[lo:hi]
        reps = [
            ("  %cS = index.constant 64 : index\n", f"  %cS = index.constant {splits_cn} : index\n"),
            ("  %sp = index.assume %sp_id [range(%sp_id, 0, 63)] : index\n",
             f"  %sp = index.assume %sp_id [range(%sp_id, 0, {splits_cn - 1})] : index\n"),
            ("      %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index\n",
             f"      %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, {splits_cn})] : index\n"),
        ]
        for old, new in reps:
            assert old in body, f"g_ink_attn_split_swa_span missing: {old!r}"
            body = body.replace(old, new)
        text = text[:lo] + body + text[hi:]

    # 3. Keep g_ink_attn_combine in sync: SWA uses splits_c1 when tokens == 1 and splits_cn when tokens > 1.
    if splits_c1 != 64 or splits_cn != 64:
        lo, hi = find_kernel_span(text, "g_ink_attn_combine")
        body = text[lo:hi]
        old_comb = """    %s_eff0 = index.min %s_need, %cS : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index"""
        if splits_c1 == splits_cn:
            new_comb = f"""    %cS_swa = index.constant {splits_c1} : index
    %cS_lim = scf.select %is_swa, %cS_swa, %cS : index
    %s_eff0 = index.min %s_need, %cS_lim : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index"""
        else:
            new_comb = f"""    %is_t1 = index.cmp eq, %tokens, %c1 : index
    %cS_swa_1 = index.constant {splits_c1} : index
    %cS_swa_n = index.constant {splits_cn} : index
    %cS_swa = scf.select %is_t1, %cS_swa_1, %cS_swa_n : index
    %cS_lim = scf.select %is_swa, %cS_swa, %cS : index
    %s_eff0 = index.min %s_need, %cS_lim : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index"""
        assert old_comb in body, "g_ink_attn_combine s_eff pattern not found"
        body = body.replace(old_comb, new_comb, 1)
        text = text[:lo] + body + text[hi:]

    # 4. Dispatch g_ink_attn_split_swa_span_v2 when t == spans (pure-decode multi-chat epochs).
    n = 0

    def rep_dispatch(m):
        nonlocal n
        n += 1
        ind, line = m.group(1), m.group(2)
        v2 = line.replace("@g_ink_attn_split_swa_span[", "@g_ink_attn_split_swa_span_v2[", 1)
        return (f"{ind}%swav2_{n} = index.cmp eq, %t, %spans : index\n"
                f"{ind}scf.if %swav2_{n} {{\n{ind}  {v2}\n{ind}}} else {{\n{ind}  {line}\n{ind}}}\n")

    text = re.sub(r"^( *)(kernel\.launch @g_ink_attn_split_swa_span\[%spans\]\(.*)\n", rep_dispatch, text, flags=re.M)
    assert n == 80, f"expected 80 swa_span call sites, got {n}"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--splits", type=int, default=None, choices=(8, 16, 32, 64), help="Set both c1 and cn splits")
    ap.add_argument("--splits-c1", type=int, default=32, choices=(8, 16, 32, 64), help="SWA splits for t == 1 (default 32)")
    ap.add_argument("--splits-cn", type=int, default=16, choices=(8, 16, 32, 64), help="SWA splits for t > 1 (default 16)")
    args = ap.parse_args()
    s1 = args.splits if args.splits is not None else args.splits_c1
    sn = args.splits if args.splits is not None else args.splits_cn
    patched = patch(open(args.src).read(), s1, sn)
    open(args.dst, "w").write(patched)
    print(f"replaced g_ink_attn_split_swa (splits={s1}) + added g_ink_attn_split_swa_span_v2 (80 sites, splits={sn})", file=sys.stderr)


if __name__ == "__main__":
    main()
