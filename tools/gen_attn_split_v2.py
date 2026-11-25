#!/usr/bin/env python3
"""Replace Inkling's single-token decode attention g_ink_attn_split with a lane-grouped version (v2).

Stock kernel: workgroup = 3 waves, one wave per query head of a KV-head group; each lane holds 4 of the 128 dims and
walks keys one at a time. Per key and head it issues 4 tiny vector-memory ops (4 B of K, 4 B of V, two 2-byte scales)
and a 32-lane reduction, and the 3 waves re-load the same K/V. Measured KV read ~128 GB/s (~20% of 640 GB/s).

v2: workgroup = 1 wave serving all 3 query heads. Lane = (key slot ks in 0..3, dim chunk dc in 0..7): each lane loads
16 int8 of K and 16 of V (one 16-byte load each) for its key, so one wave instruction covers 4 keys for all 3 heads
(~12x fewer memory instructions). Scores: 16-wide dot (+ the lane's 2 of the 16 relative-bias terms for keys within
1024) and an 8-lane xor butterfly. Each key slot keeps its own online softmax per head; the 4 slots merge at the end.
Same name, arguments, grid (tokens x 8 x 64), int8 dequant (f16 round-trip kept) and partial layout (pm/pl/po), so
the dispatch and g_ink_attn_combine are unchanged. Summation order differs from stock (validate with eqlong/eqcheck).

usage: python3 -I gen_attn_split_v2.py SRC_llm.loom DST_llm.loom
"""
import sys

H = 3  # query heads per KV head


def kernel_text():
    L = []
    a = L.append
    a('kernel.def export("g_ink_attn_split") @g_ink_attn_split(%tokens: index) {')
    a('  %c1 = index.constant 1 : index')
    a('  %cKH = index.constant 8 : index')
    a('  %cS = index.constant 64 : index')
    a('  %lanes = index.constant 32 : index')
    a('  kernel.launch.config workgroups(%tokens, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index')
    a('} launch(%tokens_arg: index, %blocks_arg: index, %pool_arg: index, %layer_arg: index, %sliding_arg: i32, %info: buffer, '
      '%origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {')
    a('''  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 131072)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 47)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %kh_id = kernel.workgroup.id<y> : index
  %kh = index.assume %kh_id [range(%kh_id, 0, 7)] : index
  %sp_id = kernel.workgroup.id<z> : index
  %sp = index.assume %sp_id [range(%sp_id, 0, 63)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %cG = index.constant 3 : index
  %cS = index.constant 64 : index
  %cEXT = index.constant 1024 : index
  %cWIN1 = index.constant 1023 : index
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
  %zero16 = vector.splatf 0.0 : vector<16xf32>
  %zero2 = vector.splatf 0.0 : vector<2xf32>
  %scale = scalar.constant 0.08838834764831843 : f32
  %scale4 = vector.splat %scale : vector<4xf32>
  %ks_v = index.div %lane, %c8 : index
  %ks = index.assume %ks_v [range(%ks_v, 0, 3)] : index
  %dc_v = index.rem %lane, %c8 : index
  %dc = index.assume %dc_v [range(%dc_v, 0, 7)] : index
  %dc2_v = index.mul %dc, %c2 : index
  %dc2 = index.assume %dc2_v [range(%dc2_v, 0, 14)] : index
  %dc4_v = index.mul %dc, %c4 : index
  %dc4 = index.assume %dc4_v [range(%dc4_v, 0, 28)] : index
  %sb_v = index.div %dc, %c2 : index
  %sb = index.assume %sb_v [range(%sb_v, 0, 3)] : index
  %hd_base = index.mul %kh, %cG : index
  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  scf.if %live {
    %row32 = view.load %info_v[%tok, 0] : view<[%tokens]x4xi32> -> i32
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 2097151)] : i32
    %row = index.cast %row_b : i32 to index
    %pos = index.cast %pos_b : i32 to index
    %is_swa = scalar.cmpi ne, %sliding_arg, %c0_i32 : i32
    %w_low = scf.if %is_swa -> (index) {
      %gt = index.cmp ugt, %pos, %cWIN1 : index
      %sub = index.sub %pos, %cWIN1 : index
      %wl = scf.select %gt, %sub, %c0 : index
      scf.yield %wl : index
    } else {
      scf.yield %c0 : index
    }
    %n_keys_0 = index.sub %pos, %w_low : index
    %n_keys_raw = index.add %n_keys_0, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index
    %c7 = index.constant 7 : index
    %nk7 = index.add %n_keys, %c7 : index
    %nk7_b = index.assume %nk7 [range(%nk7, 8, 2097159)] : index
    %s_need = index.div %nk7_b, %c8 : index
    %s_eff0 = index.min %s_need, %cS : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index
    %sp_active = index.cmp ult, %sp, %s_eff : index
    scf.if %sp_active {
      %chunk0 = index.add %n_keys, %s_eff : index
      %chunk1_raw = index.sub %chunk0, %c1 : index
      %chunk1 = index.assume %chunk1_raw [range(%chunk1_raw, 1, 2097216)] : index
      %chunk = index.div %chunk1, %s_eff : index
      %st0 = index.mul %sp, %chunk : index
      %st = index.min %st0, %n_keys : index
      %en0 = index.add %st, %chunk : index
      %en = index.min %en0, %n_keys : index
      %k_plane = index.mul %layer, %c2 : index
      %pos_bytes = index.constant 1088 : offset
      %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
      %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
      %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
      %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>''')
    # q (scaled) and r slices for the 3 heads
    for h in range(H):
        a(f'''      %hd{h}_v = index.add %hd_base, %c{h} : index''' if h in (0, 1, 2, 3) else '')
    # constants %c0 %c1 %c2 exist; build head indices
    L[-3] = '      %hd0_v = index.add %hd_base, %c0 : index'
    L[-2] = '      %hd1_v = index.add %hd_base, %c1 : index'
    L[-1] = '      %hd2_v = index.add %hd_base, %c2 : index'
    for h in range(H):
        a(f'      %hd{h} = index.assume %hd{h}_v [range(%hd{h}_v, 0, 23)] : index')
        for i in range(4):
            a(f'      %qi{h}_{i}_v = index.add %dc4, %c{i} : index' if i < 4 else '')
        for i in range(4):
            a(f'      %qi{h}_{i} = index.assume %qi{h}_{i}_v [range(%qi{h}_{i}_v, 0, 31)] : index')
            a(f'      %qr{h}_{i} = vector.load %qv[%tok, %hd{h}, %qi{h}_{i}, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>')
            a(f'      %qs{h}_{i} = vector.mulf %qr{h}_{i}, %scale4 : vector<4xf32>')
        a(f'      %qa{h} = vector.concat<0> %qs{h}_0, %qs{h}_1 : vector<4xf32>, vector<4xf32> -> vector<8xf32>')
        a(f'      %qb{h} = vector.concat<0> %qs{h}_2, %qs{h}_3 : vector<4xf32>, vector<4xf32> -> vector<8xf32>')
        a(f'      %q{h} = vector.concat<0> %qa{h}, %qb{h} : vector<8xf32>, vector<8xf32> -> vector<16xf32>')
        a(f'      %r{h} = vector.load %rv[%tok, %hd{h}, %dc2] : view<[%tokens]x24x16xf32> -> vector<2xf32>')
    a('''      %st_pos_v = index.add %w_low, %st : index
      %st_pos = index.assume %st_pos_v [range(%st_pos_v, 0, 2097152)] : index
      %en_pos_v = index.add %w_low, %en : index
      %en_pos = index.assume %en_pos_v [range(%en_pos_v, 0, 2097152)] : index
      %has_keys = index.cmp ult, %st_pos, %en_pos : index''')
    st = ", ".join(f"%m{h}f, %l{h}f" for h in range(H)) + ", " + ", ".join(f"%a{h}f" for h in range(H))
    ty = ", ".join(["f32, f32"] * H) + ", " + ", ".join(["vector<16xf32>"] * H)
    init = ", ".join(["%neg_inf, %zero_f"] * H) + ", " + ", ".join(["%zero16"] * H)
    a(f'      {st} = scf.if %has_keys -> ({ty}) {{')
    a('''        %pg0 = index.div %st_pos, %c16 : index
        %en_m1_v = index.sub %en_pos, %c1 : index
        %en_m1 = index.assume %en_m1_v [range(%en_m1_v, 0, 2097151)] : index
        %pg1_0 = index.div %en_m1, %c16 : index
        %pg1 = index.add %pg1_0, %c1 : index''')
    it_p = ", ".join(f"%pm{h} = %neg_inf : f32, %pl{h} = %zero_f : f32" for h in range(H)) + ", " + \
        ", ".join(f"%pa{h} = %zero16 : vector<16xf32>" for h in range(H))
    rp = ", ".join(f"%om{h}, %ol{h}" for h in range(H)) + ", " + ", ".join(f"%oa{h}" for h in range(H))
    a(f'        {rp} = scf.for %pg = [%pg0 to %pg1 step %c1]({it_p}) -> ({ty}) {{')
    a('''          %p_base = index.mul %pg, %c16 : index
          %lo_pos = index.max %p_base, %st_pos : index
          %p_next = index.add %p_base, %c16 : index
          %hi_pos = index.min %p_next, %en_pos : index
          %n_in_pg = index.sub %hi_pos, %lo_pos : index
          %ko_base = func.call @g_kv_offset8(%origins, %row, %lo_pos, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)''')
    it_k = ", ".join(f"%km{h} = %pm{h} : f32, %kl{h} = %pl{h} : f32" for h in range(H)) + ", " + \
        ", ".join(f"%ka{h} = %pa{h} : vector<16xf32>" for h in range(H))
    rk = ", ".join(f"%nm{h}o, %nl{h}o" for h in range(H)) + ", " + ", ".join(f"%na{h}o" for h in range(H))
    a(f'          {rk} = scf.for %jj = [%c0 to %n_in_pg step %c4]({it_k}) -> ({ty}) {{')
    a('''            %j = index.add %jj, %ks : index
            %valid = index.cmp ult, %j, %n_in_pg : index
            %kp = index.add %lo_pos, %j : index
            %k16, %v16 = scf.if %valid -> (vector<16xf32>, vector<16xf32>) {
              %j_off = index.scale %j, %pos_bytes : index, offset -> offset
              %ko = index.add %ko_base, %j_off : offset
              %kc = buffer.view %arena[%ko] : buffer -> view<8x8x16xi8>
              %ksoff = index.constant 1024 : offset
              %kso = index.add %ko, %ksoff : offset
              %ksv = buffer.view %arena[%kso] : buffer -> view<8x4xf16>
              %kq = vector.load %kc[%kh, %dc, %c0] : view<8x8x16xi8> -> vector<16xi8>
              %kss16 = view.load %ksv[%kh, %sb] : view<8x4xf16> -> f16
              %kss = scalar.extf %kss16 : f16 to f32
              %kf = vector.sitofp %kq : vector<16xi8> to vector<16xf32>
              %ksp = vector.splat %kss : vector<16xf32>
              %kd = vector.mulf %kf, %ksp : vector<16xf32>
              %kh16 = vector.fptrunc %kd : vector<16xf32> to vector<16xf16>
              %kk = vector.extf %kh16 : vector<16xf16> to vector<16xf32>
              %vo = index.add %ko, %plane_stride : offset
              %vc = buffer.view %arena[%vo] : buffer -> view<8x8x16xi8>
              %vso = index.add %vo, %ksoff : offset
              %vsv = buffer.view %arena[%vso] : buffer -> view<8x4xf16>
              %vq = vector.load %vc[%kh, %dc, %c0] : view<8x8x16xi8> -> vector<16xi8>
              %vss16 = view.load %vsv[%kh, %sb] : view<8x4xf16> -> f16
              %vss = scalar.extf %vss16 : f16 to f32
              %vf = vector.sitofp %vq : vector<16xi8> to vector<16xf32>
              %vsp = vector.splat %vss : vector<16xf32>
              %vd = vector.mulf %vf, %vsp : vector<16xf32>
              %vh16 = vector.fptrunc %vd : vector<16xf32> to vector<16xf16>
              %vv = vector.extf %vh16 : vector<16xf16> to vector<16xf32>
              scf.yield %kk, %vv : vector<16xf32>, vector<16xf32>
            } else {
              scf.yield %zero16, %zero16 : vector<16xf32>, vector<16xf32>
            }
            %dist = index.sub %pos, %kp : index
            %in_ext0 = index.cmp ult, %dist, %cEXT : index
            %in_ext = scalar.andi %in_ext0, %valid : i1
            %rb2 = scf.if %in_ext -> (vector<2xf32>) {
              %db = index.assume %dist [range(%dist, 0, 1023)] : index
              %bv = vector.load %rbv[%db, %dc2] : view<1024x16xf32> -> vector<2xf32>
              scf.yield %bv : vector<2xf32>
            } else {
              scf.yield %zero2 : vector<2xf32>
            }''')
    for h in range(H):
        a(f'''            %pq{h} = vector.dotf<reassoc|contract> %q{h}, %k16, %zero_f : vector<16xf32>, vector<16xf32>, f32
            %pr{h} = vector.dotf<reassoc|contract> %r{h}, %rb2, %zero_f : vector<2xf32>, vector<2xf32>, f32
            %pp{h} = scalar.addf %pq{h}, %pr{h} : f32
            %b{h}_1r, %b{h}_1u = kernel.subgroup.shuffle<xor> %pp{h}, %x1, %w32 : f32, i32, i32
            %b{h}_1 = scalar.addf %pp{h}, %b{h}_1r : f32
            %b{h}_2r, %b{h}_2u = kernel.subgroup.shuffle<xor> %b{h}_1, %x2, %w32 : f32, i32, i32
            %b{h}_2 = scalar.addf %b{h}_1, %b{h}_2r : f32
            %b{h}_4r, %b{h}_4u = kernel.subgroup.shuffle<xor> %b{h}_2, %x4, %w32 : f32, i32, i32
            %sraw{h} = scalar.addf %b{h}_2, %b{h}_4r : f32
            %s{h} = scf.select %valid, %sraw{h}, %neg_inf : f32
            %nm{h} = scalar.maxnumf %km{h}, %s{h} : f32
            %ms{h} = scalar.subf %km{h}, %nm{h} : f32
            %al{h} = scalar.expf<afn> %ms{h} : f32
            %ds{h} = scalar.subf %s{h}, %nm{h} : f32
            %pe{h} = scalar.expf<afn> %ds{h} : f32
            %pv{h} = scf.select %valid, %pe{h}, %zero_f : f32
            %ls{h} = scalar.mulf %kl{h}, %al{h} : f32
            %nl{h} = scalar.addf %ls{h}, %pv{h} : f32
            %alv{h} = vector.splat %al{h} : vector<16xf32>
            %as{h} = vector.mulf %ka{h}, %alv{h} : vector<16xf32>
            %pvv{h} = vector.splat %pv{h} : vector<16xf32>
            %na{h} = vector.fmaf %pvv{h}, %v16, %as{h} : vector<16xf32>''')
    yk = ", ".join(f"%nm{h}, %nl{h}" for h in range(H)) + ", " + ", ".join(f"%na{h}" for h in range(H))
    a(f'            scf.yield {yk} : {ty}')
    a('          }')
    a(f'          scf.yield {rk} : {ty}')
    a('        }')
    a(f'        scf.yield {rp} : {ty}')
    a('      } else {')
    a(f'        scf.yield {init} : {ty}')
    a('      }')
    # merge the 4 key slots (lanes that share dc differ in bits 3 and 4), then store
    for h in range(H):
        a(f'''      %M{h}_8r, %M{h}_8u = kernel.subgroup.shuffle<xor> %m{h}f, %x8, %w32 : f32, i32, i32
      %M{h}_8 = scalar.maxnumf %m{h}f, %M{h}_8r : f32
      %M{h}_16r, %M{h}_16u = kernel.subgroup.shuffle<xor> %M{h}_8, %x16, %w32 : f32, i32, i32
      %M{h} = scalar.maxnumf %M{h}_8, %M{h}_16r : f32
      %e{h}_d = scalar.subf %m{h}f, %M{h} : f32
      %e{h} = scalar.expf<afn> %e{h}_d : f32
      %lw{h} = scalar.mulf %l{h}f, %e{h} : f32
      %L{h}_8r, %L{h}_8u = kernel.subgroup.shuffle<xor> %lw{h}, %x8, %w32 : f32, i32, i32
      %L{h}_8 = scalar.addf %lw{h}, %L{h}_8r : f32
      %L{h}_16r, %L{h}_16u = kernel.subgroup.shuffle<xor> %L{h}_8, %x16, %w32 : f32, i32, i32
      %L{h} = scalar.addf %L{h}_8, %L{h}_16r : f32
      %ev{h} = vector.splat %e{h} : vector<16xf32>
      %aw{h} = vector.mulf %a{h}f, %ev{h} : vector<16xf32>''')
        els = []
        for i in range(16):
            a(f'''      %aw{h}_{i} = vector.extract %aw{h}[{i}] : vector<16xf32> -> f32
      %t{h}_{i}_8r, %t{h}_{i}_8u = kernel.subgroup.shuffle<xor> %aw{h}_{i}, %x8, %w32 : f32, i32, i32
      %t{h}_{i}_8 = scalar.addf %aw{h}_{i}, %t{h}_{i}_8r : f32
      %t{h}_{i}_16r, %t{h}_{i}_16u = kernel.subgroup.shuffle<xor> %t{h}_{i}_8, %x16, %w32 : f32, i32, i32
      %t{h}_{i} = scalar.addf %t{h}_{i}_8, %t{h}_{i}_16r : f32''')
            els.append(f"%t{h}_{i}")
        a(f'      %A{h} = vector.from_elements {", ".join(els)} : vector<16xf32>')
    a('      scf.if %is_l0 {')
    for h in range(H):
        a(f'        view.store %M{h}, %pm_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>')
        a(f'        view.store %L{h}, %pl_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>')
    a('      }')
    # lane (ks, dc) writes dims dc*16 + 4*ks .. +4  ->  po lane index dc*4 + ks
    a('      %ol_v = index.add %dc4, %ks : index')
    a('      %ol = index.assume %ol_v [range(%ol_v, 0, 31)] : index')
    for i in range(4):
        a(f'      %ksel{i} = index.cmp eq, %ks, %c{i} : index')
        a(f'      scf.if %ksel{i} {{')
        for h in range(H):
            a(f'        %A{h}s{i} = vector.slice %A{h}[{4 * i}] : vector<16xf32> -> vector<4xf32>')
            a(f'        vector.store %A{h}s{i}, %po_v[%tok, %hd{h}, %sp, %ol, %c0] : vector<4xf32>, view<[%tokens]x24x64x32x4xf32>')
        a('      }')
    a('    }')
    a('  }')
    a('  kernel.return')
    a('}')
    return "\n".join(l for l in L if l is not None) + "\n"


SPAN_HEAD = """kernel.def export("g_ink_attn_split_span_v2") @g_ink_attn_split_span_v2(%spans: index) {
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant 64 : index
  %lanes = index.constant 32 : index
  kernel.launch.config workgroups(%spans, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index
} launch(%tokens_arg: index, %spans_arg: index, %blocks_arg: index, %pool_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 131072)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 1, 2147483584)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 47)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
"""

SPAN_PROLOGUE = """  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {
    %tok = index.assume %first [range(%first, 0, 8191), lt(%first, %tokens)] : index
    %start_b = scalar.assume %start32 [range(%start32, 0, 1048575)] : i32
    %pos = index.cast %start_b : i32 to index
    %w_low = index.add %c0, %c0 : index
    %n_keys_raw = index.add %pos, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index
"""


def span_kernel_text():
    """The v2 body for single-token spans (pure-decode epochs: every span has length 1)."""
    k = kernel_text()
    # header: from the def line through the %tok assume (single uses workgroup.id<x> as the token)
    h_end = k.index("  %kh_id = kernel.workgroup.id<y>")
    k = SPAN_HEAD + k[h_end:]
    # prologue: from the info view through the %n_keys assume
    p0 = k.index("  %info_v = buffer.view %info")
    p_live_end = k.index("  %is_l0 = index.cmp eq, %lane, %c0 : index\n") + len("  %is_l0 = index.cmp eq, %lane, %c0 : index\n")
    keep = "".join(l + "\n" for l in k[p0:p_live_end].split("\n") if "%pm_v" in l or "%pl_v" in l or "%po_v" in l or "%is_l0" in l)
    n_end = k.index("    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index\n")
    n_end += len("    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index\n")
    k = k[:p0] + keep + SPAN_PROLOGUE + k[n_end:]
    assert "%info" not in k and "%sliding_arg" not in k and "%live" not in k
    return k


def patch_dispatch(text):
    """At each span call site: pure-decode epoch (t == spans) -> v2 span kernel, otherwise the stock one."""
    import re
    n = 0
    def rep(m):
        nonlocal n
        n += 1
        ind, line = m.group(1), m.group(2)
        v2 = line.replace("@g_ink_attn_split_span[", "@g_ink_attn_split_span_v2[", 1)
        return (f"{ind}%v2dec_{n} = index.cmp eq, %t, %spans : index\n"
                f"{ind}scf.if %v2dec_{n} {{\n{ind}  {v2}\n{ind}}} else {{\n{ind}  {line}\n{ind}}}\n")
    text = re.sub(r"^( *)(kernel\.launch @g_ink_attn_split_span\[%spans\]\(.*)\n", rep, text, flags=re.M)
    return text, n


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    span = "--span" in sys.argv
    text = open(src).read()
    name = "g_ink_attn_split"
    s = text.index(f'kernel.def export("{name}") @{name}(')
    e = text.index("\nkernel.def ", s + 1) + 1
    text = text[:s] + kernel_text() + "\n" + (span_kernel_text() + "\n" if span else "") + text[e:]
    if span:
        text, n = patch_dispatch(text)
        assert n == 16, n
        print(f"added g_ink_attn_split_span_v2 and patched {n} span call sites", file=sys.stderr)
    open(dst, "w").write(text)
    print(f"replaced {name} with v2 ({len(kernel_text())} chars)", file=sys.stderr)
