#!/usr/bin/env python3
"""gen_fuse_nq.py SRC.loom OUT.loom : fuse  g_ink_rms_norm_plane(..., X, W, H)  +  g_quant_q8(%t, %nH, H, Q)  into one launch of
g_ink_rms_norm_plane_q8x (decode and small-batch paths). Bit-identical by construction:
  - the norm part is the stock kernel unchanged (same thread layout, same reduction, same stores of H, which other kernels read);
  - in its second loop each lane holds 4 of the 32 values of one q8 group, and a group sits in 8 adjacent lanes. The group's
    abs-max and integer code sum are combined with 3 xor shuffles (max and integer sums are exact in any order), then every lane
    quantizes its own 4 values with the stock g_quant_q8 arithmetic (same divide, round, f16 scale and sum metadata).
(Loom already ships an unused g_ink_rms_norm_plane_q8 with a different reduction and rsqrt<afn>, so it is not bit-exact; hence _q8x.)
Only rewrites a norm launch whose next line is the q8 quant of the same output buffer with width %nH (2304).
"""
import re, sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read().split("\n")

FUSED = r'''kernel.def export("g_ink_rms_norm_plane_q8x") @g_ink_rms_norm_plane_q8x(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %planes_arg: index, %idx_arg: index, %x: buffer, %w: buffer, %out: buffer, %xq: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %pl = index.assume %planes_arg [range(%planes_arg, 1, 4)] : index
  %pi = index.assume %idx_arg [range(%idx_arg, 0, 3), lt(%idx_arg, %pl)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, 255)] : index
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %c256 = index.constant 256 : index
  %cH4 = index.constant 576 : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %inv_H = scalar.constant 0.00043402777777777775 : f32
  %c127 = scalar.constant 127.0 : f32
  %bd4 = scalar.constant 4 : i32
  %bd2 = scalar.constant 2 : i32
  %bd1 = scalar.constant 1 : i32
  %bw32 = scalar.constant 32 : i32
  %zero1i = vector.splati 0 : vector<1xi32>
  %ones4 = vector.splati 1 : vector<4xi8>
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %xv = buffer.view %x[%zero] : buffer -> view<[%pl]x[%tokens]x576x4xf32>
  %wv = buffer.view %w[%zero] : buffer -> view<576x4xf32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x576x4xf32>
  %qv = buffer.view %xq[%zero] : buffer -> view<[%tokens]x576xi32>
  %c2304 = index.constant 2304 : offset
  %m_off = index.scale %tokens, %c2304 : index, offset -> offset
  %mv = buffer.view %xq[%m_off] : buffer -> view<[%tokens]x72x2xf32>
  %sq = scf.for %i = [%tid to %cH4 step %c256](%a = %zero_f : f32) -> (f32) {
    %ch = index.assume %i [range(%i, 0, 575)] : index
    %v = vector.load %xv[%pi, %tok, %ch, %c0] : view<[%pl]x[%tokens]x576x4xf32> -> vector<4xf32>
    %na = vector.dotf<reassoc|contract> %v, %v, %a : vector<4xf32>, vector<4xf32>, f32
    scf.yield %na : f32
  }
  %tot = kernel.workgroup.reduce<addf> %sq : f32
  %mn = scalar.mulf %tot, %inv_H : f32
  %vr = scalar.addf %mn, %eps : f32
  %ir = scalar.rsqrtf %vr : f32
  %ir4 = vector.splat %ir : vector<4xf32>
  scf.for %i = [%tid to %cH4 step %c256] {
    %ch = index.assume %i [range(%i, 0, 575)] : index
    %v = vector.load %xv[%pi, %tok, %ch, %c0] : view<[%pl]x[%tokens]x576x4xf32> -> vector<4xf32>
    %gw = vector.load %wv[%ch, %c0] : view<576x4xf32> -> vector<4xf32>
    %nv = vector.mulf %v, %ir4 : vector<4xf32>
    %y = vector.mulf %nv, %gw : vector<4xf32>
    vector.store %y, %ov[%tok, %ch, %c0] : vector<4xf32>, view<[%tokens]x576x4xf32>
    %ya = vector.absf %y : vector<4xf32>
    %lm = vector.reduce<maxnumf> %ya, %zero_f : vector<4xf32>, f32
    %m4r, %m4u = kernel.subgroup.shuffle<xor> %lm, %bd4, %bw32 : f32, i32, i32
    %m4 = scalar.maxnumf %lm, %m4r : f32
    %m2r, %m2u = kernel.subgroup.shuffle<xor> %m4, %bd2, %bw32 : f32, i32, i32
    %m2 = scalar.maxnumf %m4, %m2r : f32
    %m1r, %m1u = kernel.subgroup.shuffle<xor> %m2, %bd1, %bw32 : f32, i32, i32
    %amax = scalar.maxnumf %m2, %m1r : f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<4xf32>
    %sc = vector.divf %y, %dv : vector<4xf32>
    %rd = vector.roundf %sc : vector<4xf32>
    %nzv = vector.splat %nz : vector<4xi1>
    %rs = vector.select %nzv, %rd, %zero4 : vector<4xf32>
    %q = vector.fptosi %rs : vector<4xf32> to vector<4xi8>
    %qw = vector.bitcast %q : vector<4xi8> to vector<1xi32>
    vector.store %qw, %qv[%tok, %ch] : vector<1xi32>, view<[%tokens]x576xi32>
    %qs = vector.dot4i<s8s8> %q, %ones4, %zero1i : vector<4xi8>, vector<4xi8>, vector<1xi32>
    %qsi = vector.extract %qs[0] : vector<1xi32> -> i32
    %ls = scalar.sitofp %qsi : i32 to f32
    %s4r, %s4u = kernel.subgroup.shuffle<xor> %ls, %bd4, %bw32 : f32, i32, i32
    %s4 = scalar.addf %ls, %s4r : f32
    %s2r, %s2u = kernel.subgroup.shuffle<xor> %s4, %bd2, %bw32 : f32, i32, i32
    %s2 = scalar.addf %s4, %s2r : f32
    %s1r, %s1u = kernel.subgroup.shuffle<xor> %s2, %bd1, %bw32 : f32, i32, i32
    %qsf = scalar.addf %s2, %s1r : f32
    %j = index.rem %ch, %c8 : index
    %lead = index.cmp eq, %j, %c0 : index
    scf.if %lead {
      %g_raw = index.div %ch, %c8 : index
      %g = index.assume %g_raw [range(%g_raw, 0, 71)] : index
      %dh = scalar.fptrunc %d : f32 to f16
      %da = scalar.extf %dh : f16 to f32
      %meta = vector.from_elements %da, %qsf : vector<2xf32>
      vector.store %meta, %mv[%tok, %g, %c0] : vector<2xf32>, view<[%tokens]x72x2xf32>
    }
  }
  kernel.return
}
'''

NORM = re.compile(r"^(\s*)kernel\.launch @g_ink_rms_norm_plane\[(%\w+)\]\((%\w+), (%\w+), (%\w+), (%\w+), (%\w+), (%\w+)\) : "
                  r"\[index\]\(index, index, index, buffer, (view<2304xf32>), buffer\)\s*$")
QUANT = re.compile(r"^\s*kernel\.launch @g_quant_q8\[(%\w+), %nH\]\((%\w+), %nH, (%\w+), (%\w+)\) : \[index, index\]\(index, index, buffer, buffer\)\s*$")

res, i, n, added = [], 0, 0, False
while i < len(text):
    ln = text[i]
    if not added and ln.startswith("command.def"):
        res.extend(FUSED.split("\n")); added = True
    m = NORM.match(ln)
    q = QUANT.match(text[i + 1]) if m and i + 1 < len(text) else None
    if m and q and q.group(3) == m.group(8) and q.group(2) == m.group(3):
        ind, grid, t, pl, pi, x, w, h, wty = m.groups()
        res.append(f"{ind}kernel.launch @g_ink_rms_norm_plane_q8x[{grid}]({t}, {pl}, {pi}, {x}, {w}, {h}, {q.group(4)}) : "
                   f"[index](index, index, index, buffer, {wty}, buffer, buffer)")
        n += 1; i += 2; continue
    res.append(ln); i += 1
assert added
open(dst, "w").write("\n".join(res))
print("fused norm+quant launch sites:", n)
