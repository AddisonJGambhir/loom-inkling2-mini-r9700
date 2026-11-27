#!/usr/bin/env python3
"""gen_small_decode.py SRC.loom DST.loom : optimize the hottest latency-bound small decode kernels
(336 launches/token):
  1. g_ink_ctrl_apply (96x/tok, 113.2 MB/tok):
     Stock maps 1 thread/channel loading 6 x vector<32xf32> (192 VGPRs/thread, 256-byte uncoalesced
     stride between adjacent lanes). Replaced with 16 lanes/channel loading coalesced 16-byte
     vector<4xf32> (12 VGPRs/thread, 0 LDS, 0 barriers, 4-step XOR shuffle).
  2. g_ink_mix2, g_ink_mix3, g_ink_mix4, g_ink_mix5 (144x/tok):
     Stock runs 4, 6, 8, or 10 sequential kernel.workgroup.reduce<addf> calls (8-20 barriers/launch).
     Replaced with subgroup.reduce<addf> + 1 tiny LDS array (8 x M floats) + 1 single barrier.
  3. g_ink_concat_rms (96x/tok):
     Stock reads cur and msg twice from global memory and runs 2 sequential workgroup reduces.
     Replaced with single-pass register caching (3 vec4s per thread, matching g_ink_mix2) + 1 barrier.
"""
import re
import sys


CTRL_APPLY_V2 = """// Coalesced controller apply: 16 lanes/channel (4 channels/WG = 576 WGs), 16-byte coalesced loads, 0 LDS/barriers.
kernel.def export("g_ink_ctrl_apply") @g_ink_ctrl_apply(%tokens: index) {
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %cb = index.constant 576 : index
  kernel.launch.config workgroups(%tokens, %cb, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%tokens_arg: index, %c_planes_arg: index, %c_idx_arg: index, %m_planes_arg: index, %m_idx_arg: index, %cur: buffer, %msg: buffer, %z: buffer, %w_gain: buffer, %w_forget: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %cp = index.assume %c_planes_arg [range(%c_planes_arg, 1, 4)] : index
  %ci = index.assume %c_idx_arg [range(%c_idx_arg, 0, 3), lt(%c_idx_arg, %cp)] : index
  %mp = index.assume %m_planes_arg [range(%m_planes_arg, 1, 4)] : index
  %mi = index.assume %m_idx_arg [range(%m_idx_arg, 0, 3), lt(%m_idx_arg, %mp)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %blk_id = kernel.workgroup.id<y> : index
  %blk = index.assume %blk_id [range(%blk_id, 0, 575)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, 63)] : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %x1 = scalar.constant 1 : i32
  %x2 = scalar.constant 2 : i32
  %x4 = scalar.constant 4 : i32
  %x8 = scalar.constant 8 : i32
  %w32 = scalar.constant 32 : i32
  %ch_loc = index.div %tid, %c16 : index
  %sub_v = index.rem %tid, %c16 : index
  %sub = index.assume %sub_v [range(%sub_v, 0, 15)] : index
  %ch_base = index.mul %blk, %c4 : index
  %ch_raw = index.add %ch_base, %ch_loc : index
  %ch = index.assume %ch_raw [range(%ch_raw, 0, 2303)] : index
  %zv = buffer.view %z[%zero] : buffer -> view<[%tokens]x16x4xf32>
  %wgv = buffer.view %w_gain[%zero] : buffer -> view<2304x16x4xf32>
  %wfv = buffer.view %w_forget[%zero] : buffer -> view<2304x16x4xf32>
  %z4 = vector.load %zv[%tok, %sub, %c0] : view<[%tokens]x16x4xf32> -> vector<4xf32>
  %g4 = vector.load %wgv[%ch, %sub, %c0] : view<2304x16x4xf32> -> vector<4xf32>
  %f4 = vector.load %wfv[%ch, %sub, %c0] : view<2304x16x4xf32> -> vector<4xf32>
  %gd0 = vector.dotf<reassoc|contract> %g4, %z4, %zero_f : vector<4xf32>, vector<4xf32>, f32
  %fd0 = vector.dotf<reassoc|contract> %f4, %z4, %zero_f : vector<4xf32>, vector<4xf32>, f32
  %gd1r, %gd1u = kernel.subgroup.shuffle<xor> %gd0, %x1, %w32 : f32, i32, i32
  %gd1 = scalar.addf %gd0, %gd1r : f32
  %gd2r, %gd2u = kernel.subgroup.shuffle<xor> %gd1, %x2, %w32 : f32, i32, i32
  %gd2 = scalar.addf %gd1, %gd2r : f32
  %gd4r, %gd4u = kernel.subgroup.shuffle<xor> %gd2, %x4, %w32 : f32, i32, i32
  %gd4 = scalar.addf %gd2, %gd4r : f32
  %gd8r, %gd8u = kernel.subgroup.shuffle<xor> %gd4, %x8, %w32 : f32, i32, i32
  %gd = scalar.addf %gd4, %gd8r : f32
  %fd1r, %fd1u = kernel.subgroup.shuffle<xor> %fd0, %x1, %w32 : f32, i32, i32
  %fd1 = scalar.addf %fd0, %fd1r : f32
  %fd2r, %fd2u = kernel.subgroup.shuffle<xor> %fd1, %x2, %w32 : f32, i32, i32
  %fd2 = scalar.addf %fd1, %fd2r : f32
  %fd4r, %fd4u = kernel.subgroup.shuffle<xor> %fd2, %x4, %w32 : f32, i32, i32
  %fd4 = scalar.addf %fd2, %fd4r : f32
  %fd8r, %fd8u = kernel.subgroup.shuffle<xor> %fd4, %x8, %w32 : f32, i32, i32
  %fd = scalar.addf %fd4, %fd8r : f32
  %is_lead = index.cmp eq, %sub, %c0 : index
  scf.if %is_lead {
    %cv = buffer.view %cur[%zero] : buffer -> view<[%cp]x[%tokens]x2304xf32>
    %mv = buffer.view %msg[%zero] : buffer -> view<[%mp]x[%tokens]x2304xf32>
    %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x2304xf32>
    %gain = scalar.tanhf<afn> %gd : f32
    %forget = scalar.tanhf<afn> %fd : f32
    %keep = scalar.subf %one_f, %forget : f32
    %xc = view.load %cv[%ci, %tok, %ch] : view<[%cp]x[%tokens]x2304xf32> -> f32
    %xm = view.load %mv[%mi, %tok, %ch] : view<[%mp]x[%tokens]x2304xf32> -> f32
    %p0 = scalar.mulf %keep, %xc : f32
    %y = scalar.fmaf %gain, %xm, %p0 : f32
    view.store %y, %ov[%tok, %ch] : f32, view<[%tokens]x2304xf32>
  }
  kernel.return
}
"""

CONCAT_RMS_V2 = """// Single-pass register-cached concat RMS norm with 1-barrier batched reduction.
kernel.def export("g_ink_concat_rms") @g_ink_concat_rms(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %c_planes_arg: index, %c_idx_arg: index, %m_planes_arg: index, %m_idx_arg: index, %cur: buffer, %msg: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %cp = index.assume %c_planes_arg [range(%c_planes_arg, 1, 4)] : index
  %ci = index.assume %c_idx_arg [lt(%c_idx_arg, %cp)] : index
  %mp = index.assume %m_planes_arg [range(%m_planes_arg, 1, 4)] : index
  %mi = index.assume %m_idx_arg [lt(%m_idx_arg, %mp)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, 255)] : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 7)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %c0 = index.constant 0 : index
  %c256 = index.constant 256 : index
  %c512 = index.constant 512 : index
  %cH4 = index.constant 576 : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %inv_H = scalar.constant 0.00043402777777777775 : f32
  %cv = buffer.view %cur[%zero] : buffer -> view<[%cp]x[%tokens]x576x4xf32>
  %mv = buffer.view %msg[%zero] : buffer -> view<[%mp]x[%tokens]x576x4xf32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x1152x4xf32>
  %i0 = index.assume %tid [range(%tid, 0, 575)] : index
  %i1_raw = index.add %tid, %c256 : index
  %i1 = index.assume %i1_raw [range(%i1_raw, 0, 575)] : index
  %i2_raw = index.add %tid, %c512 : index
  %has2 = index.cmp ult, %i2_raw, %cH4 : index
  %xc0 = vector.load %cv[%ci, %tok, %i0, %c0] : view<[%cp]x[%tokens]x576x4xf32> -> vector<4xf32>
  %xm0 = vector.load %mv[%mi, %tok, %i0, %c0] : view<[%mp]x[%tokens]x576x4xf32> -> vector<4xf32>
  %xc1 = vector.load %cv[%ci, %tok, %i1, %c0] : view<[%cp]x[%tokens]x576x4xf32> -> vector<4xf32>
  %xm1 = vector.load %mv[%mi, %tok, %i1, %c0] : view<[%mp]x[%tokens]x576x4xf32> -> vector<4xf32>
  %xc2, %xm2 = scf.if %has2 -> (vector<4xf32>, vector<4xf32>) {
    %i2 = index.assume %i2_raw [range(%i2_raw, 0, 575)] : index
    %a = vector.load %cv[%ci, %tok, %i2, %c0] : view<[%cp]x[%tokens]x576x4xf32> -> vector<4xf32>
    %b = vector.load %mv[%mi, %tok, %i2, %c0] : view<[%mp]x[%tokens]x576x4xf32> -> vector<4xf32>
    scf.yield %a, %b : vector<4xf32>, vector<4xf32>
  } else {
    scf.yield %zero4, %zero4 : vector<4xf32>, vector<4xf32>
  }
  %sc0 = vector.dotf<reassoc|contract> %xc0, %xc0, %zero_f : vector<4xf32>, vector<4xf32>, f32
  %sc1 = vector.dotf<reassoc|contract> %xc1, %xc1, %sc0 : vector<4xf32>, vector<4xf32>, f32
  %sc2 = vector.dotf<reassoc|contract> %xc2, %xc2, %sc1 : vector<4xf32>, vector<4xf32>, f32
  %sm0 = vector.dotf<reassoc|contract> %xm0, %xm0, %zero_f : vector<4xf32>, vector<4xf32>, f32
  %sm1 = vector.dotf<reassoc|contract> %xm1, %xm1, %sm0 : vector<4xf32>, vector<4xf32>, f32
  %sm2 = vector.dotf<reassoc|contract> %xm2, %xm2, %sm1 : vector<4xf32>, vector<4xf32>, f32
  %wc = kernel.subgroup.reduce<addf> %sc2 : f32
  %wm = kernel.subgroup.reduce<addf> %sm2 : f32
  %rb = index.constant 64 : offset
  %rlds = buffer.alloca<workgroup> align(16) %rb : buffer
  %rv = buffer.view %rlds[%zero] : buffer -> view<8x2xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  scf.if %is_l0 {
    view.store %wc, %rv[%wave, 0] : f32, view<8x2xf32>
    view.store %wm, %rv[%wave, 1] : f32, view<8x2xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %c_0 = view.load %rv[0, 0] : view<8x2xf32> -> f32
  %c_1 = view.load %rv[1, 0] : view<8x2xf32> -> f32
  %c_2 = view.load %rv[2, 0] : view<8x2xf32> -> f32
  %c_3 = view.load %rv[3, 0] : view<8x2xf32> -> f32
  %c_4 = view.load %rv[4, 0] : view<8x2xf32> -> f32
  %c_5 = view.load %rv[5, 0] : view<8x2xf32> -> f32
  %c_6 = view.load %rv[6, 0] : view<8x2xf32> -> f32
  %c_7 = view.load %rv[7, 0] : view<8x2xf32> -> f32
  %cs0 = scalar.addf %c_0, %c_1 : f32
  %cs1 = scalar.addf %c_2, %c_3 : f32
  %cs2 = scalar.addf %c_4, %c_5 : f32
  %cs3 = scalar.addf %c_6, %c_7 : f32
  %cs4 = scalar.addf %cs0, %cs1 : f32
  %cs5 = scalar.addf %cs2, %cs3 : f32
  %tot_c = scalar.addf %cs4, %cs5 : f32
  %m_0 = view.load %rv[0, 1] : view<8x2xf32> -> f32
  %m_1 = view.load %rv[1, 1] : view<8x2xf32> -> f32
  %m_2 = view.load %rv[2, 1] : view<8x2xf32> -> f32
  %m_3 = view.load %rv[3, 1] : view<8x2xf32> -> f32
  %m_4 = view.load %rv[4, 1] : view<8x2xf32> -> f32
  %m_5 = view.load %rv[5, 1] : view<8x2xf32> -> f32
  %m_6 = view.load %rv[6, 1] : view<8x2xf32> -> f32
  %m_7 = view.load %rv[7, 1] : view<8x2xf32> -> f32
  %ms0 = scalar.addf %m_0, %m_1 : f32
  %ms1 = scalar.addf %m_2, %m_3 : f32
  %ms2 = scalar.addf %m_4, %m_5 : f32
  %ms3 = scalar.addf %m_6, %m_7 : f32
  %ms4 = scalar.addf %ms0, %ms1 : f32
  %ms5 = scalar.addf %ms2, %ms3 : f32
  %tot_m = scalar.addf %ms4, %ms5 : f32
  %mc = scalar.mulf %tot_c, %inv_H : f32
  %mm = scalar.mulf %tot_m, %inv_H : f32
  %vc = scalar.addf %mc, %eps : f32
  %vm = scalar.addf %mm, %eps : f32
  %irc = scalar.rsqrtf %vc : f32
  %irm = scalar.rsqrtf %vm : f32
  %irc4 = vector.splat %irc : vector<4xf32>
  %irm4 = vector.splat %irm : vector<4xf32>
  %yc0 = vector.mulf %xc0, %irc4 : vector<4xf32>
  %ym0 = vector.mulf %xm0, %irm4 : vector<4xf32>
  %i0_m_v = index.add %i0, %cH4 : index
  %i0_m = index.assume %i0_m_v [range(%i0_m_v, 576, 1151)] : index
  vector.store %yc0, %ov[%tok, %i0, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
  vector.store %ym0, %ov[%tok, %i0_m, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
  %yc1 = vector.mulf %xc1, %irc4 : vector<4xf32>
  %ym1 = vector.mulf %xm1, %irm4 : vector<4xf32>
  %i1_m_v = index.add %i1, %cH4 : index
  %i1_m = index.assume %i1_m_v [range(%i1_m_v, 576, 1151)] : index
  vector.store %yc1, %ov[%tok, %i1, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
  vector.store %ym1, %ov[%tok, %i1_m, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
  scf.if %has2 {
    %i2 = index.assume %i2_raw [range(%i2_raw, 0, 575)] : index
    %i2_m_v = index.add %i2, %cH4 : index
    %i2_m = index.assume %i2_m_v [range(%i2_m_v, 576, 1151)] : index
    %yc2 = vector.mulf %xc2, %irc4 : vector<4xf32>
    %ym2 = vector.mulf %xm2, %irm4 : vector<4xf32>
    vector.store %yc2, %ov[%tok, %i2, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
    vector.store %ym2, %ov[%tok, %i2_m, %c0] : vector<4xf32>, view<[%tokens]x1152x4xf32>
  }
  kernel.return
}
"""


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


def batch_mix_reduces(body, expected_n):
    lines = body.splitlines(keepends=True)
    red_re = re.compile(r"^  (%[A-Za-z0-9_]+) = kernel\.workgroup\.reduce<addf> (%[A-Za-z0-9_]+) : f32\n$")
    hits = [(i, red_re.match(ln)) for i, ln in enumerate(lines)]
    matches = [(i, m.group(1), m.group(2)) for i, m in hits if m]
    if len(matches) != expected_n:
        raise ValueError(f"expected {expected_n} workgroup.reduce<addf>, found {len(matches)}")
    # All sq*/dt* inputs are already computed before `first`; emit one batched 1-barrier reduction at `first`
    # and remove the individual workgroup.reduce lines.
    first = matches[0][0]
    match_indices = {i for i, _, _ in matches}
    M = expected_n
    b = []
    b.append("  // Batched 1-barrier workgroup reduction for all states.\n")
    b.append("  %mr_wave_r = kernel.subgroup.id : index\n")
    b.append("  %mr_wave = index.assume %mr_wave_r [range(%mr_wave_r, 0, 7)] : index\n")
    b.append("  %mr_lane_r = kernel.subgroup.lane.id : index\n")
    b.append("  %mr_lane = index.assume %mr_lane_r [range(%mr_lane_r, 0, 31)] : index\n")
    b.append(f"  %mr_bytes = index.constant {8 * M * 4} : offset\n")
    b.append("  %mr_lds = buffer.alloca<workgroup> align(16) %mr_bytes : buffer\n")
    b.append(f"  %mr_v = buffer.view %mr_lds[%zero] : buffer -> view<8x{M}xf32>\n")
    for idx, (_, _, src) in enumerate(matches):
        b.append(f"  %mr_w{idx} = kernel.subgroup.reduce<addf> {src} : f32\n")
    b.append("  %mr_l0 = index.cmp eq, %mr_lane, %c0 : index\n")
    b.append("  scf.if %mr_l0 {\n")
    for idx in range(M):
        b.append(f"    view.store %mr_w{idx}, %mr_v[%mr_wave, {idx}] : f32, view<8x{M}xf32>\n")
    b.append("  }\n")
    b.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n")
    for idx, (_, dst, _) in enumerate(matches):
        for w in range(8):
            b.append(f"  %mr_s{idx}_{w} = view.load %mr_v[{w}, {idx}] : view<8x{M}xf32> -> f32\n")
        b.append(f"  %mr_a{idx}_0 = scalar.addf %mr_s{idx}_0, %mr_s{idx}_1 : f32\n")
        b.append(f"  %mr_a{idx}_1 = scalar.addf %mr_s{idx}_2, %mr_s{idx}_3 : f32\n")
        b.append(f"  %mr_a{idx}_2 = scalar.addf %mr_s{idx}_4, %mr_s{idx}_5 : f32\n")
        b.append(f"  %mr_a{idx}_3 = scalar.addf %mr_s{idx}_6, %mr_s{idx}_7 : f32\n")
        b.append(f"  %mr_a{idx}_4 = scalar.addf %mr_a{idx}_0, %mr_a{idx}_1 : f32\n")
        b.append(f"  %mr_a{idx}_5 = scalar.addf %mr_a{idx}_2, %mr_a{idx}_3 : f32\n")
        b.append(f"  {dst} = scalar.addf %mr_a{idx}_4, %mr_a{idx}_5 : f32\n")
    out = []
    for i, ln in enumerate(lines):
        if i == first:
            out.extend(b)
        if i in match_indices:
            continue
        out.append(ln)
    return "".join(out)


def patch(text):
    if "// Coalesced controller apply:" in text:
        raise ValueError("already patched by gen_small_decode.py")

    # 1. Replace g_ink_ctrl_apply
    lo, hi = find_kernel_span(text, "g_ink_ctrl_apply")
    text = text[:lo] + CTRL_APPLY_V2 + text[hi:]

    # 2. Replace g_ink_concat_rms
    lo, hi = find_kernel_span(text, "g_ink_concat_rms")
    text = text[:lo] + CONCAT_RMS_V2 + text[hi:]

    # 3. Collapse workgroup.reduce<addf> in g_ink_mix2..5 to 1 barrier each
    for k in (2, 3, 4, 5):
        name = f"g_ink_mix{k}"
        lo, hi = find_kernel_span(text, name)
        text = text[:lo] + batch_mix_reduces(text[lo:hi], 2 * k) + text[hi:]

    return text


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_small_decode.py SRC.loom DST.loom")
    patched = patch(open(sys.argv[1]).read())
    open(sys.argv[2], "w").write(patched)
    print("patched g_ink_ctrl_apply (coalesced 16-lane), g_ink_concat_rms (single-pass 1-barrier), and g_ink_mix2..5 (1-barrier)", file=sys.stderr)


if __name__ == "__main__":
    main()
