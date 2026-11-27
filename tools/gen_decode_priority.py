#!/usr/bin/env python3
"""Generate dynamic engine-level decode-priority interleaving in prepare.loom + control.loom.

Solves the concurrent prefill-stall problem inside Loom's own epoch hooks without modifying
loom-serve:
  - Solo prefill (1 span, or only big prefill spans): untouched -> 0% overhead, full 2048 chunks.
  - Pure concurrent decode (no big prefill spans): untouched -> full decode speed.
  - Vision/media epochs, speculative epochs, or draft_depth > 0: untouched.
  - Mixed epoch (at least one short/decode ordinary span <= small_max AND at least one big
    ordinary prefill span > big_thresh):
      * On (cadence - 1) of every `cadence` mixed epochs:
          defer big prefill spans (length=0, consumed=0, selected=0) and switch @epoch to the
          smallest compiled stage fitting (total_short_tokens, spans) -> ~11 ms decode epochs.
      * On 1 of every `cadence` mixed epochs:
          cap big prefill spans to (budget - small_tokens) // big_count and switch @epoch to
          the smallest compiled stage fitting (total_packed_tokens, spans) (<= budget).
      * When exiting mixed mode, resets the cadence counter to 1 (for cadence >= 2) so the
        very first epoch of a new mixed burst serves active decode / newly arrived short
        prompts immediately without waiting behind a prefill slice.

Control-state scratch layout (completely unused by stock prepare.loom / control.loom,
strictly between gate_view [1700..1828) and per-row sampling records [2048..)):
  - [1840..2020): 9x5xi32 stage-tag lookup table (tag = shape_index + 1, or 0 if none)
                  tb in 0..8 -> t_need in {2, 4, 8, 16, 32, 64, 128, 256, 512}
                  sb in 0..4 -> s_need in {2, 4, 8, 16, 32}
  - [2020..2036): 4xi32 runtime state:
                  [0] = cadence counter (0 = prefill slice, 1..cadence-1 = decode only)
                  [1] = active flag for current epoch (read by @plain_publish_epoch)
                  [2] = active cap for current epoch (read by @plain_publish_epoch)
                  [3] = reserved (0)
"""
import argparse
import sys


DP_PREP_HELPER_BEGIN = "// dp:helper:begin"
DP_PREP_HELPER_END = "// dp:helper:end"
DP_PREP_TABLE_BEGIN = "// dp:table:begin"
DP_PREP_TABLE_END = "// dp:table:end"
DP_CTRL_ENC_BEGIN = "// dp:encode:begin"
DP_CTRL_ENC_END = "// dp:encode:end"
DP_CTRL_PUB_BEGIN = "// dp:publish:begin"
DP_CTRL_PUB_END = "// dp:publish:end"


def patch_prepare(src: str) -> str:
    if DP_PREP_HELPER_BEGIN in src or DP_PREP_TABLE_BEGIN in src:
        raise ValueError("prepare.loom already contains dp markers; patch from clean stock source")

    helper = f"""{DP_PREP_HELPER_BEGIN}
func.def @dp_find_shape(%shapes: buffer, %count: index, %t_need: i64, %s_need: i64) -> (i32) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %shift16 = scalar.constant 16 : i64
  %big = scalar.constant 9223372036854775807 : i64
  %shape_view = buffer.view %shapes[%origin] : buffer -> view<[%count]x2xi64>
  %best_tag, %best_cost = scf.for %i = [%c0 to %count step %c1](%acc_tag = %c0_i32 : i32, %acc_cost = %big : i64) -> (i32, i64) {{
    %t = view.load %shape_view[%i, 0] : view<[%count]x2xi64> -> i64
    %s = view.load %shape_view[%i, 1] : view<[%count]x2xi64> -> i64
    %tok_ok = scalar.cmpi uge, %t, %t_need : i64
    %span_ok = scalar.cmpi uge, %s, %s_need : i64
    %fits = scalar.andi %tok_ok, %span_ok : i1
    %t_hi = scalar.shli %t, %shift16 : i64
    %cost = scalar.ori %t_hi, %s : i64
    %better = scalar.cmpi ult, %cost, %acc_cost : i64
    %take = scalar.andi %fits, %better : i1
    %i32_v = index.cast %i : index to i32
    %tag1 = scalar.addi %i32_v, %c1_i32 : i32
    %next_tag = scf.select %take, %tag1, %acc_tag : i32
    %next_cost = scf.select %take, %cost, %acc_cost : i64
    scf.yield %next_tag, %next_cost : i32, i64
  }}
  func.return %best_tag : i32
}}
{DP_PREP_HELPER_END}

func.def public @prepare("""

    anchor_def = "func.def public @prepare("
    if src.count(anchor_def) != 1:
        raise ValueError("expected exactly one @prepare definition in prepare.loom")
    out = src.replace(anchor_def, helper, 1)

    old_tail = """  %terminal = global.load @terminal : buffer
  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %draft, %geometry : i64, i32, buffer, buffer, buffer, buffer, buffer, buffer, buffer"""

    new_tail = f"""  {DP_PREP_TABLE_BEGIN}
  // Decode-priority stage lookup table at %state[1840..2020) (9x5xi32):
  // tb in 0..8 -> t_need = 2 << tb in {{2, 4, 8, 16, 32, 64, 128, 256, 512}}
  // sb in 0..4 -> s_need = 2 << sb in {{2, 4, 8, 16, 32}}
  %dp_at = index.constant 1840 : offset
  %dp_tbl = buffer.view %state[%dp_at] : buffer -> view<9x5xi32>
  %dp_c9 = index.constant 9 : index
  %dp_c5 = index.constant 5 : index
  %dp_c2_64 = scalar.constant 2 : i64
  scf.for %tb = [%c0 to %dp_c9 step %c1] {{
    %tb64 = index.cast %tb : index to i64
    %t_need = scalar.shli %dp_c2_64, %tb64 : i64
    scf.for %sb = [%c0 to %dp_c5 step %c1] {{
      %sb64 = index.cast %sb : index to i64
      %s_need = scalar.shli %dp_c2_64, %sb64 : i64
      %best = func.call @dp_find_shape(%shapes, %count, %t_need, %s_need) : (buffer, index, i64, i64) -> (i32)
      view.store %best, %dp_tbl[%tb, %sb] : i32, view<9x5xi32>
    }}
  }}
  {DP_PREP_TABLE_END}
  %terminal = global.load @terminal : buffer
  func.return %effective, %stage_count32, %state, %terminal, %allocations, %views, %origins, %draft, %geometry : i64, i32, buffer, buffer, buffer, buffer, buffer, buffer, buffer"""

    if out.count(old_tail) != 1:
        raise ValueError("expected exact return tail in prepare.loom")
    return out.replace(old_tail, new_tail, 1)


def patch_control(
    src: str,
    cadence: int = 5,
    budget: int = 512,
    small_max: int = 64,
    big_thresh: int = 128,
) -> str:
    if DP_CTRL_ENC_BEGIN in src or DP_CTRL_PUB_BEGIN in src:
        raise ValueError("control.loom already contains dp markers; patch from clean stock source")
    if cadence < 1 or cadence > 16:
        raise ValueError(f"cadence must be in 1..16, got {cadence}")
    if budget not in (256, 512):
        raise ValueError(f"budget must be 256 or 512, got {budget}")
    if small_max < 1 or small_max > 128:
        raise ValueError(f"small_max must be in 1..128, got {small_max}")
    if big_thresh < small_max or big_thresh >= budget:
        raise ValueError(f"require small_max <= big_thresh < budget, got {small_max}, {big_thresh}, {budget}")

    reset_ctr = 1 if cadence >= 2 else 0

    old_block = """  %gate = scalar.cmpi ne, %flag_value, %c0_i32 : i32
  %flag_at = index.constant 1676 : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  view.store %flag_value, %flag_view[0] : i32, view<1xi32>
  // cursors, flags and the total packed length
  %total = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0_i32 : i32) -> (i32) {
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %length_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length = scf.select %gate, %c1_i32, %length_host : i32
    scf.if %gate {
      view.store %length, %device[%s, 0] : i32, view<[%spans]x9xi32>
    }
    %flags = view.load %requests[%sr, 6] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
    %not_gate = scalar.cmpi eq, %flag_value, %c0_i32 : i32
    %mark_spec = scalar.andi %speculative, %not_gate : i1
    %credit_bits = scalar.shli %credit, %c8_i32 : i32
    %spec_bits = scalar.ori %credit_bits, %c4_i32 : i32
    %marked = scalar.ori %flags, %spec_bits : i32
    %device_flags = scf.select %mark_spec, %marked, %flags : i32
    view.store %cursor, %device[%s, 4] : i32, view<[%spans]x9xi32>
    view.store %device_flags, %device[%s, 6] : i32, view<[%spans]x9xi32>
    %next = scalar.addi %cursor, %length : i32
    scf.yield %next : i32
  }
  // move inputs to their cursors (last span first), zero placeholders
  scf.for %r = [%c0 to %spans step %c1] {
    %last = index.sub %spans, %c1 : index
    %s = index.sub %last, %r : index
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %sd = index.assume %s [lt(%s, %spans)] : index
    %length32_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length32 = scf.select %gate, %c1_i32, %length32_host : i32
    %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
    %known32 = view.load %requests[%sr, 5] : view<[%plan_extent]x9xi32> -> i32
    %cursor32 = view.load %device[%sd, 4] : view<[%spans]x9xi32> -> i32
    %moves = scalar.cmpi ne, %begin32, %cursor32 : i32
    %length_b = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
    %known_b = scalar.assume %known32 [range(%known32, 0, 8192)] : i32
    %begin_b = scalar.assume %begin32 [range(%begin32, 0, 8191)] : i32
    %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
    %length = index.cast %length_b : i32 to index
    %known = index.cast %known_b : i32 to index
    %begin = index.cast %begin_b : i32 to index
    %cursor = index.cast %cursor_b : i32 to index
    scf.if %moves {
      scf.for %q = [%c0 to %known step %c1] {
        %kl = index.sub %known, %c1 : index
        %j = index.sub %kl, %q : index
        %src_t = index.add %begin, %j : index
        %dst_t = index.add %cursor, %j : index
        %src0 = index.mul %src_t, %stride : index
        %dst0 = index.mul %dst_t, %stride : index
        scf.for %w = [%c0 to %stride step %c1] {
          %si = index.add %src0, %w : index
          %di = index.add %dst0, %w : index
          %sb = index.assume %si [lt(%si, %words)] : index
          %db = index.assume %di [lt(%di, %words)] : index
          %v = view.load %flat[%sb] : view<[%words]xi32> -> i32
          view.store %v, %flat[%db] : i32, view<[%words]xi32>
        }
      }
    }
    // placeholders after the known inputs (speculative spans): zero ids; media records keep text
    // positions continuing the anchor's (t = h = w)
    scf.for %j = [%known to %length step %c1] {
      %dst_t = index.add %cursor, %j : index
      %dst0 = index.mul %dst_t, %stride : index
      scf.for %w = [%c0 to %stride step %c1] {
        %di = index.add %dst0, %w : index
        %db = index.assume %di [lt(%di, %words)] : index
        view.store %c0_i32, %flat[%db] : i32, view<[%words]xi32>
      }
      scf.if %mixed {
        %anchor0 = index.mul %cursor, %stride : index
        %anchor4 = index.add %anchor0, %four : index
        %ab = index.assume %anchor4 [lt(%anchor4, %words)] : index
        %t0 = view.load %flat[%ab] : view<[%words]xi32> -> i32
        %j32 = index.cast %j : index to i32
        %tj = scalar.addi %t0, %j32 : i32
        %five = index.constant 5 : index
        %six = index.constant 6 : index
        %p4 = index.add %dst0, %four : index
        %p5 = index.add %dst0, %five : index
        %p6 = index.add %dst0, %six : index
        %b4 = index.assume %p4 [lt(%p4, %words)] : index
        %b5 = index.assume %p5 [lt(%p5, %words)] : index
        %b6 = index.assume %p6 [lt(%p6, %words)] : index
        view.store %tj, %flat[%b4] : i32, view<[%words]xi32>
        view.store %tj, %flat[%b5] : i32, view<[%words]xi32>
        view.store %tj, %flat[%b6] : i32, view<[%words]xi32>
      }
    }
  }"""

    new_block = f"""  %gate = scalar.cmpi ne, %flag_value, %c0_i32 : i32
  {DP_CTRL_ENC_BEGIN}
  // Dynamic decode-priority interleaving (cadence={cadence}, budget={budget}, small_max={small_max}, big_thresh={big_thresh})
  %dp_small_max = scalar.constant {small_max} : i32
  %dp_big_thresh = scalar.constant {big_thresh} : i32
  %dp_c32 = scalar.constant 32 : i32
  %dp_c128 = scalar.constant 128 : i32
  %dp_cadence = scalar.constant {cadence} : i32
  %dp_reset_ctr = scalar.constant {reset_ctr} : i32
  %dp_budget = scalar.constant {budget} : i32
  %dp_n1 = scalar.constant -2 : i32
  %dp_true = scalar.constant true : i1
  %dp_kind_at = index.constant 1672 : offset
  %dp_kind_view = buffer.view %control_state[%dp_kind_at] : buffer -> view<1xi32>
  %dp_draft_kind = view.load %dp_kind_view[0] : view<1xi32> -> i32
  %dp_no_draft = scalar.cmpi eq, %dp_draft_kind, %c0_i32 : i32
  %dp_no_next = scalar.cmpi eq, %next_count, %c0_i32 : i32
  %dp_n_dec, %dp_n_big, %dp_small_tok, %dp_has_spec = scf.for %s = [%c0 to %spans step %c1](%nd = %c0_i32 : i32, %nb = %c0_i32 : i32, %st = %c0_i32 : i32, %hs = %c0_i32 : i32) -> (i32, i32, i32, i32) {{
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %lh = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %cr = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %is_ord = scalar.cmpi eq, %cr, %c0_i32 : i32
    %is_sp = scalar.cmpi ne, %cr, %c0_i32 : i32
    %le_dec = scalar.cmpi sle, %lh, %dp_small_max : i32
    %gt_big = scalar.cmpi sgt, %lh, %dp_big_thresh : i32
    %is_dec = scalar.andi %is_ord, %le_dec : i1
    %is_big = scalar.andi %is_ord, %gt_big : i1
    %add_d = scalar.extui %is_dec : i1 to i32
    %add_b = scalar.extui %is_big : i1 to i32
    %add_s = scalar.extui %is_sp : i1 to i32
    %add_t = scf.select %is_big, %c0_i32, %lh : i32
    %nd1 = scalar.addi %nd, %add_d : i32
    %nb1 = scalar.addi %nb, %add_b : i32
    %st1 = scalar.addi %st, %add_t : i32
    %hs1 = scalar.addi %hs, %add_s : i32
    scf.yield %nd1, %nb1, %st1, %hs1 : i32, i32, i32, i32
  }}
  %dp_no_media = scalar.xori %mixed, %dp_true : i1
  %dp_no_gate = scalar.xori %gate, %dp_true : i1
  %dp_no_spec = scalar.cmpi eq, %dp_has_spec, %c0_i32 : i32
  %dp_has_dec = scalar.cmpi sgt, %dp_n_dec, %c0_i32 : i32
  %dp_has_big = scalar.cmpi sgt, %dp_n_big, %c0_i32 : i32
  %dp_small_fits = scalar.cmpi sle, %dp_small_tok, %dp_c128 : i32
  %dp_spans_fit = scalar.cmpi sle, %first_count, %dp_c32 : i32
  %dp_rem_pre = scalar.subi %dp_budget, %dp_small_tok : i32
  %dp_budget_ok = scalar.cmpi sge, %dp_rem_pre, %dp_n_big : i32
  %dp_c0 = scalar.andi %dp_no_media, %dp_no_gate : i1
  %dp_c1 = scalar.andi %dp_c0, %dp_no_draft : i1
  %dp_c2 = scalar.andi %dp_c1, %dp_no_next : i1
  %dp_c3 = scalar.andi %dp_c2, %dp_no_spec : i1
  %dp_c4 = scalar.andi %dp_c3, %dp_has_dec : i1
  %dp_c5 = scalar.andi %dp_c4, %dp_has_big : i1
  %dp_c6 = scalar.andi %dp_c5, %dp_small_fits : i1
  %dp_c7 = scalar.andi %dp_c6, %dp_spans_fit : i1
  %dp_active = scalar.andi %dp_c7, %dp_budget_ok : i1
  %dp_state_at = index.constant 2020 : offset
  %dp_state = buffer.view %control_state[%dp_state_at] : buffer -> view<4xi32>
  %dp_cap = scf.if %dp_active -> (i32) {{
    %ctr = view.load %dp_state[0] : view<4xi32> -> i32
    %ctr1 = scalar.addi %ctr, %c1_i32 : i32
    %wrap = scalar.cmpi sge, %ctr1, %dp_cadence : i32
    %ctr_next = scf.select %wrap, %c0_i32, %ctr1 : i32
    view.store %ctr_next, %dp_state[0] : i32, view<4xi32>
    %is_prefill_turn = scalar.cmpi eq, %ctr, %c0_i32 : i32
    %nb_nz = scalar.maxsi %dp_n_big, %c1_i32 : i32
    %slice_cap = scalar.divui %dp_rem_pre, %nb_nz : i32
    %chosen_cap = scf.select %is_prefill_turn, %slice_cap, %c0_i32 : i32
    view.store %c1_i32, %dp_state[1] : i32, view<4xi32>
    view.store %chosen_cap, %dp_state[2] : i32, view<4xi32>
    scf.yield %chosen_cap : i32
  }} else {{
    view.store %dp_reset_ctr, %dp_state[0] : i32, view<4xi32>
    view.store %c0_i32, %dp_state[1] : i32, view<4xi32>
    view.store %c0_i32, %dp_state[2] : i32, view<4xi32>
    scf.yield %c0_i32 : i32
  }}
  // cursors, flags and the total packed length
  %total = scf.for %s = [%c0 to %spans step %c1](%cursor = %c0_i32 : i32) -> (i32) {{
    %sr = index.assume %s [lt(%s, %plan_extent)] : index
    %length_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%sr, 7] : view<[%plan_extent]x9xi32> -> i32
    %is_ord = scalar.cmpi eq, %credit, %c0_i32 : i32
    %is_big = scalar.cmpi sgt, %length_host, %dp_big_thresh : i32
    %is_trunc = scalar.cmpi slt, %dp_cap, %length_host : i32
    %cap0 = scalar.andi %dp_active, %is_ord : i1
    %cap1 = scalar.andi %cap0, %is_big : i1
    %capped = scalar.andi %cap1, %is_trunc : i1
    %len_dp = scf.select %capped, %dp_cap, %length_host : i32
    %length = scf.select %gate, %c1_i32, %len_dp : i32
    %write_len = scalar.ori %gate, %capped : i1
    scf.if %write_len {{
      view.store %length, %device[%s, 0] : i32, view<[%spans]x9xi32>
    }}
    %flags_raw = view.load %requests[%sr, 6] : view<[%plan_extent]x9xi32> -> i32
    %flags_nosel = scalar.andi %flags_raw, %dp_n1 : i32
    %flags = scf.select %capped, %flags_nosel, %flags_raw : i32
    %speculative = scalar.cmpi ne, %credit, %c0_i32 : i32
    %not_gate = scalar.cmpi eq, %flag_value, %c0_i32 : i32
    %mark_spec = scalar.andi %speculative, %not_gate : i1
    %credit_bits = scalar.shli %credit, %c8_i32 : i32
    %spec_bits = scalar.ori %credit_bits, %c4_i32 : i32
    %marked = scalar.ori %flags, %spec_bits : i32
    %device_flags = scf.select %mark_spec, %marked, %flags : i32
    view.store %cursor, %device[%s, 4] : i32, view<[%spans]x9xi32>
    view.store %device_flags, %device[%s, 6] : i32, view<[%spans]x9xi32>
    %next = scalar.addi %cursor, %length : i32
    scf.yield %next : i32
  }}
  // Select minimal compiled stage from actual post-capping (%total, %first_count)
  %dp_flag = scf.if %dp_active -> (i32) {{
    %c2_i = scalar.constant 2 : i32
    %c4_i = scalar.constant 4 : i32
    %c8_i = scalar.constant 8 : i32
    %c16_i = scalar.constant 16 : i32
    %c32_i = scalar.constant 32 : i32
    %c64_i = scalar.constant 64 : i32
    %c128_i = scalar.constant 128 : i32
    %c256_i = scalar.constant 256 : i32
    %sb_gt2 = scalar.cmpi sgt, %first_count, %c2_i : i32
    %sb_gt4 = scalar.cmpi sgt, %first_count, %c4_i : i32
    %sb_gt8 = scalar.cmpi sgt, %first_count, %c8_i : i32
    %sb_gt16 = scalar.cmpi sgt, %first_count, %c16_i : i32
    %sb0 = scalar.extui %sb_gt2 : i1 to i32
    %sb1 = scalar.extui %sb_gt4 : i1 to i32
    %sb2 = scalar.extui %sb_gt8 : i1 to i32
    %sb3 = scalar.extui %sb_gt16 : i1 to i32
    %sb_a = scalar.addi %sb0, %sb1 : i32
    %sb_b = scalar.addi %sb2, %sb3 : i32
    %sb_i32 = scalar.addi %sb_a, %sb_b : i32
    %sb_bnd = scalar.assume %sb_i32 [range(%sb_i32, 0, 4)] : i32
    %sb_idx = index.cast %sb_bnd : i32 to index
    %tb_gt2 = scalar.cmpi sgt, %total, %c2_i : i32
    %tb_gt4 = scalar.cmpi sgt, %total, %c4_i : i32
    %tb_gt8 = scalar.cmpi sgt, %total, %c8_i : i32
    %tb_gt16 = scalar.cmpi sgt, %total, %c16_i : i32
    %tb_gt32 = scalar.cmpi sgt, %total, %c32_i : i32
    %tb_gt64 = scalar.cmpi sgt, %total, %c64_i : i32
    %tb_gt128 = scalar.cmpi sgt, %total, %c128_i : i32
    %tb_gt256 = scalar.cmpi sgt, %total, %c256_i : i32
    %tb0 = scalar.extui %tb_gt2 : i1 to i32
    %tb1 = scalar.extui %tb_gt4 : i1 to i32
    %tb2 = scalar.extui %tb_gt8 : i1 to i32
    %tb3 = scalar.extui %tb_gt16 : i1 to i32
    %tb4 = scalar.extui %tb_gt32 : i1 to i32
    %tb5 = scalar.extui %tb_gt64 : i1 to i32
    %tb6 = scalar.extui %tb_gt128 : i1 to i32
    %tb7 = scalar.extui %tb_gt256 : i1 to i32
    %tb_01 = scalar.addi %tb0, %tb1 : i32
    %tb_23 = scalar.addi %tb2, %tb3 : i32
    %tb_45 = scalar.addi %tb4, %tb5 : i32
    %tb_67 = scalar.addi %tb6, %tb7 : i32
    %tb_03 = scalar.addi %tb_01, %tb_23 : i32
    %tb_47 = scalar.addi %tb_45, %tb_67 : i32
    %tb_i32 = scalar.addi %tb_03, %tb_47 : i32
    %tb_bnd = scalar.assume %tb_i32 [range(%tb_i32, 0, 8)] : i32
    %tb_idx = index.cast %tb_bnd : i32 to index
    %dp_tbl_at = index.constant 1840 : offset
    %dp_tbl = buffer.view %control_state[%dp_tbl_at] : buffer -> view<9x5xi32>
    %stage_tag = view.load %dp_tbl[%tb_idx, %sb_idx] : view<9x5xi32> -> i32
    scf.yield %stage_tag : i32
  }} else {{
    scf.yield %flag_value : i32
  }}
  %flag_at = index.constant 1676 : offset
  %flag_view = buffer.view %control_state[%flag_at] : buffer -> view<1xi32>
  view.store %dp_flag, %flag_view[0] : i32, view<1xi32>
  // Move inputs: forward when shrinking (dp_active, stride=1), original backward otherwise
  scf.if %dp_active {{
    scf.for %s = [%c0 to %spans step %c1] {{
      %sr = index.assume %s [lt(%s, %plan_extent)] : index
      %sd = index.assume %s [lt(%s, %spans)] : index
      %length32 = view.load %device[%sd, 0] : view<[%spans]x9xi32> -> i32
      %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
      %known32_raw = view.load %requests[%sr, 5] : view<[%plan_extent]x9xi32> -> i32
      %known32 = scalar.minsi %known32_raw, %length32 : i32
      %cursor32 = view.load %device[%sd, 4] : view<[%spans]x9xi32> -> i32
      %moves = scalar.cmpi ne, %begin32, %cursor32 : i32
      %known_b = scalar.assume %known32 [range(%known32, 0, 8192)] : i32
      %begin_b = scalar.assume %begin32 [range(%begin32, 0, 8191)] : i32
      %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
      %known = index.cast %known_b : i32 to index
      %begin = index.cast %begin_b : i32 to index
      %cursor = index.cast %cursor_b : i32 to index
      scf.if %moves {{
        scf.for %j = [%c0 to %known step %c1] {{
          %src_t = index.add %begin, %j : index
          %dst_t = index.add %cursor, %j : index
          %sb = index.assume %src_t [lt(%src_t, %words)] : index
          %db = index.assume %dst_t [lt(%dst_t, %words)] : index
          %v = view.load %flat[%sb] : view<[%words]xi32> -> i32
          view.store %v, %flat[%db] : i32, view<[%words]xi32>
        }}
      }}
    }}
  }} else {{
    scf.for %r = [%c0 to %spans step %c1] {{
      %last = index.sub %spans, %c1 : index
      %s = index.sub %last, %r : index
      %sr = index.assume %s [lt(%s, %plan_extent)] : index
      %sd = index.assume %s [lt(%s, %spans)] : index
      %length32_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
      %length32 = scf.select %gate, %c1_i32, %length32_host : i32
      %begin32 = view.load %requests[%sr, 4] : view<[%plan_extent]x9xi32> -> i32
      %known32 = view.load %requests[%sr, 5] : view<[%plan_extent]x9xi32> -> i32
      %cursor32 = view.load %device[%sd, 4] : view<[%spans]x9xi32> -> i32
      %moves = scalar.cmpi ne, %begin32, %cursor32 : i32
      %length_b = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
      %known_b = scalar.assume %known32 [range(%known32, 0, 8192)] : i32
      %begin_b = scalar.assume %begin32 [range(%begin32, 0, 8191)] : i32
      %cursor_b = scalar.assume %cursor32 [range(%cursor32, 0, 8191)] : i32
      %length = index.cast %length_b : i32 to index
      %known = index.cast %known_b : i32 to index
      %begin = index.cast %begin_b : i32 to index
      %cursor = index.cast %cursor_b : i32 to index
      scf.if %moves {{
        scf.for %q = [%c0 to %known step %c1] {{
          %kl = index.sub %known, %c1 : index
          %j = index.sub %kl, %q : index
          %src_t = index.add %begin, %j : index
          %dst_t = index.add %cursor, %j : index
          %src0 = index.mul %src_t, %stride : index
          %dst0 = index.mul %dst_t, %stride : index
          scf.for %w = [%c0 to %stride step %c1] {{
            %si = index.add %src0, %w : index
            %di = index.add %dst0, %w : index
            %sb = index.assume %si [lt(%si, %words)] : index
            %db = index.assume %di [lt(%di, %words)] : index
            %v = view.load %flat[%sb] : view<[%words]xi32> -> i32
            view.store %v, %flat[%db] : i32, view<[%words]xi32>
          }}
        }}
      }}
      scf.for %j = [%known to %length step %c1] {{
        %dst_t = index.add %cursor, %j : index
        %dst0 = index.mul %dst_t, %stride : index
        scf.for %w = [%c0 to %stride step %c1] {{
          %di = index.add %dst0, %w : index
          %db = index.assume %di [lt(%di, %words)] : index
          view.store %c0_i32, %flat[%db] : i32, view<[%words]xi32>
        }}
        scf.if %mixed {{
          %anchor0 = index.mul %cursor, %stride : index
          %anchor4 = index.add %anchor0, %four : index
          %ab = index.assume %anchor4 [lt(%anchor4, %words)] : index
          %t0 = view.load %flat[%ab] : view<[%words]xi32> -> i32
          %j32 = index.cast %j : index to i32
          %tj = scalar.addi %t0, %j32 : i32
          %five = index.constant 5 : index
          %six = index.constant 6 : index
          %p4 = index.add %dst0, %four : index
          %p5 = index.add %dst0, %five : index
          %p6 = index.add %dst0, %six : index
          %b4 = index.assume %p4 [lt(%p4, %words)] : index
          %b5 = index.assume %p5 [lt(%p5, %words)] : index
          %b6 = index.assume %p6 [lt(%p6, %words)] : index
          view.store %tj, %flat[%b4] : i32, view<[%words]xi32>
          view.store %tj, %flat[%b5] : i32, view<[%words]xi32>
          view.store %tj, %flat[%b6] : i32, view<[%words]xi32>
        }}
      }}
    }}
  }}
  {DP_CTRL_ENC_END}"""

    if src.count(old_block) != 1:
        raise ValueError("expected exact @encode_epoch cursor/move block in control.loom")
    out = src.replace(old_block, new_block, 1)

    old_pub = """  scf.for %i = [%c0 to %length step %c1] {
    %request = index.assume %i [lt(%i, %plan_extent)] : index
    %consumed = view.load %requests[%request, 0] : view<[%plan_extent]x9xi32> -> i32
    %source_position = view.load %requests[%request, 2] : view<[%plan_extent]x9xi32> -> i32
    %source_advance = scalar.muli %consumed, %c2_i32 : i32
    %next_source_position = scalar.addi %source_position, %source_advance : i32
    %flags = view.load %requests[%request, 6] : view<[%plan_extent]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    %selected_count = scalar.extui %selected : i1 to i32"""

    new_pub = f"""  {DP_CTRL_PUB_BEGIN}
  %dp_state_at = index.constant 2020 : offset
  %dp_state = buffer.view %control_state[%dp_state_at] : buffer -> view<4xi32>
  %dp_act_i32 = view.load %dp_state[1] : view<4xi32> -> i32
  %dp_cap = view.load %dp_state[2] : view<4xi32> -> i32
  %dp_active = scalar.cmpi ne, %dp_act_i32, %c0_i32 : i32
  %dp_big_thresh = scalar.constant {big_thresh} : i32
  %dp_true = scalar.constant true : i1
  scf.for %i = [%c0 to %length step %c1] {{
    %request = index.assume %i [lt(%i, %plan_extent)] : index
    %len_host = view.load %requests[%request, 0] : view<[%plan_extent]x9xi32> -> i32
    %credit = view.load %requests[%request, 7] : view<[%plan_extent]x9xi32> -> i32
    %is_ord = scalar.cmpi eq, %credit, %c0_i32 : i32
    %is_big = scalar.cmpi sgt, %len_host, %dp_big_thresh : i32
    %is_trunc = scalar.cmpi slt, %dp_cap, %len_host : i32
    %cap0 = scalar.andi %dp_active, %is_ord : i1
    %cap1 = scalar.andi %cap0, %is_big : i1
    %capped = scalar.andi %cap1, %is_trunc : i1
    %consumed = scf.select %capped, %dp_cap, %len_host : i32
    %source_position = view.load %requests[%request, 2] : view<[%plan_extent]x9xi32> -> i32
    %source_advance = scalar.muli %consumed, %c2_i32 : i32
    %next_source_position = scalar.addi %source_position, %source_advance : i32
    %flags = view.load %requests[%request, 6] : view<[%plan_extent]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %want_sel = scalar.cmpi ne, %bits, %c0_i32 : i32
    %not_cap = scalar.xori %capped, %dp_true : i1
    %selected = scalar.andi %want_sel, %not_cap : i1
    %selected_count = scalar.extui %selected : i1 to i32
  {DP_CTRL_PUB_END}"""

    if out.count(old_pub) != 1:
        raise ValueError("expected exact @plain_publish_epoch loop in control.loom")
    return out.replace(old_pub, new_pub, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prep_in", help="Input stock prepare.loom")
    ap.add_argument("prep_out", help="Output patched prepare.loom")
    ap.add_argument("ctrl_in", help="Input stock control.loom")
    ap.add_argument("ctrl_out", help="Output patched control.loom")
    ap.add_argument("--cadence", type=int, default=5, help="Mixed-epoch cadence (1 prefill slice per N epochs; default 5)")
    ap.add_argument("--budget", type=int, default=512, choices=[256, 512], help="Token budget for prefill slice epochs (default 512)")
    ap.add_argument("--small-max", type=int, default=64, help="Max span length treated as interactive/decode (default 64)")
    ap.add_argument("--big-thresh", type=int, default=128, help="Min span length treated as long prefill (default 128)")
    ap.add_argument("--check", action="store_true", help="Dry-run only; verify anchors without writing output files")
    args = ap.parse_args()

    prep_out = patch_prepare(open(args.prep_in).read())
    ctrl_out = patch_control(
        open(args.ctrl_in).read(),
        cadence=args.cadence,
        budget=args.budget,
        small_max=args.small_max,
        big_thresh=args.big_thresh,
    )
    if args.check:
        print(f"OK (dry-run): cadence={args.cadence} budget={args.budget} small_max={args.small_max} big_thresh={args.big_thresh}")
        return
    open(args.prep_out, "w").write(prep_out)
    open(args.ctrl_out, "w").write(ctrl_out)
    print(f"wrote {args.prep_out} and {args.ctrl_out} (cadence={args.cadence}, budget={args.budget})")


if __name__ == "__main__":
    main()
