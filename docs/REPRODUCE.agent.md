# Agent runbook: reproduce the R9700 Loom speed-ups (Inkling-2-Mini)

Audience: a coding agent with a shell on the target machine. Human-oriented version: `docs/REPRODUCE.md`.
Each step has **run**, **expect**, and **on fail**. Do not skip verification. Stop and report to the user when a step fails twice.

## Objective
Install or upgrade `loom-inkling2-mini-r9700` with three changes and prove each with `tools/verify_r9700.py`:
K = decode attention v2 kernel, P = proxy fixes, C = concurrent profile (optional), V = vision offload (optional, user decision).

## Variables (confirm with the user or detect)
| name | default | how to detect |
|--|--|--|
| `REPO` | `./loom-inkling2-mini-r9700` | the checkout you work in |
| `HOME_DIR` | `~/.local/share/inkling-r9700` | `INKLING_HOME` if set |
| `PROXY` | `http://127.0.0.1:1234` | `INKLING_PORT` |
| `BACKEND` | `http://127.0.0.1:11437` | `INKLING_BACKEND_PORT` |
| `FORK_BRANCH` | `https://github.com/NT7S/loom-inkling2-mini-r9700` branch `r9700-decode-attn-v2-concurrency` | |
| `UPSTREAM_PR` | `https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700/pull/1` | `gh pr view 1 --repo AddisonJGambhir/loom-inkling2-mini-r9700 --json state` |

## Invariants (never violate)
1. Do not enable the prefix cache (`prefix_cache_capacity > 0`) together with several rows. Loom never replaces its endpoints (0.2.0 and 0.3.0),
   and on 0.2.0 they block memory reclaim (the model fails).
2. Do not run several long chats without the proxy budget guard (`LOOM_TOKEN_BUDGET`). On 0.2.0, KV overcommit fails the whole model
   (0.3.0 queues instead; keep the guard anyway).
3. Do not set `continuation_epochs: 2` with `draft_depth: 0`. Startup fails with INVALID_ARGUMENT.
4. Do not restart Loom while the user's clients are mid-request. Check `curl -s $PROXY/budget` (`"active": 0`) or ask.
   A Loom restart may push clients with a cloud fallback onto that fallback.
5. Never kill processes with `pkill -f` patterns. Kill by PID (e.g. the PID owning a port from `ss -ltnp`).
6. Vision offload (V) changes what the model sees (a text description instead of pixels): enable it only if the user agrees.
7. Log lines and verify output contain no conversation content. Do not add logging that does.

## Preflight
| # | run | expect | on fail |
|--|--|--|--|
| 0.1 | `python3 -c "import sys; assert sys.version_info >= (3,12)"` | exit 0 | stop: Python too old |
| 0.2 | `"$HOME_DIR/venv/bin/amdf-info" --format=json \| grep -c gfx1201` | >= 1 (or not installed yet) | not an R9700: warn the user (untested hardware) |
| 0.3 | `"$HOME_DIR/venv/bin/pip" show hrx-loom \| grep Version` | `0.2.0` or `0.3.0` (both validated) | another version: untested; tell the user, continue only if they agree |
| 0.4 | `gh pr view 1 --repo AddisonJGambhir/loom-inkling2-mini-r9700 --json state --jq .state` | `MERGED` -> Path A from upstream; otherwise Path A from the fork, or Path B | — |

## Path A (merged upstream, or the fork branch)
| # | run | expect | on fail |
|--|--|--|--|
| A1 | `git clone -b r9700-decode-attn-v2-concurrency https://github.com/NT7S/loom-inkling2-mini-r9700 "$REPO"` (upstream URL if merged) | checkout exists | network/auth: report |
| A2 | `cd "$REPO" && bin/inkling-r9700 install` | ends with `installed.` | read the error; disk (~20 GB) and glibc >= 2.39 are the usual causes |
| A3 | `cd "$REPO" && bin/inkling-r9700 stop; INKLING_PROFILE=concurrent bin/inkling-r9700 start` (or no profile for single chat) | `ready ... (concurrent: 262144 ctx, images)` | `journalctl --user -u inkling-backend -n 50`; on INVALID_ARGUMENT, check invariant 3 |
| A4 | go to Verify | | |

## Path B (stock upstream checkout, PR not merged)
Fetch seven files from the fork branch: `tools/gen_attn_split_v2.py`, `tools/gen_flash2_multi.py`, `tools/gen_fuse_nq.py`, `tools/gen_moe_f16_act.py`, `tools/gen_swa_v2.py`, `bin/proxy.py`, `tools/verify_r9700.py`.
| # | run | expect | on fail |
|--|--|--|--|
| B1 | `sha256sum package/program/llm.loom` | starts `4429a038eb9a30fb` (upstream 72294fb) or another stock hash | continue; B2 decides |
| B2 | `python3 -I tools/gen_attn_split_v2.py package/program/llm.loom /tmp/llm.v2.loom --span` | stderr: `added g_ink_attn_split_span_v2 and patched 16 span call sites` | `ValueError`/`AssertionError`: the stock kernel changed upstream; stop and report (do not hand-edit) |
| B2b | `python3 -I tools/gen_flash2_multi.py /tmp/llm.v2.loom /tmp/llm.v2.loom --nsub 4 --key-major --uniform --prefetch` (prefill kernel, recommended) | stderr: `rewrote g_ink_attention_flash2_ts2_rdna4: 4 query tiles per workgroup (384 lanes)` | `ValueError`: the stock prefill kernel changed; skip B2b and report |
| B2c | `python3 -I tools/gen_fuse_nq.py /tmp/llm.v2.loom /tmp/llm.v2.loom` (fused norm+q8, recommended) | stderr mentions `g_ink_rms_norm_plane_q8x` | `ValueError`: skip B2c and report |
| B2d | `python3 -I tools/gen_moe_f16_act.py /tmp/llm.v2.loom /tmp/llm.v2.loom` (direct f16 MoE act, recommended) | stderr mentions `removed 96 g_row_scale_f16_tiles` | `ValueError`: skip B2d and report |
| B2e | `python3 -I tools/gen_swa_v2.py /tmp/llm.v2.loom /tmp/llm.v2.loom` (SWA decode v2, recommended) | stderr mentions `g_ink_attn_split_swa_span_v2` | `ValueError`: skip B2e and report |
| B3 | `sha256sum /tmp/llm.v2.loom` | starts `16b3c7c20523ae04` (`7617edbcaa80e50f` without B2e, `cdb6a1894f000744` without B2d/B2e) when B1 was `4429a038...` | another B1 hash gives another output; continue, Verify decides |
| B4 | `cp package/program/llm.loom /tmp/llm.stock.loom && cp /tmp/llm.v2.loom package/program/llm.loom && cp package/program/llm.loom "$HOME_DIR/model/program/llm.loom"` | files copied | — |
| B5 | `cp bin/proxy.py /tmp/proxy.stock.py`, then put the fork's `bin/proxy.py` in place | — | — |
| B6 | (C, optional) apply the card settings below to `"$HOME_DIR/model/model.json"` (or a copy passed to `--model`) and the proxy env vars to the proxy process | — | — |
| B7 | `bin/inkling-r9700 restart` | `ready` | first start compiles changed kernels (<= 1 min); on `registration failed`, restore B4/B5 backups and report |

Card settings for C (`presets.chat.execution`):
`rows 4, max_sessions 4, context_capacity 262144, max_tokens 48000, draft_depth 0, prefix_cache_capacity 0, session_cache_capacity 0, cache_target_bytes 2147483648, checkpoint_capacity 2`.
Proxy env for C: `LOOM_NCTX=262144 LOOM_MAX_SESSIONS=4 LOOM_TOKEN_BUDGET=450000 LOOM_MAX_TOKENS=48000 LOOM_CHECKPOINT_CAPACITY=2 LOOM_SYS_CHECKPOINT=1 LOOM_DECODE_PRIORITY=1`.
Budget rule: `17.2 GiB + 3.6 GiB + budget x 17.4 KB <= pool - 1.5 GB` (drafter off). Recompute if the pool (`--memory_bytes` / `INKLING_MEMORY`) is not ~31 GB.

## Optional V: vision offload (only with user consent, invariant 6)
| # | run | expect | on fail |
|--|--|--|--|
| V1 | `echo LOOM_VISION_OFFLOAD=1 >> "$REPO/proxy.env"` | — | — |
| V2 | `systemctl --user restart inkling-proxy` (untested with transient units), or `bin/inkling-r9700 restart` (also restarts Loom) | proxy log line ends with `vision offload: on` | check that `proxy.env` sits next to `bin/` |

## Verify (all must PASS)
| # | run | expect (reference: R9700, 210 W cap, 0.2.0; 0.3.0 within ~3%) |
|--|--|--|
| T1 | `python3 -I tools/verify_r9700.py --proxy $PROXY --backend $BACKEND kernel` (add `--stock /tmp/llm.stock.loom` on Path B) | `PASS`; `prefill_4tile: true` and `prefill_prefetch: true` (unless B2b was skipped); `regenerated_identically: true` with `--stock` |
| T2 | `... decode` (~2 min, loads the GPU) | `PASS`, ratio >= 0.75 (reference 0.873; stock ~0.68) |
| T3 | `... reuse` | `PASS`, pct >= 90 (reference 99.9, ttft ~0.1-0.2 s). The stock proxy fails this |
| T4 | `... concurrency` (only with C) | `PASS`, `budget_guard: 570000` |
| T5 | `... --image SCREENSHOT.png vision` (only with V) | `PASS`, follow-up pct >= 90 (reference 99.9, ~0.3 s) |
On a FAIL: rerun it once (sampling). If it fails again: T1 -> the kernel is not installed in `$HOME_DIR/model/program`; T2 -> wrong
kernel or the drafter is on; T3 -> the proxy was not replaced, or the client path bypasses the proxy; T4 -> budget guard off, or
too little memory (lower `LOOM_TOKEN_BUDGET`); T5 -> `LOOM_VISION_OFFLOAD` not loaded.

## Diagnose a cache miss later
The proxy log (`journalctl --user -u inkling-proxy`) shows `prefix-diff` / `echo conv` lines naming the cause (reasoning dropped,
content edited, tool-call arguments re-serialized (auto-restored), settings changed, image parts, history shrank).
A POST line with `cached=0` and no such line usually means: the previous reply hit `max_tokens` (`finish: length`), or it is the
first request after a restart.

## Rollback
Path A: `git checkout main` (upstream) or a stock clone, then `bin/inkling-r9700 install && bin/inkling-r9700 restart`.
Path B: `cp /tmp/llm.stock.loom "$HOME_DIR/model/program/llm.loom"; cp /tmp/proxy.stock.py "$REPO/bin/proxy.py"; bin/inkling-r9700 restart`.

## Report to the user
List the paths applied (K/P/C/V), the T1-T5 JSON lines, and any deviations (hardware, Loom version, budget changes).
