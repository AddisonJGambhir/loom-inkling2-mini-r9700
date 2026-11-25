# Reproducing the R9700 speed-ups for Inkling-2-Mini on AMD Loom

This guide shows how to get the same gains on your own machine, whether or not the upstream pull request
([AddisonJGambhir/loom-inkling2-mini-r9700#1](https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700/pull/1)) has been merged.
An agent-oriented version of the same procedure is in [`REPRODUCE.agent.md`](REPRODUCE.agent.md).

## What you get

Measured on one Radeon AI PRO R9700 (gfx1201, 32 GB, 210 W power cap, also driving a display), hrx-loom **0.2.0**, drafter off:

| | stock package | with these changes |
|--|--:|--:|
| Single-chat decode at 60K / 121K context | 52 / 37 tok/s | **74 / 62 tok/s** |
| Agent follow-up turn at ~110K context (tool-using agent) | 0% of the prompt reused, 38-58 s to first token | **99%+ reused, ~2 s** |
| Turn after a screenshot (vision tool result) | full re-read (30-100 s at 80K) | **99.9% reused, ~0.2 s** (optional vision offload) |
| Concurrent chats | 1 chat (448K) | **up to 4 chats x 262K**, with a memory guard |

Short-context decode, prefill speed and 4-chat throughput are unchanged (within ~1%).

The changes come in three independent parts. You can take any of them:

1. **Decode attention v2** (one kernel in `package/program/llm.loom`). Long-context decode reads the int8 KV cache at ~413 GB/s
   instead of ~128 GB/s. It is not bit-exact (float summation order); see the README for the equivalence tests.
2. **Proxy fixes** (`bin/proxy.py`). Loom 0.2.0 only reuses a conversation's cached state if the request looks text-only and the
   history is sent back exactly. Two common agent behaviours break that, and the proxy fixes both:
   - a tool with a parameter literally named `image_url` (for example a `vision_analyze` tool) makes Loom treat every request as an
     image request, so no turn is ever reused;
   - agents often re-serialize the tool-call arguments they send back.
   The proxy also adds an optional memory guard for several chats, optional system-prompt reuse, and optional screenshot handling.
3. **A multi-chat profile** (`INKLING_PROFILE=concurrent` in `bin/inkling-r9700`): 4 chats x 262K, drafter off, prefix cache off.

## Requirements

- AMD Radeon AI PRO R9700 (gfx1201, 32 GB). Other gfx12 cards should work but are untested. The kernel is not target-specific, so
  RDNA3 cards would also get it, also untested.
- Linux with the in-kernel `amdgpu` driver, Python >= 3.12, about 20 GB of disk.
- **hrx-loom 0.2.0** (what the package installs) or **0.3.0**. All of this was also validated on 0.3.0 (`pip install "hrx-loom[hf]==0.3.0"`
  in the package venv): every verify check passes, output is token-identical to 0.2.0 on our equivalence tests, and speed is the same
  within ~2-3%.

## Path A: the upstream PR is merged, or you use the fork branch

```bash
# if the PR is merged, clone upstream instead:
#   git clone https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700
git clone -b r9700-decode-attn-v2-concurrency https://github.com/NT7S/loom-inkling2-mini-r9700
cd loom-inkling2-mini-r9700
bin/inkling-r9700 install                         # venv + hrx-loom 0.2.0 + ~19 GB of weights (first time only)
INKLING_PROFILE=concurrent bin/inkling-r9700 start
```

If you already had the package installed, `install` refreshes the programs (it copies `package/program/*.loom` into the model
directory). You can also copy them yourself into `~/.local/share/inkling-r9700/model/program/` and restart.

- Without `INKLING_PROFILE=concurrent` you keep the original single-chat profile (448K, drafter on). The kernel still applies, but
  with the drafter on, decode mostly goes through a kernel that was not changed. See "Drafter" below.
- **Optional, screenshot handling:** create `proxy.env` in the repo root containing `LOOM_VISION_OFFLOAD=1`, then restart the
  proxy: `systemctl --user restart inkling-proxy` should restart only the proxy, so Loom keeps its weights and caches.
  (Untested with this launcher's transient units; `bin/inkling-r9700 restart` always works but also restarts Loom.)
- The endpoint stays `http://127.0.0.1:1234/v1`, model `local`.

## Path B: stock upstream, PR not merged

You need three files from the fork branch:
`tools/gen_attn_split_v2.py`, `bin/proxy.py` and (optional) `tools/verify_r9700.py`.

**1. Kernel.** Generate it from your own `llm.loom` rather than copying ours, so it matches your package version:

```bash
python3 -I gen_attn_split_v2.py package/program/llm.loom /tmp/llm.v2.loom --span
sha256sum package/program/llm.loom /tmp/llm.v2.loom
```

From upstream commit `72294fb` (stock `sha256` starting `4429a038eb9a30fb`), the result starts with `9fcfa2f4f2c6b3d3`, the exact
file in the PR. If your stock file differs, the generator still works as long as the kernel it replaces is unchanged; it stops
with an error otherwise. Install it with `cp /tmp/llm.v2.loom ~/.local/share/inkling-r9700/model/program/llm.loom`, then
`bin/inkling-r9700 restart`. The first start compiles the changed kernels (seconds to a minute).

**2. Proxy.** `bin/proxy.py` from the branch is a drop-in replacement (same command line). Copy it over `bin/proxy.py` and restart.
Four fixes are on by default; each can be turned off with an environment variable (see the README): the `image_url` parameter
alias, tool-argument restore, diagnostics, and a better token estimate for image requests.

**3. Several chats (optional).** Either apply the launcher change from the branch, or set the same things yourself.

In the model card, under `presets.chat.execution`:

```json
"rows": 4, "max_sessions": 4, "context_capacity": 262144, "max_tokens": 48000,
"draft_depth": 0, "prefix_cache_capacity": 0, "session_cache_capacity": 2,
"cache_target_bytes": 2147483648, "checkpoint_capacity": 2
```

For the proxy process, set these environment variables:
`LOOM_NCTX=262144 LOOM_MAX_SESSIONS=4 LOOM_TOKEN_BUDGET=570000 LOOM_MAX_TOKENS=48000 LOOM_CHECKPOINT_CAPACITY=2 LOOM_SYS_CHECKPOINT=1`.
They can also go in `proxy.env`.

## Check it worked

```bash
python3 -I tools/verify_r9700.py --proxy http://127.0.0.1:1234 kernel decode reuse            # ~3 minutes
python3 -I tools/verify_r9700.py --proxy http://127.0.0.1:1234 concurrency                     # 4 chats at once
python3 -I tools/verify_r9700.py --proxy http://127.0.0.1:1234 --image some-screenshot.png vision   # needs LOOM_VISION_OFFLOAD=1
```

Each check prints one JSON line with `PASS`/`FAIL` and the numbers. Our results:

| check | what it shows | ours |
|--|--|--|
| `kernel` | v2 kernels present; with `--stock FILE`, regenerated identically | PASS |
| `decode` | tok/s at ~47K context / tok/s at ~6K (hardware-independent ratio) | 0.873 (stock ~0.68; pass >= 0.75) |
| `reuse` | agent follow-up cached, with an `image_url` tool and re-serialized arguments | 99.9%, 0.11-0.17 s (stock proxy: FAIL) |
| `concurrency` | 4 x ~24K chats all answer correctly; guard on | PASS, budget 570000 |
| `vision` | follow-up after a screenshot turn | 99.9% cached, 0.3 s |

## Make your agent cache-friendly

Loom 0.2.0 continues a conversation from its in-memory state only if the next request repeats the history **exactly as generated**.
The proxy handles the two most common breakages. A few others are on the client side:

- Send assistant turns back with their `reasoning_content` (or `reasoning`), unmodified.
- Keep `reasoning_effort`, the tool list and `tool_choice` the same from turn to turn. They are rendered into the prompt.
- A reply cut off by `max_tokens` (`finish_reason: length`) makes the next turn a full re-read. Leave enough room for replies.
- Compacting or truncating the history forces one full re-read, as expected.
- Watch the proxy log: `prefix-diff` and `echo conv` lines say exactly what changed when a turn misses
  (fields and lengths only, never content).

## Memory budget

With the drafter off, each context token costs ~17.4 KB of KV. The weights use ~17.2 GiB, plus ~3.6 GiB of fixed overhead under load.
Pick `LOOM_TOKEN_BUDGET` so that `weights + overhead + budget x 17.4 KB` stays at least ~1.5 GB under your Loom pool
(`--memory_bytes`). 570,000 was validated on a ~31 GB pool: 2 x 260K + 45K at once peaked at 29.4 GB. If the card also drives a
display, or you set `INKLING_MEMORY` lower, lower the budget too. With the drafter on, use ~21.5 KB per token (we validated 400,000).

## Drafter

With the drafter on, decode verifies 4-token spans through the span kernel, which v2 does not change. Drafter-on decode measured
76.7 / 61.9 / 48.9 / 35.1 tok/s at 8K / 30K / 60K / 121K, against 88 / 81 / 74 / 62 with the drafter off and v2. The drafter
still helps short-context single-chat coding (+17-29%) but costs ~19% of the KV budget. The concurrent profile turns it off.

## Loom behaviours to know about (0.2.0; 0.3.0 differences noted)

- **0.2.0: overcommitting the KV pool fails the whole model.** Streams end as empty HTTP 200s until a restart.
  `bin/loom-watchdog.sh` restarts it; the budget guard prevents it. **Fixed in 0.3.0:** Loom queues a request that does not fit.
  The guard is still useful there (it keeps the queue in the proxy, with a timeout).
- **Prefix-cache entries are never replaced.** Once full, later conversations get no reuse from them (still true on 0.3.0).
  On 0.2.0 they also block memory reclaim and can fail the model; 0.3.0 no longer fails. Keep `prefix_cache_capacity` at 0 with several chats.
- **A tool parameter named `image_url` turns a request into an image request** (no reuse, occasional "trailing JSON input" 500).
  Still true on 0.3.0. The proxy alias handles it.
- **Image turns are always re-read in full.** Vision offload (`LOOM_VISION_OFFLOAD=1`) replaces images with a description from
  Loom's own vision, so the conversation stays cacheable. The model then works from the description, not the pixels.
- **`continuation_epochs: 2` needs the drafter.**

## Undo

`git checkout package/program/llm.loom bin/proxy.py bin/inkling-r9700`, re-run `bin/inkling-r9700 install` (or copy the programs),
then `bin/inkling-r9700 restart` without `INKLING_PROFILE`.

## Running it in Docker instead

The numbers above came from a Docker setup, not the systemd launcher. If you do the same:
- Keep the card overrides in an env file that the entrypoint reads at start, so changes need a container restart, not a recreate.
- Before sizing `--memory_bytes` from free VRAM, wait until VRAM is drained. A quick restart after heavy use otherwise sees a half-full card.
- Supervise the proxy separately inside the container, so a proxy change doesn't restart Loom.
  A Loom restart sends clients with a cloud fallback to that fallback.
- Run a watchdog that restarts the container when `/v1/status` shows `"registration":"failed"`.
