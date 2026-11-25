# Inkling-2-Mini on one Radeon AI PRO R9700 (AMD Loom)

A tuned [AMD Loom](https://github.com/ROCm/hrx-system) package for **Inkling-2-Mini** (Unsloth UD-Q4_K_XL) on a
**single 32 GB Radeon AI PRO R9700 (gfx1201)**. It serves an OpenAI-compatible endpoint, with image input.

| | stock Loom package | this repo |
|--|--|--|
| context | f16 KV, ~288K max on 32 GB | **448K** (int8 KV) |
| decode (BetterBench, temp 1.0) | 124 tok/s | **127 tok/s** |
| prefill 2K-16K | ~4,300-4,500 tok/s | ~4,300-4,400 tok/s |
| 440K-token prompt | does not fit | **fits, 345 s to first token, needle found at 50% and 90% depth** |
| images | text only | **yes** (verified after 300K tokens of text) |
| R9700 kernels | RDNA3 attention tiles | RDNA4 (gfx12) WMMA attention kernels |

All numbers were measured on one R9700 (Ubuntu 26.04, hrx-loom 0.2.0; re-validated on 0.3.0, same results within ~3%) with the model pinned and `--memory_bytes=32256mib`.

## Requirements
- AMD Radeon AI PRO R9700 (32 GB, gfx1201). It works best when the card is **not** driving your display
  (if it is, set `INKLING_MEMORY=30gib`).
- x86_64 Linux with glibc >= 2.39 (Ubuntu 24.04 or newer), the in-kernel `amdgpu` driver, and systemd user sessions.
- Python >= 3.12, about 20 GB of disk, and an internet connection for the first download.

## Install and run
```bash
git clone https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700
cd loom-inkling2-mini-r9700
bin/inkling-r9700 install     # venv + hrx-loom 0.2.0, then ~19 GB of weights via loom-fetch (pinned HF revisions)
bin/inkling-r9700 start       # first start compiles ~2,000 GPU kernels (a few minutes); later starts take seconds
```
The endpoint is `http://127.0.0.1:1234/v1`, model `local`. Other commands: `bin/inkling-r9700 status | logs | stop | restart`.

```bash
curl -s http://127.0.0.1:1234/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"temperature":1.0,"top_p":0.95,"top_k":20}'
```
- Clients that read llama.cpp's `/props` (for example Pi's llama-server provider) automatically see **458,752 context** and image support.
- The model thinks before answering. Use `reasoning_effort` (`none` ... `max`) to control how much.
- Use temperature 1.0, top_p 0.95, top_k 20. Avoid temperature 0.

If you have more than one gfx1201 card, set `INKLING_DEVICE=amdgpu://id/<id>`. Get the id from `~/.local/share/inkling-r9700/venv/bin/amdf-info --format=json`.
GPU ordinals (`amdgpu://N`) are not stable across Loom tools, so the launcher selects the card by id.

## What is changed vs. the upstream Loom package
`package/program/` holds the full programs. The changes, in order, with the scripts that made them in `tools/`:
1. **RDNA4 attention** (`tools/port_rdna4.py`): gfx12 twins of the four flash2 attention kernels, with per-target dispatch.
2. **Faster sampled top-k** (`tools/patch_sampler.py`): picks the top-k vocabulary chunks first instead of making k full passes over 200K logits.
3. **Image input**: vision programs and the mixed text/image embedding from the community Inkling vision port, plus the Unsloth BF16 mmproj.
4. **int8 KV cache** (`tools/kv8_patch.py`): q8_0-style int8 values with an f16 scale per 32 values. Each paged
   K/V position goes from 2048 to 1088 bytes. This applies to the 16 full-attention planes only; the 1024-token
   sliding-window ring and the DSpark drafter stay f16. Long prompts also prefill faster, because attention reads half the bytes.
5. **Drafter**: DSpark speculative decoding at depth 3.
6. **Decode attention v2** (`tools/gen_attn_split_v2.py SRC DST --span`): rewrites the single-token decode attention
   `g_ink_attn_split` and adds `g_ink_attn_split_span_v2`. The new kernel is dispatched for pure-decode epochs, where every
   span is one token (`t == spans`). Multi-token spans keep the stock span kernel. See below.

`bin/proxy.py` sits in front of Loom on :1234:
- serves the `local` alias and `/props`
- replaces images in older turns with a placeholder, so long conversations stay cacheable
- frees the held conversation and retries once when Loom refuses an image turn for memory
- optional **KV budget guard** (`LOOM_TOKEN_BUDGET`, off by default). It admits a chat only if all running chats' prompt + reply
  reservation fit the budget; otherwise the chat waits in a FIFO queue (`LOOM_QUEUE_TIMEOUT`, default 1800 s). `GET /budget` shows usage.
- **`prefix-diff` / `echo` diagnostics** (on; `LOOM_PREFIX_DIFF=0` turns them off). Loom continues a conversation from its idle row only
  if the client re-sends the history exactly as generated. The proxy logs an earlier message that changed, an assistant turn whose content,
  reasoning or tool calls differ from what the model generated, render-affecting settings that changed (reasoning_effort, tools, ...), a
  changed system prompt, and image parts. Fields, lengths and flags only, never content.
- **tool-argument restore** (on; `LOOM_ARG_RESTORE=0` turns it off): some agents re-serialize the tool-call arguments they send back (same
  JSON value, different spacing/escaping), which breaks reuse on every turn. The proxy restores the exact generated string when the echoed
  arguments parse to the same JSON. Values are never changed.
- **tool-parameter alias** (on; `LOOM_PARAM_ALIAS=0` turns it off): see the `image_url` item under Known limits.
- optional **vision offload** (`LOOM_VISION_OFFLOAD=1`): image parts are described by Loom's own vision in a side request (temperature 0,
  cached by image hash) and sent as text, so image turns no longer force Loom 0.2.0 to re-read the whole conversation. The model then
  works from the description instead of the pixels. Screenshot loop at 37K: follow-up turns 99.96% cached, ~0.2 s to first token.
- proxy-only settings (KEY=VALUE lines) can live in `proxy.env` in the repo root (or `$LOOM_PROXY_ENV`). They are read at proxy
  start, so `systemctl --user restart inkling-proxy` applies them without reloading Loom.
- optional **system-prompt checkpoints** (`LOOM_SYS_CHECKPOINT=1` with `LOOM_CHECKPOINT_CAPACITY>0` and a card `checkpoint_capacity`).
  New conversations reuse a shared long system prompt without Loom's prefix cache; see below.

### Decode attention v2
The stock decode attention gives each of the 3 query heads of a KV head its own wave. Each lane holds 4 of the 128 dims and walks
keys one at a time: per key and head, 4-byte K and V loads, two 2-byte scale loads, and a 32-lane reduction; the 3 waves load the
same K/V. On the R9700 that reads the int8 KV cache at ~128 GB/s, so long-context decode is bound by memory-instruction issue.
v2 uses one wave per KV head for all 3 query heads. Lane = 4 key slots x 8 dim chunks, with 16-byte K/V loads, an 8-lane
xor reduction (the relative-position bias for keys within 1024 is folded in), and a per-slot online softmax merged at the end.
Same arguments, grid, int8 dequant (f16 round trip kept) and partial layout, so `g_ink_attn_combine` is unchanged.

Single-chat decode, drafter off (one R9700 with a 210 W cap that also drives a display; tok/s, 600-token replies):

| context | stock | v2 |
|--|--:|--:|
| 8K | 84.1 | 88.3 |
| 30K | 67.3 | 81.4 |
| 60K | 52.4 | 73.5 |
| 121K | 36.6 | 61.9 |

KV read rises from ~128 to ~413 GB/s. With 4 chats decoding at ~60K each, combined throughput is ~+24%. Short-context decode
(-0.6%), 4K/32K prefill and the prefill stall are unchanged within noise. **It is not bit-exact** (different float summation
order). Greedy outputs matched the stock kernel on 2/2 long-context prompts (16K, 47K) and on 4/5 short prompts; the one
divergence is at a near-tie inside reasoning. Needle retrieval passed at 133K and 260K, and with 2 x 260K + 45K concurrent.
With the drafter on (the default profile), decode verifies 4-token spans, which still use the stock span kernel. Measured with
the drafter on: 76.7 / 61.9 / 48.9 / 35.1 tok/s at 8K / 30K / 60K / 121K. That is slower than drafter-off v2 beyond ~8K, so the
concurrent profile turns the drafter off. (A lane-grouped multi-token span kernel was tried and gave no measurable gain.)

### Prefill attention: 4 tiles per workgroup
`g_ink_attention_flash2_ts2_rdna4` (gfx12 prefill attention, made by `tools/gen_flash2_multi.py --nsub 4 --key-major` on top of v2).
A stock workgroup covers 2 query tiles of 16 tokens x the 3 query heads of one KV head (6 waves), so each 32-key K/V tile it
dequantizes into LDS serves only 32 queries. Now a workgroup covers 4 tiles (12 waves, 384 lanes): half the staging per query.
The staging lanes are also reordered key-major. In the stock order, the transposed V stores of one wave land on V^T rows 16 dims
apart, which all hit the same LDS bank. Per query nothing changes (same key tiles, same order, same online-softmax steps), so the
output is **bit-identical** (same tokens on the long, short and span equivalence sets). Compile report: 207 VGPRs (was 216), same
occupancy, no spills.

Time to first token, single cold prompt (same session, drafter off):

| prompt | v2 only | + 4-tile prefill |
|--|--:|--:|
| 8K | 2.05 s | 2.02 s |
| 32K | 8.53 s | 8.30 s |
| 64K | 20.0 s | 18.8 s |
| 130K | 52.4 s | 46.2 s |
| 255K | 149.7 s | **122.2 s** |

8 tiles per workgroup was slightly slower than 4 (255K: 130.6 s without key-major staging, against 129.7 s for 4).

### Running several chats: `INKLING_PROFILE=concurrent`
```bash
INKLING_PROFILE=concurrent bin/inkling-r9700 start
```
This writes `model-concurrent.json` (rows 4, `max_sessions` 4, 262,144 context per chat, drafter off, prefix cache off,
`checkpoint_capacity` 2) and starts the proxy with a 570,000-token budget (`INKLING_TOKEN_BUDGET`) and system-prompt checkpoints.
In Loom 0.2.0, `rows` is only an initial capacity hint. Rows share one elastic KV pool, ~17.4 KB per context token with the
drafter off (16 x 1088 B int8 KV) or ~21.5 KB with it. The budget was validated on a ~31 GB pool: 2 x 260K + 45K at once peaked
at 29.4 GB. If you lower `INKLING_MEMORY`, lower the budget too. Turn-to-turn reuse comes from idle rows (~97-99.9% of the prompt
cached on follow-up turns) as long as the client re-sends the history exactly as generated, including `reasoning_content`.
*The launcher profile itself was checked offline only.* The same settings were validated in a different (Docker) launcher.

### System-prompt checkpoints
Loom's named checkpoints only fork when a request's history extends the pinned history exactly, and `reasoning_effort` and
`tools` must match because the template renders them. So, the second time a new conversation arrives with the same system prompt,
tools, tool_choice and reasoning_effort, the proxy runs a seed `[system, "Reply with only the word: ok"]`. Loom only pins a
naturally finished reply. The proxy pins it, inserts the same 2-message seed after the system prompt of every new conversation with
that key (and on their later turns), and sends `X-Loom-Checkpoint` on each conversation's first request. At most
`LOOM_CHECKPOINT_CAPACITY` pins (LRU); pinned tokens come off the KV budget; a refused fork is retried plain.
18K-token system prompt + tools: a new conversation takes ~0.9-1.8 s instead of ~5.2 s (18,423 tokens cached).
**Trade-off: the model sees one extra short exchange after the system prompt.**

`bin/loom-watchdog.sh` restarts and re-pins the model if Loom ever marks it failed.

## Known limits
- **Turns that contain an image are not prefix-cached by Loom 0.2.0**: an image turn re-processes the whole conversation
  (~190 s at 300K tokens). Text turns after it are cached again, because the proxy swaps old images for a placeholder.
- 512K loads, but it fills the card (32.4 GB) and was unreliable, so the card is set to 448K.
- One conversation is served at a time (`rows: 1`) in the default profile; see `INKLING_PROFILE=concurrent`.
- **Loom 0.2.0 fails the whole model instead of rejecting a request** when live conversations overcommit the KV pool
  ("model residency is waiting for capacity"). Streams then return empty HTTP 200 until a restart; `loom-watchdog.sh` restarts it.
  Reproduced with 4 x ~101K concurrent prompts at 262K context per row and no admission control. Hence the proxy budget guard.
- **Prefix-cache endpoints are never evicted** once `prefix_cache_capacity` is full, so later conversations get only the first
  2048-token block. Each endpoint also holds recurrent state, and they block reclaim of idle histories: with 512 endpoints the
  model failed after ~30. The concurrent profile turns the prefix cache off.
- `continuation_epochs: 2` requires a nonzero `draft_depth` (startup fails with INVALID_ARGUMENT otherwise).
- **A tool parameter named `image_url` turns the whole request into a vision request** in loom-serve 0.2.0. There is then no
  turn-to-turn reuse, and "trailing JSON input" 500s happen sometimes. Agents with a vision tool hit this on every turn. `bin/proxy.py` aliases
  such parameters to `image_ref` and maps them back (`LOOM_PARAM_ALIAS=0` to disable). Measured with a 20-tool agent: 0% -> 99.9% reuse.
- int8 KV is near-lossless in our tests (needle retrieval at 75K and 440K, same BetterBench speed). It is still a
  quantization of the attention cache. If you suspect it, compare against the f16 package.

## Credits and license
Apache-2.0 (see `LICENSE` and `NOTICE`). This repo builds on:
- AMD Loom / hrx-system, Apache-2.0 WITH LLVM-exception
- the `loom-fun/unsloth-Inkling-2-Mini-loom` package
- the `Yanuk-K` Inkling Loom vision port
- `unsloth/Inkling-2-Mini-GGUF` (all Apache-2.0)

Weights are downloaded from their original repositories and are not redistributed here.
