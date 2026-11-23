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

All numbers were measured on one R9700 (Ubuntu 26.04, hrx-loom 0.2.0) with the model pinned and `--memory_bytes=32256mib`.

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

`bin/proxy.py` sits in front of Loom on :1234:
- serves the `local` alias and `/props`
- replaces images in older turns with a placeholder, so long conversations stay cacheable
- frees the held conversation and retries once when Loom refuses an image turn for memory

`bin/loom-watchdog.sh` restarts and re-pins the model if Loom ever marks it failed.

## Known limits
- **Turns that contain an image are not prefix-cached by Loom 0.2.0**: an image turn re-processes the whole conversation
  (~190 s at 300K tokens). Text turns after it are cached again, because the proxy swaps old images for a placeholder.
- 512K loads, but it fills the card (32.4 GB) and was unreliable, so the card is set to 448K.
- One conversation is served at a time (`rows: 1`).
- int8 KV is near-lossless in our tests (needle retrieval at 75K and 440K, same BetterBench speed). It is still a
  quantization of the attention cache. If you suspect it, compare against the f16 package.

## Credits and license
Apache-2.0 (see `LICENSE` and `NOTICE`). This repo builds on:
- AMD Loom / hrx-system, Apache-2.0 WITH LLVM-exception
- the `loom-fun/unsloth-Inkling-2-Mini-loom` package
- the `Yanuk-K` Inkling Loom vision port
- `unsloth/Inkling-2-Mini-GGUF` (all Apache-2.0)

Weights are downloaded from their original repositories and are not redistributed here.
