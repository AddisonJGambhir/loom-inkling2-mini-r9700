# Measurements (one R9700, hrx-loom 0.2.0, pinned, --memory_bytes=32256mib, temp 1.0 / top_p 0.95 / top_k 20)

| test | f16 KV, 288K card | int8 KV, 448K card (this repo) |
|--|--|--|
| BetterBench decode, combined | 124.1 tok/s | 126.9 tok/s |
| BetterBench prefill 2K / 8K / 16K | 4,287 / 4,424 / 4,478 | 4,277 / 4,404 / 4,432 |
| 285K-token prompt, time to first token | 653 s | - |
| 440K-token prompt, time to first token | does not fit | 344-345 s |
| needle (code hidden in filler), 75K | - | 5/6 found (1 miss = over-long thinking at temp 1.0) |
| needle at 440K, 50% / 90% depth | - | found / found |
| image after 300K tokens of text | fails above ~253K | works (192 s, correct description) |
| 512K card, 510K prompt | - | loads at 32.4 GB VRAM, 781 s, answer did not finish: not used |

BetterBench 0.6.1, 3 passes + 1 warmup per category, `config/qwen-thinking-xtx.json` sampling.
