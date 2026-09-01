# Quantization quality — baseline (QINT-008)

Harness: `tests/test_qwen_quant_eval.c` / `make quant-eval`. Teacher-forces a
fixed 6-prompt set (EN factual, EN reasoning, JA ×2, Python, tool-style) one
token at a time through a resident KV session — so every scored position runs
the `rows == 1` decode path — and compares next-token logits.

- **Reference**: the BF16 *decode* path (`h3_linear_gemv_bf16` + per-layer
  fusion), not BF16 canonical (`forward_full`). So the deltas below are the
  **marginal** cost of quantization on top of the decode kernels the runtime
  already ships. (The decode kernels' own drift vs canonical is ~1e-3–1e-2
  rel L2, measured in `phase2-parity`.)
- **Test**: `W4A16` — group-wise symmetric RTN, group 128, all 64 decoder
  projections (lm_head stays BF16). `H3_QWEN_Q4=1`.

## Results — RTN W4A16, M4 Max, 85 scored positions

| prompt | kind | top-1 | top-5 | logit rel L2 | logit cos | KL(ref‖test) nats |
|---|---|---|---|---|---|---|
| 0 | EN factual | 1.000 | 0.900 | 0.165 | 0.986 | 0.031 |
| 1 | EN reasoning | 0.889 | 0.889 | 0.121 | 0.994 | 0.064 |
| 2 | JA | 0.750 | 0.850 | 0.135 | 0.992 | 0.050 |
| 3 | JA | 0.889 | 0.689 | 0.307 | 0.958 | 0.302 |
| 4 | Python | 1.000 | 0.965 | 0.183 | 0.987 | 0.008 |
| 5 | tool-style | 0.842 | 0.884 | 0.296 | 0.947 | 0.091 |
| **ALL** | | **0.894** | **0.878** | **0.199** | **0.977** | **0.078** |

Reading it:

- **~11 % of next tokens flip argmax** vs the BF16 decode path. Consistent
  with the greedy divergence at ~step 4 in `q4-decode-check`.
- **Japanese is the weak spot** (prompt 3: KL 0.30, cos 0.958). English
  factual and Python hold up best (top-1 1.0).
- These are teacher-forced (no divergence cascade), so they are the honest
  per-token quantization error, not a compounding worst case.

## Target for quality-qualified (QINT-014)

A `W4A16-AWQ` path should reach roughly: **top-1 ≥ 0.99, top-5 ≥ 0.99,
KL ≤ 0.01, cos ≥ 0.999** on this set, and hold on the JA prompts specifically,
before `H3_QWEN_Q4` can default on. Plus the non-logit gates (VLM, tool
calling, layer-49 drift, H3 regression — QINT-010..013), which need their own
harnesses.

## Not yet covered

- VLM (image + text) — needs the `image_url` front-end (P7-004).
- Tool calling end-to-end (only a tool-style text prompt here).
- Layer-49 hidden drift — needs a readback hook in `qwen_kv_eval`.
- H3 generation regression — needs the media pipeline in the loop.
