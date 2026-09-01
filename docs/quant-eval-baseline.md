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

## Results — AWQ-lite W4A16, same 85 positions

`make quant-calib` captures mean |x_j| per projection input over a **disjoint**
12-prompt calibration set (224 tokens); `make quant-eval-awq` then quantizes
with a per-input-channel scale `s[j] = (act_scale[j] / mean)^alpha`, alpha
grid-searched (0.1–0.9) against an activation-weighted, row-subsampled
reconstruction error. The decode GEMV folds `1/s` into the x load (no extra
dispatch, decode stays 0.16 s/tok).

| prompt | kind | top-1 | top-5 | logit rel L2 | logit cos | KL nats |
|---|---|---|---|---|---|---|
| 0 | EN factual | 0.800 | 0.860 | 0.172 | 0.980 | 0.044 |
| 1 | EN reasoning | 0.944 | 0.889 | 0.093 | 0.995 | 0.044 |
| 2 | JA | 0.750 | 0.850 | 0.186 | 0.989 | 0.073 |
| 3 | JA | 0.889 | 0.800 | 0.263 | 0.967 | 0.109 |
| 4 | Python | 1.000 | 0.906 | 0.141 | 0.988 | 0.008 |
| 5 | tool-style | 0.842 | 0.874 | 0.233 | 0.974 | 0.072 |
| **ALL** | | **0.882** | **0.871** | **0.174** | **0.983** | **0.054** |

RTN → AWQ-lite delta:

| | top-1 | top-5 | rel L2 | cos | KL |
|---|---|---|---|---|---|
| RTN | 0.894 | 0.878 | 0.199 | 0.977 | 0.078 |
| AWQ-lite | 0.882 | 0.871 | 0.174 | 0.983 | **0.054** |

**AWQ-lite cuts KL ~31 % and improves rel-L2 / cosine, but top-1 does not
improve (−0.012, within noise).** The distribution matches better overall;
the near-tie tokens that flip argmax are not helped. Likely causes: the
diagonal (per-channel) reconstruction proxy instead of a true
`||Q(W·s)·(x/s) − W·x||` over stored calibration activations; a small
calibration set (224 tokens); no weight-clipping search. Group 128 vs 32
barely moved RTN, so group size is not the lever here either.

Next levers (QINT-006 refinement): real activation-in-loss AWQ (store sample
activations, minimize the actual reconstruction), a clip search, or mixed
precision (`k/v` or `down` at BF16 / INT8).

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
