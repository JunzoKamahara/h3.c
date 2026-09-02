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

## Ablation — which tensors cause the argmax flips (QINT-016)

`make quant-ablate` re-runs the eval with parts of the resident INT4 set forced
back to BF16 (`H3_QWEN_Q4_BF16_LAYERS=a-b`, `H3_QWEN_Q4_BF16_PROJ=kv,down,…`).
The eval now also buckets every argmax flip by the reference top1−top2 margin
and reports flips per prompt / per position.

| # | config | top-1 | top-5 | rel L2 | cos | KL | flips (mid/large) | decode s/tok |
|---|---|---|---|---|---|---|---|---|
| A | all W4 (incl. lm_head) | 0.894 | 0.852 | 0.228 | 0.971 | 0.101 | 9 / 0 | 0.16 |
| B | W4 + BF16 lm_head *(current opt-in default)* | 0.894 | 0.878 | 0.199 | 0.977 | 0.078 | 9 / 0 | 0.16 |
| C | W4 + BF16 layers 56–63 | 0.929 | 0.892 | 0.118 | 0.991 | 0.053 | 6 / 0 | — |
| D | W4 + BF16 layers 50–63 | 0.941 | 0.906 | 0.107 | 0.992 | 0.042 | 5 / 0 | 0.19 |
| E | W4 + BF16 K/V (all layers) | 0.929 | 0.885 | 0.170 | 0.984 | 0.067 | 6 / 0 | 0.17 |
| F | W4 + BF16 down_proj (all layers) | 0.929 | 0.913 | 0.155 | 0.986 | 0.046 | 6 / 0 | — |
| G | BF16 layers 50–63 + **AWQ-lite** on 0–49 | 0.918 | 0.918 | 0.099 | 0.993 | 0.036 | 6 / **1** | ~0.16 |
| H | W4 + BF16 K/V and down (all layers) | 0.906 | 0.925 | 0.130 | 0.991 | 0.042 | 8 / 0 | — |
| **I** | **BF16 layers 50–63 + BF16 K/V on 0–49** | **0.953** | 0.913 | 0.091 | 0.995 | **0.033** | **4 / 0** | **0.20** |

Findings:

- **Every flip in every config is mid-margin (ref top1−top2 in 0.1–1.0);
  zero large-margin (≥1.0), zero tiny (<0.1)**, and they cluster in the first
  ~10 positions. No confident prediction is being flipped — so a top-1 target
  of 0.99 is too strict as a *gate*; it is a diagnostic. KL / cosine /
  margin-bucketed flips are the real signal.
- **The chat tail (layers 50–63) carries a disproportionate share of the
  error** (D: top-1 +0.047, KL −0.036 vs B). These 14 layers never feed H3, so
  BF16 there has no H3 downside — only ~0.03 s/tok.
- **K/V quantization matters and is nearly free to undo** (E: top-1 +0.035 at
  +0.01 s/tok; K/V are 5120×1024, ~10 MB/layer).
- **AWQ-lite on 0–49 is counterproductive here** (G worse than D on top-1, and
  it introduced the only large-margin flip seen). The diagonal-proxy objective
  is steering some channel scales the wrong way — do not ship it; a real
  activation-in-loss AWQ or dropping AWQ is the call.
- **Config I is the recommended mixed policy**: `H3_QWEN_Q4=mixed` →
  BF16 layers 50–63 + BF16 K/V on 0–49 + W4 (RTN) elsewhere + BF16 lm_head.
  top-1 0.953, KL 0.033, 4 mid-margin flips, 0.20 s/tok (~5 tok/s, still 3.1×
  over BF16 tiled). Resident ~30 GB (vs 62 BF16, 16 pure-W4).

## `Mixed-W4/BF16` current status vs the default gate (QINT-014)

`H3_QWEN_Q4=mixed` (ablation config I) on the text set:

```
Text-logit quality:
  large-margin flips = 0        PASS   (gate: zero)
  KL                 = 0.033    PASS   (gate: <= 0.05)
  cosine             = 0.995    PASS   (gate: >= 0.99)
  Japanese task qual = ?        NOT YET (needs a task-level check)
VLM  : NOT YET (QINT-010)
Tool : NOT YET (QINT-011 -- tool-selection parity, valid-JSON rate)
H3   : layer-49 drift MEASURED (QINT-012, below) -> H3 path stays BF16, so
       no H3 regression to gate; QINT-013 N/A for this policy.
Perf : 5.0 tok/s  PASS (gate: >= 4.5)   resident ~30 GB  PASS (gate: <= 32)
```

Top-1 (0.953) is a **diagnostic, not a gate** — every flip in every ablation
config is a mid-margin close call in the first ~10 positions. The full
default gate is in `TASKS.md` (QINT-014).

## Not yet covered

- VLM (image + text) — needs the `image_url` front-end (P7-004) — QINT-010.
- Tool calling end-to-end (only a tool-style text prompt here) — QINT-011.
- A Japanese *task*-level check (short generations scored), not just logits.

## Layer-49 hidden drift (QINT-012)

`make l49-drift` compares the layer-49 residual stream from the chat KV decode
path (captured via `H3_QWEN_DUMP_L49`) against the BF16 canonical
(`qwen_session_forward_to_layer(50)`, the H3-path `h3_text_encoder.c`).

| decode path | rel L2 | cosine | per-channel RMS ratio (mix/bf16) |
|---|---|---|---|
| BF16 (kernel drift only) | 1.0e-2 | 0.99995 | mean 1.000, all channels in [0.994, 1.008] |
| **Mixed-W4/BF16** | **0.138** | **0.991** | mean 0.984, **min 0.71 / max 2.82, 185/5120 channels outside [0.9, 1.1]** |

Reading it:

- The **BF16 decode path** (GEMV + `h3_qk_headnorm_rope_bf16` + `h3_add_rms_norm_bf16`
  vs the text encoder's separate kernels) drifts only ~1 % / cos 0.99995 — the
  fused decode kernels are faithful. (`max_abs` up to ~128 is a bf16 ULP on
  Qwen's known outlier activation channels, magnitude ~10^4.)
- **Mixed-W4/BF16 drifts the layer-49 hidden by ~14 % / cos 0.991**, and
  **185 of 5120 channels have their RMS magnitude changed by >10 %** (some
  0.7×, some ~3×). W4 on layers 0–49 measurably distorts the internal
  representation — the error accumulates through 50 residual adds.
- **This is acceptable for chat** (only ~1 % logit drift, top-1 0.953) because
  layers 50–63 and lm_head are BF16 and absorb it.
- **The layer-49 *interface* is unaffected — what changes is whether the
  0..49 *compute* can be shared.** In `--quality` (BF16) mode Chat and H3
  produce the identical layer-49 state and a combined request can run 0..49
  once (QEXP-002). In `--mixed` / `--fast` the Chat 0..49 is quantised, so H3
  runs its own BF16 0..49 (`h3_text_encoder.c`, already separate; `H3_QWEN_Q4`
  never touches it). `qwen_execution_policy` + `h3_conditioning_accepts()`
  make a quantised state un-passable to the DiT.
- QINT-013 (H3 generation regression) is **not a `Mixed-W4/BF16` default
  gate** — no quantised conditioning reaches the DiT. A one-shot same-seed
  BF16-vs-Mixed conditioning → H3 comparison is worth doing anyway (QEXP-001,
  non-blocking): "~14 % drift almost certainly degrades" is still inference,
  and a robust DiT would re-open full 0..49 sharing.

## QEXP-001 — layer-49 drift → H3 DiT sensitivity (non-blocking)

`make qexp-001` (`tests/test_qwen_l49_h3_sensitivity.c`): captures the layer-49
conditioning for one prompt two ways — BF16 canonical
(`forward_to_layer(50)`) and Mixed-W4/BF16 chat decode (`H3_QWEN_DUMP_L49`) —
then loads a T2VA DiT with each, denoises the **same seeded noise** (16 steps,
tiny latent geometry), and compares the output latents.

| stage | rel L2 (mixed vs bf16) | cosine |
|---|---|---|
| layer-49 conditioning (this prompt) | 0.118 | 0.882 |
| **DiT video latent (16 steps)** | **0.092** | **0.996** |
| **DiT audio latent (16 steps)** | **0.097** | **0.995** |

In *latent* space the DiT looks tolerant (cos 0.88 → 0.996). **This read was
misleading** — see QEXP-001b: the VAE decode re-amplifies the difference.
Latent cosine is a poor proxy for perceptual similarity here.

## QEXP-001b — perceptual sensitivity (VAE-decoded)

`make qexp-001b` (`tests/test_qwen_l49_h3_perceptual.c`): a real 256×256,
25-frame (39 aligned), 12-serving-step generation for each conditioning —
DiT `denoise_euler` + video VAE + audio VAE — same seed, then SSIM / PSNR on
the decoded pixels and correlation / SNR on the waveform.

| output | Mixed-W4 cond vs BF16 cond |
|---|---|
| video | **SSIM 0.731, PSNR 17.4 dB, mean\|Δpix\| 0.063** |
| audio | **corr 0.513, SNR −0.28 dB** |
| control (BF16 cond twice, `make qexp-001b-control`) | SSIM 1.0000, corr 1.0000 — the DiT+VAE pipeline is bit-deterministic |

**The ~14 % layer-49 conditioning drift decodes to a clearly different video
(SSIM 0.73) and a largely different audio sample (corr 0.51).** The control
proves this is real conditioning sensitivity, not run-to-run noise.

Conclusion: **feeding a Mixed-W4 conditioning to H3 would fail a same-seed
regression.** The H3 conditioning path must stay BF16 canonical — no unified
quantized layers-0..49 forward. `h3_conditioning_accepts()` /
`qwen_execution_policy` are measured-necessary, not just cautious. (Caveat:
one prompt, one small resolution; a larger generation is unlikely to be *more*
similar.)

## QEXP-002 — `--quality` shared layers-0..49 prefix (non-blocking)

`make qexp-002` (`tests/test_qexp002_shared_prefix.c`): in all-BF16 mode, a
combined Chat + H3 request needs the same layers-0..49 forward for both the
Chat prefill and the H3 conditioning. This runs it **once** and splits.

- **Chat tail from the shared layer-49 state == `qwen_engine_forward_full`,
  bit-for-bit** (next-token logits `memcmp`-equal, same argmax).
- **Shared layer-49 state == `qwen_session_get_h3_conditioning`, bit-for-bit**
  (`memcmp`-equal), so the H3 branch is unchanged.
- Wall time (6-token prompt, warm cache): naive combined (0..49 twice) ~13 s;
  shared (0..49 once) ~5 s. The saving is **one full prompt-length layers-0..49
  forward** — ~3.4 s here, and it grows with prompt length. (The `forward_full`
  path carries extra per-call overhead, so the headline % is soft; the robust
  statement is "the redundant 0..49 forward is eliminated".) The H3 DiT + VAE
  (~46 s) is unchanged either way.

So `--quality` mode can safely fuse the 0..49 prefix for a combined request
with **zero numerical difference** on either branch. (This is the opposite of
`--mixed` — QEXP-001b — where the branches must stay separate.) The
`h3_serve` / `/v1/responses` path does not implement this fusion yet.
