# H3 Runtime — Quantization Terminology

Canonical vocabulary for weight quantization in this runtime. Code, docs, task
IDs and benchmark tables all use these names so nothing drifts.

## 1. Core principle

`BF16`, `INT8`, `INT4` and `AWQ` are distinct.

```
INT4 = a 4-bit-integer weight storage/compute format
AWQ  = an activation-aware method for choosing INT4 quantization parameters
```

They are **not** synonyms. `INT4` and `INT4-AWQ` are tracked as separate
states. `INT4` on its own never implies AWQ.

## 2. BF16

Weights held as Brain Floating Point 16-bit.

```
weight format: BF16
bits/weight:   16
bytes/weight:  2
```

The **canonical reference** format: correctness reference, parity tests, H3
conditioning reference, and the baseline every quantized path is compared
against. Written `BF16` (e.g. "BF16 decode", "BF16 canonical path").

## 3. INT8

Weight-only signed 8-bit integer quantization.

```
weight format: INT8
bits/weight:   8
bytes/weight:  1        (+ a separate scale)
w_real ≈ scale × q_int8
```

Bare `INT8` means **weight-only**; it does not imply INT8 activations.

## 4. INT4

Weight-only ~4-bit quantization.

```
bits/weight:          4
nominal bytes/weight: 0.5   (packed nibbles + scale [+ zero-point/metadata]
                             → effective slightly above 0.5)
```

Bare `INT4` means **ordinary 4-bit weight quantization with no
activation-aware optimization** — the format the current fast decode path
uses ("INT4 decode", "INT4 resident", "INT4 GEMV"). It is a
performance-oriented format and does not by itself imply high quality.

## 5. Group-wise quantization

INT4/INT8 keep one scale per small group of weights along the contraction
axis rather than one per tensor.

```
weights[0..G-1]    → scale0
weights[G..2G-1]   → scale1
...
```

Group size is part of the format and is always recorded: `INT4-G32`,
`INT4-G64`, `INT4-G128`.

## 6. Symmetric / asymmetric

```
symmetric:   zero_point = 0        w ≈ scale × q
asymmetric:  w ≈ scale × (q - zero_point)
```

Symmetric is simpler and decode-kernel friendly; asymmetric can improve
accuracy at the cost of metadata and dequant work. When unspecified, assume
**symmetric**.

## 7. AWQ

**Activation-aware Weight Quantization**: instead of looking only at the
weight distribution, run calibration inputs through the model and account for
the activation channels that matter when choosing the per-channel weight
quantization scaling. Goal: INT4's weight-traffic cut **without** the
important-channel error.

`INT4-AWQ` = 4-bit weight quantization **with AWQ calibration/scaling
applied**, and is kept distinct from bare `INT4`.

## 8. Calibration

Representative inputs fed through the model to fit AWQ parameters. Split by
use: **chat**, **VLM**, **H3 conditioning**. Minimum coverage: Japanese text,
English text, code, tool-call-style prompts, image+text, H3 generation
prompts.

## 9. Weight-only quantization / activation notation

The initial INT4/INT8 work is **weight-only**: activations stay `BF16`/`F32`.

```
W8A16       = 8-bit weights, 16-bit activations
W4A16       = 4-bit weights, 16-bit activations
W4A16(BF16) = ... with BF16 activations specifically
W4A16-AWQ   = AWQ-calibrated 4-bit weights, 16-bit activations
```

Group size may be appended: `W4A16-G64`, `W4A16-AWQ-G64`.

## 10. Preferred formal names

Benchmarks and specs use: `BF16`, `W8A16`, `W4A16`, `W4A16-AWQ` (with group
size where relevant). Shorthand `INT8`, `INT4`, `INT4-AWQ` is allowed, but
**`INT4 ≠ AWQ`** always holds.

## 11. Per-tensor quantization

Tensors need not share a format. Reference policy:

```
embed_tokens   BF16
q_proj         W4A16-AWQ
k_proj         W8A16 or BF16
v_proj         W8A16 or BF16
o_proj         W4A16-AWQ
gate_proj      W4A16-AWQ
up_proj        W4A16-AWQ
down_proj      W4A16-AWQ
final_norm     BF16
lm_head        W4A16-AWQ
```

Suggested design hook:

```c
typedef enum {
    QWEN_WEIGHT_BF16,
    QWEN_WEIGHT_INT8,       /* W8A16 */
    QWEN_WEIGHT_INT4,       /* W4A16, RTN */
    QWEN_WEIGHT_INT4_AWQ    /* W4A16-AWQ */
} qwen_weight_format;
```

carried per tensor (or per layer + tensor).

## 12. Mixed quantization

Different formats per tensor = **mixed quantization**, written e.g.
`Mixed W4/W8/BF16`:

```
MLP      W4A16-AWQ
Q/O      W4A16-AWQ
K/V      W8A16
Norm     BF16
LM Head  W4A16-AWQ
```

## 13. H3 conditioning and quantization

H3 uses the Qwen layer-49 hidden state as conditioning, so quantizing layers
0..49 must be judged on **more than chat accuracy**: also `layer49 hidden
drift` and `H3 generation quality`.

- **Canonical H3 path:** layers 0..49 = `BF16`.
- **Quantized H3 path** (future, layers 0..49 = `W4A16-AWQ`) must pass:
  layer49 cosine similarity, relative error, H3 prompt adherence, video visual
  quality, audio quality, same-seed regression.

In this runtime the two paths are already separate code: chat decode uses
`qwen_layers.c` / the resident set; H3 conditioning uses `h3_text_encoder.c`
with its own BF16 weights. Quantizing the resident set does not touch the H3
path.

## 14. Chat-only quantization

`layers 50..63`, `final_norm` and `lm_head` do not feed H3 conditioning and
can be quantized more freely. `lm_head` is never used by H3 generation, so it
is the safest first target — though it is also the single largest logit-error
contributor, so it is quantized last in practice / kept BF16 by default.

## 15. Decode path names

Benchmark rows use: `BF16 tiled`, `BF16 GEMV`, `BF16 fused`, `INT4 GEMV`,
`INT4 fused`, `W4A16-AWQ fused`.

## 16. Performance-qualified vs quality-qualified

A quantized path is tracked in two stages.

**Performance-qualified**

```
[ ] build passes
[ ] decode works
[ ] quantized-kernel checks pass  (q4-check)
[ ] benchmark reproduced
```

**Quality-qualified** (additionally)

```
[ ] language quality
[ ] Japanese quality
[ ] top-k / logit comparison vs BF16
[ ] VLM quality
[ ] tool calling
[ ] layer-49 drift
[ ] H3 generation regression
```

**Default-on requires both.** Hitting a tok/s number alone does not qualify a
path to be the default.

## 17. Current state (commit `e604557`)

The fast decode path is **`INT4 fused decode`** — a `W4A16` path
(group-wise symmetric RTN, group 128, `h3_linear_gemv_q4` + fused per-layer
kernels). **AWQ calibration is not implemented**, so it is **not** called
`INT4-AWQ` / `W4A16-AWQ`.

Status: **performance-qualified, not quality-qualified.** Opt-in via
`H3_QWEN_Q4=1`; default off.

Measured (M4 Max, 128 GB, resident, `bench-chat` steady state):

| path        | s/token | tok/s |
|-------------|---------|-------|
| BF16 tiled  | 0.63    | 1.6   |
| BF16 GEMV   | 0.31    | 3.2   |
| BF16 fused  | 0.29    | 3.5   |
| INT4 fused  | 0.16    | 6.1   |

Decode kernel-fusion milestone: **complete**.

### `Mixed-W4/BF16` (`H3_QWEN_Q4=mixed`) — canonical policy

```
Layers 0-49
  q_proj     W4A16 RTN
  k_proj     BF16
  v_proj     BF16
  o_proj     W4A16 RTN
  gate_proj  W4A16 RTN
  up_proj    W4A16 RTN
  down_proj  W4A16 RTN
Layers 50-63
  all projections BF16
Embedding    BF16
Final norm   BF16
LM head      BF16
```

Localised by `make quant-ablate` (QINT-016) as the min-cost config that
recovers most of the argmax accuracy. Text: top-1 0.953 / KL 0.033 / cos
0.995 / 0 large-margin flips (vs 0.894 / 0.078 for pure `W4A16`). Decode
~0.20 vs 0.16 s/token (~5 tok/s); resident ~30 GB.

AWQ-lite is **research-only** and is not part of this preset — on layers 0–49
it was worse than plain RTN.

Default gate (`TASKS.md` QINT-014):

| sub-gate | status |
|---|---|
| Text (no large-margin flips, KL ≤ 0.05, cos ≥ 0.99) | **PASS** (QINT-016) |
| Perf (≥ 4.5 tok/s, resident ≤ 32 GB) | **PASS** (~5 tok/s, ~30 GB) |
| Tool (selection parity ≥ 99 %, valid JSON ≥ 99.5 %) | **PASS** (QINT-011, 9/9) |
| VLM (no answer regression) | **PASS** (QINT-010) |
| H3 (layer-49 drift measured, no A/V regression) | **measured** — Mixed-W4 conditioning fails a same-seed H3 regression (QEXP-001b: video SSIM 0.73). H3 stays on canonical BF16; the quantized Chat path is walled off by `h3_conditioning_accepts()`. QEXP-003: cheap K_M levers recover audio, not video. |

`Mixed-W4/BF16` is quality-qualified **for the Chat/VLM/Tool decode path**.
Pending: flip it to default and expose `Mixed-W4/BF16` = default,
`Pure W4A16` = `--fast`, `BF16` = `--quality` in `h3_serve` (QINT-014).
