# Chat decode speedup — investigation

Target hardware: **Apple M4 Max, 128 GB Unified Memory, Apple GPU Family 9,
Metal 4** (primary). M5 Max / Apple GPU Family 10 secondary.

## 1. Where the time goes

Measured (`make bench-chat`, resident weights = the default):

| phase | cost (pre step #1) | cost (with step #1 BF16 GEMV) |
|---|---|---|
| prefill | ~25 tok/s (weight-load bound) | unchanged (rows>1, tiled path) |
| **decode** | ~0.63 s/token (~1.5 tok/s) | **~0.31 s/token (~3.2 tok/s)** steady state |
| decode, streaming (`--stream`, opt-in) | ~14 s/token — pure UMA weight I/O | ~13 s/token |

Per-shape decode microbenchmark (`make bench-matmul`, M4 Max). `rows=1` is now
the dedicated `h3_linear_gemv_bf16` kernel (step #1); the pre-step-#1 GEMV
column was the MPSGraph batch-1 path:

```
shape            K       N | rows=1 GEMV pre#1 | rows=1 GEMV step#1 | rows=128 GEMM
                           |        us    GB/s |         us    GB/s |  bf16 us
q_proj        5120    8192 |    1004      83.5 |      852      98.4 |    1293
k_proj        5120    1024 |     314      33.3 |      196      53.5 |     333
v_proj        5120    1024 |     308      34.0 |      194      54.2 |     328
o_proj        8192    5120 |     919      91.3 |      508     165.0 |    1098
gate_proj     5120   25600 |    2522     103.9 |     1237     212.0 |    2613
up_proj       5120   25600 |    2544     103.0 |     1243     210.9 |    2611
down_proj    25600    5120 |    2563     102.3 |     1392     188.4 |    3085
lm_head       5120  151936 |   13677     113.8 |     6246     249.1 |   27306
```

Step #1 roughly doubles sustained GB/s on the wide shapes (gate/up/down/o) and
lm_head, and ~1.6× on q. Projected linear-only floor: ~665 → ~366 ms/token;
measured decode 0.63 → 0.31 s/token (**2.0×**). Small k/v_proj stay
latency-bound (low threadgroup count at N=1024) but their absolute cost is
tiny.

Reading this:

- **Decode is ~95 % weight movement.** 64 layers × (q,k,v,o,gate,up,down) +
  lm_head ≈ **63 GB of BF16 weights read per token**. The per-shape rows=1
  loop projects ~665 ms/token of linears; measured decode is ~630 ms because a
  real layer overlaps its ops inside one command buffer. Everything else
  (RMSNorm, RoPE, GQA, softmax, the per-layer host K/V round-trip, ~128
  command submits) is together < 10 %.
- **The GEMV kernel left ~4× on the table (step #1 recovers ~half).** The
  MPSGraph batch-1 path sustained ~102–114 GB/s; M4 Max UMA peak is
  ~400–546 GB/s. `h3_linear_gemv_bf16` (step #1) now sustains ~190–250 GB/s on
  the wide shapes — one threadgroup per 8 output rows, K-axis strided with a
  shared-memory input tile, SIMD-reduced. The remaining gap to peak is the
  next target (larger tiles / multi-threadgroup split-K, or INT4 in step #2).
- **Small projections are latency-bound.** k/v_proj move 10 MB but take ~310 µs
  (≈ 32 GB/s): ~290 µs of that is fixed per-dispatch overhead, not bandwidth.
- **int8 on M4 is not a compute win.** h3.c's int8 / TensorOps kernels are
  GEMM-only (`rows >= 128`) and, on this M4 Max, run **slower than bf16** for
  every projection except the very wide `lm_head` (1.78×). This matches the
  note already in `h3_gpu.m` ("measured slightly slower on an M4 Max"). There
  is **no int8 GEMV path** in the tree today.

So on M4 Max the lever is **bandwidth**, in two independent multipliers:
a GEMV kernel that actually saturates UMA, and fewer bytes per weight
(quantisation).

## 2. Hard constraint — layer-49 parity

`Qwen layer-49 hidden` parity is release-blocking (spec §43/§44): any change
to decoder layers 0..49 can silently degrade H3 video/audio generation.

Therefore **quantisation and fused kernels apply only to the Chat tail** —
decoder layers 50..63, the final RMSNorm and `lm_head` — never to layers
0..49. Layers 0..49 stay BF16 on the exact current path. `lm_head` is
Chat-only already, so it is the safe first quantisation target; H3 never runs
it.

A faster *BF16* GEMV kernel changes only the reduction order, so its output
differs from the tiled kernel at the bf16-truncation level (~1e-4 relative on
the logits, argmax preserved). It is still safe for layers 0..49 because H3
media conditioning always runs the prompt at `rows > 1` (the tiled path) —
only single-token Chat decode takes the `rows == 1` GEMV route. The
`phase0-parity` / `h3_real_prompt_test` hash gates (all `rows > 1`) stay
bit-exact; `phase2-parity` checks decode on argmax + a tight relative bound.

## 3. Plan (M4 Max first)

| # | change | expected | risk / notes |
|---|---|---|---|
| 0 | **microbench harness** (`bench_qwen_matmul.c`) | — | ✅ done; re-run per change |
| 1 | **bandwidth-saturating BF16 GEMV kernel** for the decode projections: `h3_linear_gemv_bf16`, one threadgroup per 8 output rows, shared-memory input tile, K-axis strided FMA, SIMD reduction; routed from `h3_gpu_linear_bf16` when `rows == 1` (`H3_DISABLE_GEMV=1` to fall back) | ~110 → ~190–250 GB/s ⇒ **2.0× decode** (0.63 → 0.31 s/tok, 3.2 tok/s) | ✅ done. Not bit-exact vs the tiled kernel: logits shift ~1e-4 rel, argmax preserved; only single-token decode hits it, so layer-49 parity gates are untouched. `phase2-parity` compares decode on argmax + rel_l2 < 3e-2. Left ~2× still on the table. |
| 2 | **INT4 weights + INT4 GEMV kernel** for the Chat tail (layers 50..63) and `lm_head`; group-wise scales (group 64 or 128), dequant in-register during the streaming load | halves/quarters the tail+head bytes ⇒ combined with #1 **~5–10×** (→ 0.06–0.1 s/tok, 10–17 tok/s) | tail-only (parity), INT4 not INT8 (win is bandwidth, not compute); needs an accuracy check vs BF16 logits |
| 3 | **fuse per-layer submits**: one command buffer per decoder layer covering prep + attention + finish (128 → ~66 submits/token), and **keep K/V on the GPU** (drop the host read-back / re-upload in `qwen_kv.c` via a small GPU copy-range) | ~30–50 ms/token (~7 %) | mechanical; do after #1/#2 so it is measurable |
| 4 | **INT4 layers 0..49** *iff* a separate BF16 path is retained for H3 conditioning (two weight sets, or on-the-fly requant) | up to ~2× more on decode | large; only if #1–#3 are not enough and H3 parity can be fully isolated |
| 5 | speculative decoding (draft model / n-gram) | 1.5–3× on top | independent; orthogonal to the kernel work |

Not worth pursuing on M4 Max: h3.c's existing int8 GEMM kernels for decode
(rows≥128, slower than bf16 here), TensorOps BF16 for decode (compute is not
the bottleneck).

## 4. M5 / Apple GPU Family 10

M5 adds a per-core neural accelerator that executes TensorOps (incl. Int4 /
Int2 / FP8 with block scaling) in hardware. There, step #2's INT4 GEMV should
be a `mpp::tensor_ops` cooperative-tensor kernel rather than a hand-rolled
one, and an INT4 `lm_head` + fused attention become clearly favourable.

Keep the kernels switchable so one runtime picks per device:

```
gpu/metal/
  legacy_matmul.metal        # current MPSGraph / direct path
  gemv_bf16.metal            # step #1
  gemv_int4.metal            # step #2 (hand kernel; M4)
  tensorops_int4.metal       # step #2 (mpp::tensor_ops; M5)
  tensorops_attention.metal  # fused decode attention (M5-first)
```

Selection mirrors the existing capability probes (`h3_gpu_is_m5`,
`h3_gpu_has_int8_mlp`, `gpu.metal4Capable`, `H3_FORCE_TENSOROPS`,
`H3_GPU_CLASS`).

## 5. Bench shapes to track

Every kernel change re-runs `make bench-matmul` and compares GB/s per shape
plus the projected ms/token. The shapes are the Qwen3-VL decode projections:
q `[1,5120]×[5120,8192]`, k/v `×[5120,1024]`, o `[1,8192]×[8192,5120]`,
gate/up `×[5120,25600]`, down `[1,25600]×[25600,5120]`, lm_head
`×[5120,151936]`. A change is only adopted if the microbench shows the win
**and** `phase0-parity` + `phase1-parity` + `h3_real_prompt_test` stay green
(and, for tail quantisation, a logits-error check vs the BF16 path).
