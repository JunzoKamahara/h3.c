# Chat decode speedup — investigation

Target hardware: **Apple M4 Max, 128 GB Unified Memory, Apple GPU Family 9,
Metal 4** (primary). M5 Max / Apple GPU Family 10 secondary.

## 1. Where the time goes

Measured, resident weights (`make bench-chat`, `H3_QWEN_RESIDENT=1`):

| phase | cost |
|---|---|
| prefill | ~25 tok/s (weight-load bound, ~independent of prompt length) |
| **decode** | **~0.63 s/token (~1.5 tok/s)** steady state |
| decode, streamed weights (default) | ~14 s/token — pure SSD/UMA weight I/O |

Per-shape decode microbenchmark (`make bench-matmul`, M4 Max):

```
shape            K       N | rows=1 bf16 GEMV  | rows=128 GEMM  bf16 vs int8
                           |        us    GB/s |  bf16 us  int8 us     x
q_proj        5120    8192 |    1004      83.5 |    1116     1302   0.86
k_proj        5120    1024 |     314      33.3 |     323      856   0.38
v_proj        5120    1024 |     308      34.0 |     315      848   0.37
o_proj        8192    5120 |     919      91.3 |    1116     1271   0.88
gate_proj     5120   25600 |    2522     103.9 |    2611     2844   0.92
up_proj       5120   25600 |    2544     103.0 |    2616     2847   0.92
down_proj    25600    5120 |    2563     102.3 |    3091     3556   0.87
lm_head       5120  151936 |   13677     113.8 |   27249    15327   1.78
```

Reading this:

- **Decode is ~95 % weight movement.** 64 layers × (q,k,v,o,gate,up,down) +
  lm_head ≈ **63 GB of BF16 weights read per token**. The per-shape rows=1
  loop projects ~665 ms/token of linears; measured decode is ~630 ms because a
  real layer overlaps its ops inside one command buffer. Everything else
  (RMSNorm, RoPE, GQA, softmax, the per-layer host K/V round-trip, ~128
  command submits) is together < 10 %.
- **The GEMV kernel leaves ~4× on the table.** The best shapes sustain
  ~102–114 GB/s; M4 Max UMA peak is ~400–546 GB/s. `h3_gpu_linear_bf16` routes
  wide matmuls through MPSGraph, which is tuned for GEMM, not batch-1 GEMV.
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

A faster *BF16* GEMV kernel (no numeric change) may be used everywhere,
guarded by the existing `phase0-parity` / `h3_real_prompt_test` hash gates.

## 3. Plan (M4 Max first)

| # | change | expected | risk / notes |
|---|---|---|---|
| 0 | **microbench harness** (`bench_qwen_matmul.c`) | — | done; re-run per change |
| 1 | **bandwidth-saturating BF16 GEMV kernel** for the decode projections: one threadgroup per output tile, coalesced 128-bit weight loads, register accumulate, single pass; replaces the MPSGraph batch-1 path in `h3_gpu_linear_bf16` when `rows == 1` | ~110 → ~300+ GB/s ⇒ **~2.5–3× decode** (0.63 → ~0.22 s/tok) | pure bf16, parity-gated; biggest single win |
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
