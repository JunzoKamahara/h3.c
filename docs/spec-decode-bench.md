# QINT-015d-3 — speculative verifier benchmark

`make spec-bench` (`tests/bench_qwen_spec.c`). Mixed-W4/BF16 target
(`H3_QWEN_Q4=mixed`), resident weights, M4 Max 128 GB. Warm-up first, the
per-trial rewind back to the base frontier is **not** timed. 10 reps, median
reported. Draft quality is removed from the picture: this measures only "verify
M candidate token positions in one target weight sweep" vs "decode M tokens one
at a time".

## Numbers

### short context (175 tokens)

| rows | scalar-M ms | verify-M ms | verify / (M·scalar-1) | upper tok/s | ideal speedup | batch compression (scalar-M / verify-M) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 194.6 | — | 1.00 | 5.14 | — | — |
| 2 | 389.2 | 455.3 | 1.17 | 4.39 | 0.85× | 0.85× |
| 3 | 584.6 | 528.2 | 0.90 | 5.68 | 1.11× | 1.11× |
| 4 | 780.9 | 606.5 | 0.78 | 6.60 | 1.28× | 1.29× |
| 5 | 976.5 | 693.1 | 0.71 | **7.21** | **1.40×** | 1.41× |

### long context (1505 tokens)

| rows | scalar-M ms | verify-M ms | verify / (M·scalar-1) | upper tok/s | ideal speedup | batch compression |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 224.5 | — | 1.00 | 4.45 | — | — |
| 2 | 448.0 | 485.5 | 1.08 | 4.12 | 0.92× | 0.92× |
| 3 | 672.2 | 560.6 | 0.83 | 5.35 | 1.20× | 1.20× |
| 4 | 897.6 | 641.1 | 0.71 | 6.24 | 1.40× | 1.40× |
| 5 | 1124.5 | 729.9 | 0.65 | **6.85** | **1.54×** | 1.54× |

`upper tok/s(M) = 1000·M / verify-M ms` — the throughput a *perfect* draft
would reach. `ideal speedup(M) = M·scalar-1 / verify-M` vs scalar decode.
End-to-end `qwen_spec_step()` under an oracle full-accept was within 0.5 ms of
the raw `verify-M` at every M and context — **the coordinator (draft propose +
top-2 scan + accept loop + pending / rewind state) adds essentially nothing**;
all the cost is the verify forward.

## Reading

- **The structure is right, the kernel is not fast enough yet.** A perfect
  draft at width 5 buys ~1.4–1.5× (5.1 → 7.2 tok/s short, 4.5 → 6.9 long), not
  the ~3× that "one weight sweep, 5 token positions" would allow if the verify
  forward were bandwidth-bound the way scalar decode is.
- `verify-M` has a large *fixed* cost: at M=2 it is already ~2× scalar-1
  (~390–450 ms), and each extra row then adds only ~75–90 ms. If the W4
  decode-batch projection kernel loaded each weight once and reused it across
  all M rows at full Unified-Memory bandwidth (which is the whole point of
  `h3_linear_q4_decode_batch`), `verify-5` should land near
  `scalar-1 + a little`, i.e. ~250–350 ms, not ~700 ms.
- Likely causes, for QINT-015e to profile and fix (do **not** optimise here):
  the batch kernel's `H3_GEMVB_KC = 1024` (vs the scalar GEMV's 4096) → 4× the
  K-chunk iterations and threadgroup barriers; `acc[8][5]` ≈ 40 vector
  registers per thread → low occupancy; the BF16 *tiled* lm_head for M rows vs
  the BF16 *GEMV* for 1 row; the `[m, vocab]` readback.
- Context length matters only a little: scalar-1 194 → 224 ms (+15 %) from 175
  to 1505 tokens, `verify-5` 693 → 730 ms (+5 %). Decode stays
  weight-bandwidth-dominated; the batch compression improves slightly with
  context (1.41× → 1.54×) because `scalar-M` grows with context faster than
  `verify-M`.

## Decision

Per `SPEC-DECODE-INSTRUCT.md` §66.9: `verify-5 ≈ 700 ms` is in the "needs
optimisation" band, not the "verifier is done, move to a learned draft" band.
So the next step is **QINT-015e** — decompose `verify-M` (W4 projections /
BF16 K/V / BF16 tail / attention / lm_head / readback / CPU top-2) and target
the W4 decode-batch kernel — before spending effort on DFlash-class drafts.

---

# QINT-015e-1 — which projection carries the verify-M fixed cost

`make spec-stage-bench` (`tests/bench_qwen_verify_stage.c`, no model weights,
real Qwen3-VL projection shapes, 20 reps median, M4 Max). Per projection and M:
`scalar-1` (one `rows==1` call), `scalar-M` (M sequential `rows==1` calls),
`batch-M` (one `rows==M` call). `b vs sM = batch-M / scalar-M` — below 1.0 means
the batched kernel is genuinely sharing weight bandwidth across the M rows.

| stage | K | N | kind | scalar-1 ms | scalar-5 ms | **batch-5 ms** | b vs sM (M=5) |
|---|---:|---:|---|---:|---:|---:|---:|
| q_proj | 5120 | 8192 | W4 | 0.443 | 0.997 | **1.110** | **1.11** |
| o_proj | 8192 | 5120 | W4 | 0.339 | 1.085 | **1.097** | **1.01** |
| gate / up | 5120 | 25600 | W4 | 1.208 | 3.14 | **3.030** | **0.97** |
| down | 25600 | 5120 | W4 | 0.959 | 3.02 | **3.155** | **1.05** |
| k / v | 5120 | 1024 | BF16 | 0.171 | 0.347 | 0.311 | 0.90 |
| tail q-shape | 5120 | 8192 | BF16 | 0.488 | 1.925 | 0.970 | **0.50** |
| tail o-shape | 8192 | 5120 | BF16 | 0.519 | 2.159 | 0.911 | **0.42** |
| tail MLP-shape | 5120 | 25600 | BF16 | 1.976 | 5.429 | 2.439 | **0.45** |
| lm_head | 5120 | 151936 | BF16 | 6.213 | 30.35 | 13.11 | **0.43** |

## Reading — the W4 decode-batch kernel is the whole problem

- **The BF16 batched path is already good.** The 16×16 tiled kernel at rows 5
  is 2–2.4× faster than 5 scalar GEMV calls (`b vs sM` 0.42–0.50). lm_head at
  width 5 is 13 ms, not 30 ms. `h3_gpu_linear_bf16` for rows 2..5 needs no
  work.
- **The W4 `h3_linear_q4_decode_batch` kernel does not share bandwidth at
  all.** `b vs sM` is 0.97–1.11 — running the 5-row batch costs the same as,
  or more than, 5 independent GEMVs. The "dequantise each nibble once, fetch
  each group scale once, accumulate into 5 registers" design is correct on
  paper but the sharing never turns into throughput.

## Where the 700 ms verify-5 goes (per the shapes above)

- W4 layers 0..49: `(q 1.11 + o 1.10 + gate 3.03 + up 3.03 + down 3.16)` ≈
  **11.4 ms/layer × 50 = ~570 ms** — of which gate/up/down alone are ~460 ms.
- BF16 K/V 0..49: `2 × 0.31 × 50` ≈ 31 ms.
- BF16 layers 50..63 (all projections): ~14 × ~7.7 ms ≈ 108 ms.
- lm_head batch-5: 13 ms.
- attention + norms + readback + CPU top-2: the remainder (~30–40 ms).

Total ≈ 720 ms, matching the measured verify-5.

If the W4 decode-batch kernel reached the BF16 tiled kernel's `b vs sM ≈ 0.45`,
the W4-layer cost would drop from ~570 ms to ~260 ms and **verify-5 would land
near ~410 ms → perfect-draft upper bound ~12.2 tok/s ≈ 2.4× scalar decode** —
the "worth a learned draft" band.

## QINT-015e plan

- **015e-0** `spec-chain-drift-check` (done): chained `verify_block` vs
  teacher-forced scalar, argmax agreement + margin-at-divergence. The
  numerical baseline to re-check after any kernel change.
- **015e-1** this table (done). Verdict: fix `h3_linear_q4_decode_batch`; the
  BF16 batched path is fine.
- **015e-2** rewrite the W4 decode-batch kernel so weight sharing across the M
  rows actually reduces wall time. Suspects from the code: `H3_GEMVB_KC = 1024`
  vs the scalar GEMV's 4096 → 4× the K-chunk iterations and threadgroup
  barriers; `acc[8][5]` + `p[5]` ≈ 45 vector registers per thread → occupancy
  collapse, so the sharing has no parallelism to hide the weight load behind.
  First experiments: M-specialised kernels (M=2 → `acc[8][2]`, …, M=5 →
  `acc[8][5]`) to isolate the register-pressure effect; fewer output rows per
  threadgroup; then a KC sweep (1024 → 1536).
- **015e-3** re-run `spec-stage-bench`, `spec-bench`, `spec-chain-drift-check`;
  compare against these baselines.

---

# QINT-015e-0 — batch-chain numerical drift baseline

`make spec-chain-drift-check` (`tests/test_qwen_spec.c` `run_chain_drift`).
Build the whole context out of CHAINED `verify_block` calls (block size B ∈
2..5) feeding the exact teacher-forced token sequence, and compare each row's
argmax to a teacher-forced scalar decode. Mixed target, M4 Max. This is the
numerical baseline; re-run it after any 015e kernel change.

| prompt | B | rows | argmax agreement | first divergence | margin at divergence |
|---|---:|---:|---:|---:|---|
| EN | 2..5 | 80–84 | **98.8 %** | pos 10 | one flip, gap in [0.05, 0.2) |
| JA | 2..5 | 95–96 | **99.0 %** | pos 60 | one flip, gap < 0.01 |
| code | 2..5 | 50–54 | **100 %** | — | — |

- The chained batched verifier's argmax tracks teacher-forced scalar decode at
  ~99 % of positions. Every disagreement is a small/medium-margin position
  (gap < 0.2); **the `>= 0.2` bucket is empty** — the batched chain never
  lands on a token the scalar path would confidently reject. This is the
  "batch-chain drift" behind the top-2 close-calls seen in `pending-parity`
  (e.g. EN position 11, gap 0.125), quantified.
- Block size B does not change the drift (same divergence position for all B) —
  it comes from the per-`verify_block` numeric path, not from block width.
- `spec-chain-drift-check` hard-fails if any divergence lands at a
  `>= 0.2`-margin position; that would mean a real error, not a close call.
