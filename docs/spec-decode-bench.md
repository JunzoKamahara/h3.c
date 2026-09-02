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
