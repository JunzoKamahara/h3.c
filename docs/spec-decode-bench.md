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

## Reading (pre-015e-2 numbers above) — the W4 decode-batch kernel is the target

- **The BF16 batched path is already good.** The 16×16 tiled kernel at rows 5
  is 2–2.4× faster than 5 scalar GEMV calls (`b vs sM` 0.42–0.50). lm_head at
  width 5 is 13 ms, not 30 ms. `h3_gpu_linear_bf16` for rows 2..5 needs no
  work.
- **The W4 `h3_linear_q4_decode_batch` kernel shares weight bandwidth only
  weakly.** It is *not* true that it does no sharing: q_proj batch-5 (1.11 ms)
  is about half of 5 × scalar-1 (`b/s1M` ≈ 0.50). But `b vs sM` ≈ 0.97–1.11 —
  a 5-row batch costs as much as `scalar-M`. `scalar-M` is a *hot-cache*
  baseline (the same compressed W4 weight is streamed 5 times back-to-back and
  stays partly resident), so `b vs sM` flatters the scalar side. The honest
  physical framing: **the W4 batch processes 5 rows in ~2.5–3.3× the time of
  one row (`b/s1`), where the goal is ~1.5–2×.**

## Where the ~700 ms verify-5 goes (per the pre-015e-2 shapes)

- W4 layers 0..49: `(q 1.11 + o 1.10 + gate 3.03 + up 3.03 + down 3.16)` ≈
  **11.4 ms/layer × 50 = ~570 ms** — of which gate/up/down alone are ~460 ms.
- BF16 K/V 0..49: `2 × 0.31 × 50` ≈ 31 ms.
- BF16 layers 50..63 (all projections): ~14 × ~7.7 ms ≈ 108 ms.
- lm_head batch-5: 13 ms.
- attention + norms + readback + CPU top-2: the remainder (~30–40 ms).

Total ≈ 720 ms, matching the measured verify-5.

A realistic target for the W4 batch is `b/s1` ≈ 1.5–2.0 (BF16 batch-5/scalar-1
runs 1.2–2× depending on shape). That would take the W4-layer cost from
~570 ms to ~**350–400 ms** and verify-5 to ~**480–530 ms** → perfect-draft
upper bound ~9.5–10.5 tok/s ≈ 1.9–2.1× scalar decode. `b/s1` ≈ 1.23
(→ verify-5 ≈ 410 ms) is a strong stretch, not the expected outcome.

## QINT-015e plan

- **015e-0** `spec-chain-drift-check` (done): chained `verify_block` vs
  teacher-forced scalar, argmax agreement + margin-at-divergence. The
  numerical baseline to re-check after any kernel change.
- **015e-1** this table (done). Verdict: tune `h3_linear_q4_decode_batch`; the
  BF16 batched path is fine.
- **015e-2** (below): M-specialise the W4 kernel, then re-measure.
- **015e-3** re-run `spec-stage-bench`, `spec-bench`, `spec-chain-drift-check`;
  compare against these baselines. Then a speed table over acceptance
  60/70/80/90/100 % (verifier + draft cost) to decide whether a learned draft
  is worth it.

---

# QINT-015e-0 — batch-path numerical drift baseline

`make spec-chain-drift-check` (`tests/test_qwen_spec.c` `run_chain_drift`).
Build the whole context out of CHAINED `verify_block` calls (block size B ∈
2..5) feeding the exact teacher-forced token sequence, and compare each row's
argmax to a teacher-forced scalar decode. Mixed target, M4 Max. This is the
numerical baseline; re-run it after any 015e kernel change.

Index convention: verify-block row *r* (0-based) predicts the token at
sequence position *j + r + 1*, where *j* is the block's start; scalar logit
"position *p*" predicts token *p*. So `verify_block` row 10 and scalar
position 10 both predict token 11.

| prompt | B | rows | argmax agreement | first divergence | margin at divergence |
|---|---:|---:|---:|---:|---|
| EN | 2..5 | 80–84 | **98.8 %** | row 10 (token 11) | one flip, gap in [0.05, 0.2) |
| JA | 2..5 | 95–96 | **99.0 %** | row 60 (token 61) | one flip, gap < 0.01 |
| code | 2..5 | 50–54 | **100 %** | — | — |

- The chained batched verifier's argmax tracks teacher-forced scalar decode at
  ~99 % of positions. Every disagreement is a small/medium-margin position
  (gap < 0.2); **the `>= 0.2` bucket is empty** — the batched chain never
  lands on a token the scalar path would confidently reject. This is the
  batch-path numerical drift behind the top-2 close-calls seen in
  `pending-parity` (e.g. EN row 10 / token 11, gap 0.125), quantified.
- **Block size B does not change the divergence position** — so this is the
  per-`verify_block` numeric path differing from single-token decode, *not*
  something that grows with the number of chained blocks. (It is not proven
  to be "cumulative"; "batch-chain" is only the bench's name.)
- `spec-chain-drift-check` hard-fails if any divergence lands at a
  `>= 0.2`-margin position; that would mean a real error, not a close call.

---

# QINT-015e-2 — M-specialise the W4 decode-batch kernel

`h3_linear_q4_decode_batch` is now emitted once per M (2..5) by a macro
(`h3_linear_q4_decode_batch_m2` … `_m5`), so `acc[8][M]`, `p[M]` and
`x_tile[M][1024]` are all fixed-size: the compiler allocates exactly the
registers and threadgroup memory each M needs, instead of the M=5 worst case
(`acc[8][5]` ≈ 40 vector registers) for every call. `sg_partial` shrunk from
`[8][32]` to `[8][8]` (only 8 simdgroups exist at 256 threads). The
K-accumulation order is unchanged — `q4-check` still reports the batch rows
bit-for-bit equal to the scalar q4 GEMV (`rel = 0`).

`make spec-stage-bench`, W4 stages, M=5, before → after:

| stage | batch-5 before | batch-5 after | b/s1 after | b vs sM after |
|---|---:|---:|---:|---:|
| q_proj | 1.11 ms | **0.76 ms** | 1.66 | 0.76 |
| o_proj | 1.10 ms | **0.74 ms** | 2.22 | 0.68 |
| gate / up | 3.03 ms | **1.89 ms** | 1.59 | 0.63 |
| down | 3.16 ms | **2.04 ms** | 1.83 | 0.57 |

**−32 % to −38 % on every W4 projection at width 5.** gate/up landed at 1.9 ms
(the "register-pressure confirmed" prediction was ≤ 2.2 ms), so the fixed cost
was occupancy collapse from the M=5-sized `acc`/`p` arrays, not the KC size or
the barrier count. `maxabsdiff` (batch row vs scalar row, separate submits) is
≤ 0.002 — BF16 rounding noise, same as before.

A KC bump to 1536 was tried and made W4 M=5 *worse* (gate/up 1.89 → 2.44 ms):
the larger `x_tile` (30 KB) costs more occupancy than the fewer K-chunks buy.
KC stays 1024. "Fewer output rows per threadgroup" was not needed.

New verify-5 estimate from the shapes: W4 layers 0..49
`(0.76 + 0.74 + 1.89 + 1.89 + 2.04)` ≈ 7.3 ms/layer × 50 ≈ **370 ms** (was
~570 ms), so verify-5 should fall to ~**490 ms** → perfect-draft upper bound
~10.2 tok/s ≈ 2.0× scalar decode. Confirmed end-to-end numbers from
`spec-bench` / `spec-chain-drift-check` follow.

---

# QINT-015e-3 — end-to-end after 015e-2, and the acceptance break-even

`spec-verify-parity` still **345/345 EXACT** and `spec-chain-drift-check`
byte-identical to the 015e-0 baseline (EN 98.8 % / JA 99.0 % / code 100 %,
zero `>= 0.2` flips) — M-specialisation changed timing only, not numerics.

`make spec-bench`, before 015e-2 → after:

| M | verify-M short | verify-M long | v/(M·s1) short | upper tok/s short | end-to-end oracle short |
|---:|---:|---:|---:|---:|---:|
| 2 | 455 → **344** | 486 → **374** | 1.17 → **0.88** | 4.39 → 5.82 | 5.81 |
| 3 | 528 → **383** | 561 → **415** | 0.90 → **0.65** | 5.68 → 7.84 | 7.83 |
| 4 | 607 → **423** | 641 → **457** | 0.78 → **0.54** | 6.60 → 9.45 | 9.45 |
| 5 | **693 → 486** | **730 → 522** | 0.71 → **0.50** | 7.21 → **10.29** | **10.30** |

- **verify-5: 693 → 486 ms short (−30 %), 730 → 522 ms long.** Perfect-draft
  upper bound **7.21 → 10.29 tok/s short (2.01× scalar decode)**, 6.85 → 9.57
  long (2.15×). verify-2 is now *faster* than scalar-2 (344 vs 391 ms).
- Coordinator overhead is still ≈ 0 (oracle full-accept end-to-end within
  0.5 ms of raw verify-M).
- This is the "significant improvement, learned draft still a stretch"
  band (~486 ms, upper ~2×).

## Acceptance vs effective speed (pending-anchor, width 5, verify-5 = 486 ms)

The coordinator commits `1 + Σ_{i=1..4} a^i` tokens per cycle for a per-token
draft acceptance `a`; effective tok/s = that / `(verify-5 + draft) ms`.

| per-token accept `a` | committed / cycle | tok/s (draft 0 ms) | tok/s (draft 20 ms) | tok/s (draft 50 ms) |
|---:|---:|---:|---:|---:|
| 0.50 | 1.94 | 3.99 | 3.83 | 3.62 |
| 0.60 | 2.31 | 4.75 | 4.56 | 4.30 |
| **0.66** | **2.55** | **≈ 5.1 (= scalar)** | — | — |
| 0.70 | 2.77 | 5.71 | 5.48 | 5.17 |
| 0.80 | 3.36 | 6.92 | 6.64 | 6.27 |
| 0.90 | 4.10 | 8.43 | 8.09 | 7.64 |
| 1.00 | 5.00 | 10.29 | 9.88 | 9.33 |

Scalar decode is **5.11 tok/s**. So speculative decoding at width 5 only wins
above **~0.66–0.73 per-token acceptance** (depending on draft cost).

- The n-gram draft's measured per-token acceptance is **0.07–0.13**
  (`pending-parity`), giving ~1.1 committed/cycle ≈ **2.3 tok/s — well below
  scalar**. n-gram is a dead end here, as `SPEC-DECODE-INSTRUCT.md` §19/§66.8
  predicted.
- An EAGLE / DFlash-class head in-domain typically reaches ~0.75–0.85, i.e.
  ~6.5–7.5 tok/s here (~1.3–1.5× scalar). Worth doing (QINT-015h/i), and worth
  another verify-M pass first if time allows: at verify-5 ≈ 350 ms the same
  0.8 acceptance would give ~9.6 tok/s (~1.9×).

## Decision

- Keep the M-specialised W4 batch kernel. Do not chase KC / row-count tuning
  further now (KC 1536 was worse; the register-pressure win is banked).
- A learned draft (QINT-015h/i) is the next lever with the higher expected
  value — n-gram cannot clear the ~0.7 acceptance break-even.
- QINT-015f (adaptive scheduler) still matters: it must fall back to scalar
  whenever measured `committed/cycle · scalar-1 / (verify + draft)` < ~1.1.
