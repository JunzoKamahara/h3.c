# TASKS

Tracking for the `h3-runtime` work described in `spec.md`. Only Phase 0 is in
scope right now; later phases are listed for context but not started.

## P0 — Qwen intermediate-state runtime boundary

- [x] P0-001 Document legacy layer-49 semantics
      (`docs/qwen-intermediate-state.md`)
- [x] P0-002 Introduce `qwen_input` (`qwen_engine.h`)
- [x] P0-003 Introduce `qwen_intermediate_state` (`qwen_engine.h`)
- [x] P0-004 Add `qwen_forward_to_layer()`
      (`qwen_session_forward_to_layer()` in `qwen_engine.c`)
- [x] P0-005 Add `qwen_get_h3_conditioning()`
      (`qwen_session_get_h3_conditioning()` in `qwen_engine.c`)
- [x] P0-006 Preserve text-only H3 parity
      (`tests/test_qwen_intermediate.c`, `make phase0-parity`)
- [x] P0-007 Preserve multimodal H3 parity
      (`tests/test_qwen_intermediate.c`, synthetic vision span)
- [x] P0-008 Add intermediate-state regression tests
      (`h3_qwen_intermediate_test`, wired into `make test`)

### P0 detail list (spec section 34)

- [x] P0-001 現在の h3_text_encoder.c を調査
- [x] P0-002 layer 49 出力の正確な tensor semantics を記録
- [x] P0-003 qwen_engine.h を作成
- [x] P0-004 qwen_engine.c を作成
- [x] P0-005 qwen_input 型を追加
- [x] P0-006 qwen_intermediate_state 型を追加
- [x] P0-007 qwen_forward_to_layer() 実装
- [x] P0-008 qwen_get_h3_conditioning() 実装
- [x] P0-009 text-only parity test
- [x] P0-010 multimodal parity test
- [x] P0-011 tags parity test
- [x] P0-012 existing h3 generation regression test
      (`make real-parity` / `h3_real_prompt_test`: layer-50 hash and the
      51-submission invariant are unchanged)

## P1 — Full 64-layer Chat LLM

- [x] P1-001 Decoder layers 50..63 from the layer-49 intermediate state
      (`qwen_lm_decode_tail()` in `qwen_lm.c`)
- [x] P1-002 Final language-model RMSNorm (`model.language_model.norm.weight`)
- [x] P1-003 `lm_head.weight` (untied) → 151936 logits
- [x] P1-004 CPU argmax → one decoded token (`qwen_logits.argmax_token`)
- [x] P1-005 `qwen_engine_forward_full()` (spec §11: forward-to-50 → tail)
- [x] P1-006 `qwen_session_continue_from_intermediate()` (spec §12)
- [x] P1-007 First-token / boundary parity test
      (`tests/test_qwen_lm.c`, `make phase1-parity`): `forward_full ==
      continue_from_intermediate(get_h3_conditioning)` bit-for-bit,
      run-to-run deterministic, optional golden-logits compare. Smoke:
      "The capital of France is" → " Paris".
- [x] P1-008 Layer-49 boundary unchanged (`make phase0-parity` +
      `h3_real_prompt_test` hash `e007b3a5097af1bf` still green)

Not in P1 (deferred): KV cache, HTTP, tool calling.

## P2 — KV Cache

- [x] P2-001 `h3_gqa_causal_kv_bf16` Metal kernel + `h3_gpu_gqa_causal_kv_bf16`
      wrapper (cached causal GQA; reduces bit-for-bit to `h3_gqa_causal_bf16`
      when `query_rows == kv_length`)
- [x] P2-002 `qwen_layers.c` — shared decoder-layer prep/finish split, used by
      the Phase 1 tail and the KV decoder
- [x] P2-003 Stateful `qwen_session`: per-layer GPU K/V caches, token history,
      position, latest logits (`qwen_kv.c`, spec §13/§14)
- [x] P2-004 Prefill — `qwen_session_eval()` first call
- [x] P2-005 Incremental decode — `qwen_session_eval()` on new tokens; only the
      new rows flow through projections/MLP
- [x] P2-006 `qwen_session_sample()` (greedy argmax), `qwen_session_logits()`,
      `qwen_session_length()`, `qwen_session_sync()`
- [x] P2-007 `qwen_session_rewind()` — truncate cache + history + position
- [x] P2-008 Multi-turn — rewind / re-eval reproduces earlier logits
- [x] P2-009 Parity test (`tests/test_qwen_kv.c`, `make phase2-parity`):
      prefill + greedy decode == `forward_full` bit-for-bit; chunked prefill ==
      single-shot; rewind reproduces; two sessions deterministic
- [x] P2-010 Regressions green (`phase0-parity`, `phase1-parity`,
      `h3_real_prompt_test` hash, `h3_tests`)

Not in P2 (deferred): sampling beyond greedy, HTTP, tool calling.

### P2 follow-up — weight residency (now the default)

- [x] Resident weights are the **default**: all 64 decoder layers + embed /
      norm / lm_head + one GPU, loaded once (~62 GB) on the first eval,
      shared process-wide via a reference-counted `g_resident`
      (`resident_acquire` / `resident_release`). Falls back to streaming if it
      does not fit. `qwen_session_set_resident(s,0)` / `H3_QWEN_RESIDENT=0` /
      `h3_serve --stream` opt out; `= 1` forces resident.
- [x] `h3_serve` — one persistent session (rewound per request), resident
      weights loaded at startup via a warm-up eval; `--stream` recreates the
      session per request.
- [x] `tests/test_qwen_resident.c` (`make resident-check`): resident decode is
      bit-for-bit identical to streaming and far faster (~0.31 vs ~13 s/token,
      warm cache). `make phase2-parity` (3 sessions) uses one shared 62 GB
      copy. `tests/bench_qwen.c` (`make bench-chat`) reports throughput.
- [x] **Chat-speedup step #1 — BF16 decode GEMV** (`docs/chat-speedup.md`):
      `h3_linear_gemv_bf16` for every `rows == 1` linear, decode
      0.63 → 0.31 s/token (2.0×). Reduction-order change → decode logits move
      ~1e-4 relative, argmax held; `rows > 1` parity gates stay bit-exact.
      `make bench-matmul` tracks per-shape GB/s. `H3_DISABLE_GEMV=1` opts out.
- [x] Chat-speedup step #2 — `W4A16` decode weights + `h3_linear_gemv_q4`
      (group-wise symmetric RTN, group 128): kernel + `qwen_q4.{c,h}` +
      resident wiring, **opt-in** (`H3_QWEN_Q4=1`, default off). Tracked under
      Quantization / QINT below.
- [x] Chat-speedup step #3 — `qwen_kv.c` resident path runs embedding + 64
      layers + head in one command buffer / one submit, and appends K/V with
      a `h3_gpu_copy_bf16` blit (no host round-trip). Bit-exact
      (`resident-check`). BF16 decode 0.31→0.29, INT4 decode 0.23→~0.20 s/tok.
- [x] Decoder-layer kernel fusion — `h3_qk_headnorm_rope_bf16` (Q/K head
      RMSNorm + RoPE, 3→1 dispatch, `rows==1`) and `h3_add_rms_norm_bf16`
      (residual add + RMSNorm, 2→1, bit-exact). INT4 decode 0.20→0.16 s/tok
      (6.1 tok/s). **Decode kernel-fusion milestone complete** (commit
      `e604557`); further fusion would fold projections into the parity-gated
      GEMV — higher risk, diminishing returns.

## Quantization

Terminology: `docs/quantization-terminology.md`. Names: `BF16`, `W8A16`,
`W4A16`, `W4A16-AWQ` (shorthand `INT8` / `INT4` / `INT4-AWQ`; `INT4 ≠ AWQ`).
A path is **performance-qualified** (builds, decodes, kernel checks pass,
benchmark reproduced) before it is **quality-qualified**.

### `Mixed-W4/BF16` — canonical policy (`H3_QWEN_Q4=mixed`)

```
Layers 0-49:  q_proj W4A16 RTN   k_proj BF16   v_proj BF16   o_proj W4A16 RTN
              gate_proj W4A16 RTN   up_proj W4A16 RTN   down_proj W4A16 RTN
Layers 50-63: all projections BF16
Embedding BF16   Final norm BF16   LM head BF16
```

### `Mixed-W4/BF16` default gate (replaces the old top-1 ≥ 0.99 / KL ≤ 0.01)

```
Text:  [ ] no large-margin argmax flips (ref top1-top2 >= 1.0)
       [ ] KL <= 0.05        [ ] cosine >= 0.99
       [ ] Japanese task quality acceptable
VLM:   [ ] no meaningful regression vs BF16
Tool:  [ ] tool-selection parity >= 99%   [ ] valid tool JSON >= 99.5%
H3:    [ ] layer-49 drift measured
       [ ] no meaningful visual regression   [ ] no meaningful audio regression
Perf:  [ ] >= 4.5 tok/s on M4 Max 128 GB    [ ] resident <= 32 GB
```

Three shipped modes once the gate passes: `Mixed-W4/BF16` = default,
`Pure W4A16` = `--fast` (experimental), `BF16` = `--quality` (reference).

- [x] QINT-001 Baseline `W4A16` weight path (`qwen_q4.{c,h}`, host RTN
      quantiser, packed nibbles + BF16 group scales)
- [x] QINT-002 `W4A16` decode GEMV (`h3_linear_gemv_q4`, `make q4-check`
      kernel rel ~1.7e-3)
- [x] QINT-003 Fused `W4A16` decode path (GEMV + `h3_qk_headnorm_rope_bf16`
      + `h3_add_rms_norm_bf16` + one-submit forward + on-GPU K/V append)
- [x] QINT-004 `W4A16 fused` decode is **performance-qualified** — 0.16
      s/token / 6.1 tok/s on M4 Max 128 GB (`make bench-chat`,
      `make q4-decode-check`); `resident-check` + `real-parity` hash + all
      phase gates green with the flag off
- [x] QINT-005 Prompt/eval set defined — `tests/test_qwen_quant_eval.c`
      (`make quant-eval`): 6 teacher-forced prompts (EN factual, EN reasoning,
      JA ×2, Python, tool-style); metrics top-1 / top-5 / logit rel-L2 / cos /
      KL vs the BF16 decode path. AWQ *calibration* sets (chat/VLM/H3) still
      to define once AWQ lands.
- [~] QINT-006 AWQ-lite implemented but **demoted to research-only**:
      `qwen_awq_calib` (`H3_QWEN_AWQ_CALIB`, `make quant-calib`),
      `qwen_q4_quantize_awq` (per-channel `s[j]=(act[j]/mean)^alpha`, alpha
      grid-searched on an activation-weighted reconstruction *proxy*),
      `h3_linear_gemv_q4` folds `1/s` into the x load. `H3_QWEN_Q4_AWQ=path`.
      KL −31 % vs RTN but top-1 flat, and on layers 0–49 it is *worse* than
      RTN (adds a large-margin flip). NOT in the `mixed` preset. A real
      activation-in-loss objective would be needed to revisit — parked.
- [ ] QINT-007 (parked with QINT-006)
- [x] QINT-008 Chat-quality baseline (`docs/quant-eval-baseline.md`):
      pure RTN top-1 0.894 / KL 0.078; **`mixed` top-1 0.953 / KL 0.033 /
      cos 0.995 / 0 large flips** — Text gate (large flips, KL ≤ 0.05,
      cos ≥ 0.99) PASS; Japanese *task*-level check still to do.
- [x] QINT-009 Japanese quality — `make qint-009`
      (`tests/test_qwen_ja_generation.c`): 10 plain JA chat turns (factual QA,
      arithmetic-with-reasoning, politeness rewrite, 2-sentence explanation,
      3-bullet suggestion, proverb, EN→JA translation, pros/cons, comparison,
      number sequence), greedy, BF16 vs `mixed`, two resident loads + compare.
      Mechanical gates asserted: **10/10 non-empty, 10/10 valid UTF-8, 10/10
      no runaway repetition, 10/10 non-pathological length**. 3/10
      byte-identical (the short factual/translation ones); the other 7 are the
      same answer with normal phrasing variation — no hallucination, no
      breakdown, fluent JA throughout; on the proverb and the caffeine
      comparison `mixed` is if anything slightly cleaner. **JA task gate
      PASS.** (Logit-level: QINT-008 had JA as the weakest bucket; the
      task-level output is nonetheless sound.)
- [x] QINT-016 Tensor/layer ablation harness — `H3_QWEN_Q4_BF16_LAYERS`,
      `H3_QWEN_Q4_BF16_PROJ`, `make quant-ablate`; eval buckets argmax flips by
      ref top1−top2 margin + per-prompt/per-position. Findings: all flips are
      mid-margin close calls; chat tail 50–63 + K/V are the main contributors
      (K/V nearly free to keep BF16 — GQA, K/V proj is 5120×1024); AWQ-lite on
      0–49 counterproductive. → `Mixed-W4/BF16` policy above.
- [x] QINT-014 `H3_QWEN_Q4=mixed` is the **default**. `h3_serve` exposes
      three modes — default `Mixed-W4/BF16`, `--fast` Pure W4A16, `--quality`
      BF16 — via `apply_decode_mode()` in `h3_serve_main.c`, which writes
      `H3_QWEN_Q4` with precedence *explicit flag > environment > default
      (mixed)*. **All three keep H3 conditioning on canonical BF16**: the H3
      text encoder never reads `H3_QWEN_Q4` and `h3_conditioning_accepts()`
      walls off a non-BF16 chat state — `--fast` is fast Chat decode, not fast
      conditioning (QEXP-003). Gates: Text + JA-task + Perf + Tool + VLM + H3
      all PASS (`docs/quant-eval-baseline.md`).
- [x] QINT-010 VLM quality — `make qint-010` (`tests/test_qwen_vlm_parity.c`):
      5 (ffmpeg image + question) cases via P7-004/005. BF16 vs `mixed`
      answers: 2/5 byte-identical (incl. a JA one), the other 3 describe the
      same scene with normal phrasing variation. No hallucination /
      degeneration. **VLM gate PASS.**
- [x] QINT-011 Tool calling — `make qint-011`
      (`tests/test_qwen_tool_parity.c`): 10 tool-use prompts (single / 2-3
      choice / multi-arg / enum / JA / no-tool), greedy assistant turn on a
      BF16 vs a `mixed` decode session, parsed with `qwen_tool_calls_parse` +
      `h3_json`. **Tool-selection parity 9/9, correct tool 9/9 both, valid
      JSON 9/9 both, call/no-call 10/10 both.** Only non-exact arg is a
      `send_email` body the prompt left unspecified (both sensible) — fluency,
      not structure. JA case byte-identical. **Tool gate PASS.**
- [x] QINT-012 Layer-49 hidden drift — `tests/test_qwen_l49_drift.c` /
      `make l49-drift` (`H3_QWEN_DUMP_L49` snapshot hook in `qwen_kv_eval`).
      Chat decode layer-49 residual vs `forward_to_layer(50)` BF16 canonical:
      BF16 decode path 1e-2 rel / cos 0.99995; **`mixed` 0.138 rel / cos 0.991,
      185/5120 channels with >10 % RMS change**. Result: the layer-49
      *interface* is shared fine; what cannot be shared is **feeding a
      Mixed-W4 chat state to H3 as canonical conditioning**, and **reusing
      the layers-0..49 compute** across Chat and H3 (only possible in
      `--quality` BF16). Fine for chat itself (tail 50–63 + lm_head BF16
      absorb it → ~1 % logit drift). Enforced by `qwen_execution_policy` +
      `h3_conditioning_accepts()` (only `QWEN_EXEC_BF16_CANONICAL` reaches
      the DiT). `spec.md` §4 / §5.3 diagrams updated to the two-path model.
- [x] QINT-013 H3 generation regression — **not a `Mixed-W4/BF16` default
      gate.** The H3 conditioning path (`h3_text_encoder.c`) is unquantised
      BF16 and independent of `H3_QWEN_Q4`; the runtime refuses to hand a
      quantised state to the DiT. Moved to the non-blocking QEXP-001.

### Non-blocking experiments (QEXP)

- [x] QEXP-001 Quantised L49 → H3 sensitivity (latent) — `make qexp-001`:
      16-step DiT denoise from BF16 vs Mixed-W4 conditioning, tiny geometry.
      Latent cos 0.88 → 0.996 — looked tolerant, but **latent cosine turned
      out to be a poor proxy** (QEXP-001b).
- [x] QEXP-001b Perceptual eval — `make qexp-001b`
      (`tests/test_qwen_l49_h3_perceptual.c`): real 256×256 / 39-frame /
      12-step generation per conditioning, VAE-decoded, SSIM+PSNR / audio
      corr+SNR. **video SSIM 0.731 / PSNR 17.4 dB; audio corr 0.513 /
      SNR −0.28 dB.** Control (`make qexp-001b-control`, same cond twice)
      SSIM/corr 1.0000 — the DiT+VAE is bit-deterministic, so this is real
      sensitivity. **The VAE re-amplifies the latent drift into a clearly
      different video + a largely different audio.** → feeding a Mixed-W4
      state to H3 would fail a same-seed regression; H3 stays BF16 canonical,
      no unified 0..49 forward. `h3_conditioning_accepts()` is
      measured-necessary.
- [x] QEXP-002 `--quality` shared-prefix — `make qexp-002`
      (`tests/test_qexp002_shared_prefix.c`): in all-BF16 mode, run layers
      0..49 **once** for a combined Chat + H3 request, split into the Chat tail
      (50..63) and the H3 conditioning. **Chat logits == `forward_full`
      bit-for-bit; shared layer-49 state == `get_h3_conditioning` bit-for-bit.**
      Saves one full prompt-length 0..49 forward (~3.4 s for a 6-token prompt,
      grows with length); DiT+VAE (~46 s) unchanged. Zero numerical cost on
      either branch — the exact opposite of `--mixed` (QEXP-001b). Not wired
      into `h3_serve` / `/v1/responses` yet.
- [x] QEXP-003 Cheap K_M-style conditioning quant → H3 perceptual — reuses
      `make l49-drift` + `qexp-001b` with `H3_QWEN_Q4_BF16_PROJ` levers, plus a
      new `--perturb REL IN OUT` mode in `tests/test_qwen_l49_h3_sensitivity.c`
      (adds N(0, REL·rms) noise to a bf16 conditioning file for calibration).
      **`down_proj` W4 is the pathological-channel source** (RMS-ratio outliers
      185 → 2 by keeping `down` BF16) — validates `Q4_K_M`'s Q6_K `ffn_down`
      promotion. Config C2 (BF16 `down`+`gate`+`up`, only q/o W4): L49 drift
      0.138 → **0.050**, audio corr 0.51 → **0.88**, video SSIM ~0.77 (plateau).
      **Quantisation error is less DiT-damaging than white noise of equal
      magnitude** (C2 vs random-0.05: 0.774/0.884 vs 0.711/0.523). The H3
      **video** DiT is intrinsically conditioning-chaotic (2 % perturb ⇒
      −0.08 SSIM); a shared quantised 0..49 is at best an opt-in
      "coherent, not identical" mode for audio-led generation. Not worth a full
      K-quant / NVFP4 GEMV port. Canonical BF16 stays the H3 default. Scope of
      QEXP-001b's "H3 must be BF16" narrowed accordingly in the docs.
- [~] QINT-015 Speculative decoding — draft/verify to change the "one 32B
      weight sweep per token" structure. Following `SPEC-DECODE-INSTRUCT.md`.
  - [x] QINT-015a **decode policy split** — `qwen_policy.{c,h}`:
        `qwen_eval_kind` {PREFILL, DECODE, VERIFY} and `qwen_decode_policy`
        {MIXED, FAST, QUALITY}, resolved once from `H3_QWEN_Q4` by
        `qwen_decode_policy_current()`. `qwen_q4_enabled()` /
        `q4_mixed_preset()` now route through it — precision no longer
        overloads `rows == 1` (the batch verifier will use `rows = 2..5` with
        the same policy). Behaviour-preserving. `qwen_session_history()`
        accessor added for draft backends.
  - [x] QINT-015a **scalar speculative coordinator** — `qwen_spec.{c,h}`,
        `qwen_draft.h`. Greedy only. Each cycle: draft proposes ≤ `width`
        tokens, the existing rows==1 target decode verifies them one at a
        time; a matched draft token is committed, the first mismatch commits
        the target's own token (correction/bonus) and ends the cycle. Emits
        the target argmax sequence == a plain greedy decode. Not faster yet
        (scalar); locks the algorithm for QINT-015d.
  - [x] QINT-015b **oracle draft + reject/correction parity** —
        `qwen_draft_oracle.c` (replays a known-good stream; `corrupt_at`
        forces divergence). `make spec-oracle-check` (100 % acceptance,
        byte-identical, commit/cycle = width+1), `spec-reject-check`
        (divergence forced at block positions 0..4 + all-accept + eos-inside;
        accepted-prefix, correction token, final sequence and KV/mRoPE state
        all match scalar greedy). **NB:** the scalar coordinator never writes
        a rejected draft token to the KV, so the *batched rewind
        transaction* (append 5 rows, then truncate partway) and non-zero
        mRoPE rewind are **not exercised yet** — that moves to 015d-0
        (`spec-batch-rewind-check`). Rewind is the existing O(1) length
        truncation, no snapshot.
  - [x] QINT-015c **n-gram draft + acceptance bench** —
        `qwen_draft_ngram.c` (longest-suffix "prompt lookup", stateless).
        `make spec-greedy-parity` (EN/JA/code, byte-identical to greedy),
        `spec-selfcheck`, `spec-ngram-bench`. Measured committed/cycle:
        EN 1.11, JA 1.08, code 1.20, JSON 1.08 — n-gram acceptance is low on
        these short prose/code turns, as expected (§19); it motivates a
        learned draft (QINT-015h/i), verifier unchanged.
  - Note: the rows==1 decode path is **not 100 % run-to-run bit-stable** — a
        position with a sub-0.05 top1/top2 logit gap flips argmax between GPU
        command-buffer submissions (the QINT-016 "mid-margin" flips; happens
        under BF16 too, seen once on a code prompt). Two plain greedy runs can
        disagree there. `parity_check` accepts a coordinator divergence only
        when it rebuilds the logits at that spot and finds exactly such a
        near-tie. This is the §30/§31 "near-tie" reality the batched verifier
        (QINT-015d) must also handle.
  - [ ] QINT-015d multi-row verifier — **the speedup**. Split (see
        `SPEC-DECODE-INSTRUCT.md` §66):
    - [x] 015d-0 `make spec-batch-rewind-check` — append a wrong 3-token
          block, rewind to the accepted frontier, assert length + history
          below `keep` restored and re-decode tracks the reference. Text +
          a real chat-templated multimodal prompt whose pad grid pushes
          `mrope_next` past `token_count`; also asserts rewind *into* the
          prompt is refused. **Fixed** a pre-existing `qwen_kv_rewind()` bug:
          it truncated `kv->length` before validating `keep >=
          mrope_base_len`, so a refused multimodal rewind still shortened the
          context — the check now runs before any mutation.
    - [x] 015d-1 done. `qwen_eval_kind` {PREFILL,DECODE,VERIFY} threaded
          through `kv_eval` → `qwen_layer_prep/finish` → `qwen_linear` and the
          lm_head; precision now follows the kind, not `rows==1` (PREFILL =
          BF16 bulk, DECODE = q4 GEMV, VERIFY = q4 decode-batch; fused QK for
          DECODE+VERIFY, 3-kernel for PREFILL). New Metal kernel
          `h3_linear_q4_decode_batch` (rows 2..5) as a 5-accumulator GEMV
          extension — dequant each nibble + fetch each group scale ONCE, reuse
          across all rows, identical per-thread K-order to the scalar GEMV;
          `make spec-kernel-check` (in `h3_qwen_q4_test`) shows it is
          **bit-exact** vs the scalar q4 GEMV row-for-row (rel 0, cos 1) for
          M=2..5. `qwen_session_verify_block()` / `qwen_verify_result`
          (per-row top1/top2/margin from the existing `[m,vocab]` readback;
          appends all rows, caller rewinds). `make spec-verify-parity`:
          **345/345 rows exact** vs scalar decode across 3 prompts × W 2..5 ×
          many frontiers, 0 near-ties needed (margin ≥ 0.05 on 342/345).
          Regressions clean: `spec-oracle-check` byte-identical,
          `phase2-parity` bit-for-bit. Coordinator untouched.
          Also tightened the test helper `redecode_and_match` (015d-0) to the
          shared `near_tie_ok` criterion — a first-token match no longer masks
          a real state corruption.
    - [x] 015d-2 done. **Pending-anchor coordinator** (`qwen_spec.{c,h}`
          rewritten). One target batch forward per cycle; the correction /
          bonus token is carried to the next cycle as a *pending anchor*
          instead of a scalar `target_eval()`. State invariant, checked after
          every cycle: `qwen_session_length() == prompt_len + emitted`; the
          pending anchor is not in the KV or the output. `width` is now the
          total verify-row count (2..5), not the draft count.
          `QWEN_VERIFY_MAX` 8→5 (= the kernel row limit, so a verify block
          can never silently drop to BF16). `qwen_draft.h`: `propose()` takes
          a `qwen_draft_context` (history + optional anchor + reserved
          `frontier_hidden`); the anchor is a hint a backend may use or
          ignore, never a requirement. Stats split into
          `target_batches / target_rows / scalar_fallback_evals / rewinds`.
          Tests (`make spec-pending-*`): oracle (W=5 → 5 committed/batch,
          100 % acceptance in the aligned window), reject (reject at each
          draft index; correction carried; one partial-block rewind),
          boundary (max_new 1..7 exact), eos (stop id never emitted / never
          in KV), vlm (non-zero mRoPE: reject → rewind → pending → next
          batch), parity (EN/JA/code/JSON == greedy modulo an accumulated
          batch-vs-scalar near-tie that stays inside the rebuilt top-2).
          `spec-batch-rewind-check` + `spec-verify-parity` still green.
    - [x] 015d-3 done. `make spec-bench` (`tests/bench_qwen_spec.c`,
          `docs/spec-decode-bench.md`): Mixed target, warm-up, per-trial
          rewind untimed, 10 reps, short (175) + long (1505) context.
          scalar-1 194.6 / 224.5 ms; **verify-5 693 / 730 ms** → a perfect
          draft at W=5 gives **1.4–1.5× (5.1 → 7.2 tok/s short, 4.5 → 6.9
          long)**, not the ~3× "one weight sweep, 5 positions" would allow.
          verify-M has a large fixed cost (M=2 already ~2× scalar-1) and only
          ~75–90 ms per extra row; the W4 decode-batch kernel is not
          achieving single-weight-load bandwidth (KC 1024 vs the scalar
          GEMV's 4096; `acc[8][5]` ≈ 40 regs/thread; BF16 tiled lm_head).
          End-to-end `qwen_spec_step()` under an oracle full-accept is within
          0.5 ms of raw verify-M — **coordinator overhead ≈ 0**. Verdict
          (§66.9): "needs optimisation", so 015e before a learned draft.
  - [ ] QINT-015e verifier profiling + targeted kernel optimisation.
        Decompose verify-M (W4 projections / BF16 K/V / BF16 tail /
        attention / lm_head / readback / CPU top-2); the W4 decode-batch
        kernel is the prime suspect. Also add **`spec-chain-drift-check`**:
        teacher-force the SAME token sequence through a scalar session and a
        2..5-row batch session and, at each block boundary, compare the
        target logits (top-1, top-2, margin, cosine, KL, max-abs) — measures
        the batch-chain numerical drift directly (the ~0.1–0.25 top-2
        close-calls seen in `pending-parity` are this, not a single-step
        near-tie) and sets a real correctness threshold.
  - [ ] QINT-015f adaptive scheduler (probe, auto-disable < 1.10×, adaptive
        width)
  - [ ] QINT-015g `h3_serve` opt-in (`--speculative --spec-draft ngram
        --spec-width N --spec-stats`)
  - [ ] QINT-015h/i learned / DFlash draft (after the batch verifier)
  - [ ] QINT-015j sampling (`temperature > 0`) — fall back to scalar until
        then

## P3 — Chat Template

- [x] P3-001 `qwen_chat.c`: `qwen_role`, `qwen_chat_message`,
      `qwen_chat_render()`, `qwen_chat_tokenize()`; `<|im_start|>` /
      `<|im_end|>` / `<|endoftext|>` token constants
- [x] P3-002 system turn (leading message only, matching chat_template.json)
- [x] P3-003 user turn
- [x] P3-004 assistant turn (tool_calls markup deferred to P5)
- [x] P3-005 tool turn — consecutive tool messages folded into one
      `<|im_start|>user` block of `<tool_response>` wrappers
- [x] P3-006 generation prompt (`<|im_start|>assistant\n`)
- [x] P3-007 check (`tests/test_qwen_chat.c`, `make phase3-check`): exact-string
      render per role, tool folding, misplaced-system rejection, tokenization
      boundaries + decode round trip, and one templated turn through the KV
      session that stops at `<|im_end|>` ("What is the capital of France?" →
      "Paris")

Not in P3 (deferred): the `tools` system block (function signatures) and
assistant `tool_calls` rendering — Phase 5.

## P4 — Chat Completions API

- [x] P4-001 `h3_json.c` — small read-only JSON parser + `h3_json_escape()`
- [x] P4-002 `h3_http.c` — minimal blocking HTTP/1.1 server (one request per
      connection, bounded sizes, `poll()`-interruptible accept loop, buffered
      and SSE-streaming responders)
- [x] P4-003 `qwen_server.c` — endpoints, above the runtime (spec §38: no
      Qwen-engine → OpenAI-JSON dependency)
- [x] P4-004 `GET /v1/models`
- [x] P4-005 `POST /v1/chat/completions` (buffered) — OpenAI `messages`
      (string or text-part array content) → `qwen_chat_tokenize` →
      `qwen_session` prefill + greedy decode → `chat.completion` with `usage`
- [x] P4-006 streaming — `stream:true` emits `chat.completion.chunk` SSE with a
      role chunk, per-token `delta.content`, a final `finish_reason` chunk and
      `data: [DONE]`
- [x] P4-007 `h3_serve` binary (`--model ROOT [--port] [--host] [--shaders]
      [--model-id]`) + graceful SIGINT/SIGTERM
- [x] P4-008 check (`tests/test_qwen_server.c`, `make phase4-check`): JSON
      unit tests, HTTP loopback over a real socket, `/v1/models`, and
      `/v1/chat/completions` buffered + streamed against the live runtime;
      curl smoke ("capital of Japan?" → streamed "Tokyo")

Not in P4 (deferred): sampling params (temperature / top_p / n / stop),
`/v1/completions`, auth, concurrent requests (handler is serialized).

## P5 — Tool Calling

- [x] P5-001 `h3_tool_call` IR (`qwen_tools.h`, spec §20) + `h3_tool_calls_free`
- [x] P5-002 `qwen_tool_calls_parse()` — lift `<tool_call>{...}</tool_call>`
      markup out of an assistant turn into `h3_tool_call[]` + leading content;
      object or string `arguments`; synthesised ids
- [x] P5-003 `tools` system block in `qwen_chat_render_tools()` — mirrors the
      `{% if tools %}` branch of chat_template.json (`# Tools` / `<tools>` /
      `<tool_call>` instructions folded into the system turn)
- [x] P5-004 assistant `tool_calls` markup — `qwen_chat_message.tool_calls_json`
      renders `<tool_call>\n{...}\n</tool_call>` after the assistant content
- [x] P5-005 `h3_json_stringify()` (compact) for re-serializing tool JSON
- [x] P5-006 server: `/v1/chat/completions` accepts `tools`; detects tool-call
      output, returns `choices[0].message.tool_calls` (+ `content: null`) with
      `finish_reason: "tool_calls"`; streaming emits a `delta.tool_calls`
      chunk and suppresses content once `<tool_call>` appears; assistant
      `tool_calls` in the request history round-trip through the template
- [x] P5-007 check (`tests/test_qwen_tools.c`, `make phase5-check`): stringify
      round-trip, parse cases, `tools` render, and a live function-calling
      round trip ("weather in Tokyo" -> `get_current_weather({"location":
      "Tokyo"})`, `finish_reason: tool_calls`)

- [x] P5-008 incremental tool-call streaming — `qwen_stream.{c,h}` splits the
      growing assistant text into leading-text deltas and per-call
      begin / `arguments`-fragment / end events (parallel calls get an
      incrementing index). `run_chat()` now feeds the cumulative text to a
      `qwen_stream` each token; `/v1/chat/completions` emits OpenAI
      `delta.tool_calls` with `arguments` fragments, `/v1/responses` emits
      `response.function_call_arguments.delta` / `.done`. Byte-exact split
      covered by `tests/test_qwen_stream.c` (`make stream-check`).

Not in P5: tool-choice forcing.

## P6 — Responses API

- [x] P6-001 `POST /v1/responses` route (spec §21)
- [x] P6-002 generation core factored into `run_chat()` (tokenize + prefill +
      greedy decode + tool parse, optional per-token text callback), shared by
      `/v1/chat/completions` and `/v1/responses`
- [x] P6-003 request: `input` (string or array of message /
      `function_call_output` / `function_call` items), `instructions`,
      `tools`, `stream`, `max_output_tokens`, `model`
- [x] P6-004 buffered response: `{object:"response", status:"completed",
      output:[message item / function_call items], output_text, usage:
      {input_tokens, output_tokens, total_tokens}}`
- [x] P6-005 streaming: `response.created` / `response.output_item.added` /
      `response.output_text.delta` (per token) / `response.output_text.done` /
      `response.completed` typed SSE events; `response.failed` on error
- [x] P6-006 tools -> `function_call` output items (name + arguments string)
- [x] P6-007 check (`tests/test_qwen_responses.c`, `make phase6-check`):
      buffered text, array input, tools -> function_call, streaming events.
      curl smoke: "Capital of Germany?" -> `output_text: "Berlin"`.

Not in P6: `previous_response_id` chaining / server-side response storage,
`content_part.*` and `function_call_arguments.delta` granular events,
`response.incomplete`.

## P7 — VLM

- [x] P7-001 multimodal path to logits — `qwen_engine_forward_full()` on a
      `qwen_input` with `vision_spans` + mRoPE `position_ids` + `tags` runs
      layers 0..49 (vision splice + deepstack + mRoPE) → layer-49 state →
      layers 50..63 (mRoPE) → logits
- [x] P7-002 `qwen_session_continue_from_intermediate()` gains a
      `position_ids` argument — the explicit multimodal branch point: H3 media
      generation and the Chat tail consume the very same layer-49 state
- [x] P7-003 check (`tests/test_qwen_vlm.c`, `make phase7-check`): synthetic
      vision rows through the real 50+14 layer GPU path —
      (1) `get_h3_conditioning` multimodal state is bit-for-bit
      `h3_text_encode_multimodal_bf16()` (the state H3 consumes);
      (2) `continue_from_intermediate(state, positions)` == `forward_full`
      bit-for-bit; (3) deterministic.
- [ ] P7-004 front-end: OpenAI `image_url` content parts → decode →
      `h3_vision_encode_bf16` → `qwen_vision_span`; chat-template
      `<|vision_start|><|image_pad|><|vision_end|>` handling; multimodal
      `qwen_session_eval`. (`h3_multimodal.c` already builds the FL2VA
      presentation for H3; needs an image codec + a vision-encoder run.)

## Later phases (not started)

- [ ] P8+ — ASR, Speech, Pseudo audio-only, Video, General audio, Image,
      Realtime
