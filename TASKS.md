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

Not in P2 (deferred): weight residency (layer weights still streamed per eval),
sampling beyond greedy, HTTP, tool calling.

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

## Later phases (not started)

- [ ] P4 — Chat Completions API (`/v1/models`, `/v1/chat/completions`,
      streaming)
- [ ] P5 — Tool calling
- [ ] P6 — Responses API
- [ ] P7 — VLM (shared multimodal layer-49 state for H3 and Chat)
- [ ] P8+ — ASR, Speech, Pseudo audio-only, Video, General audio, Image,
      Realtime
