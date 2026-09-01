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

## Later phases (not started)

- [ ] P1 — Full 64-layer LLM (layers 50..63, final RMSNorm, LM head, logits,
      one-token generation)
- [ ] P2 — KV cache (prefill, incremental decode, rewind, multi-turn)
- [ ] P3 — Chat template (system / user / assistant / tool)
- [ ] P4 — Chat Completions API (`/v1/models`, `/v1/chat/completions`,
      streaming)
- [ ] P5 — Tool calling
- [ ] P6 — Responses API
- [ ] P7 — VLM (shared multimodal layer-49 state for H3 and Chat)
- [ ] P8+ — ASR, Speech, Pseudo audio-only, Video, General audio, Image,
      Realtime
