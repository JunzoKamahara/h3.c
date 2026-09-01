# IMPLEMENTATION_STATUS

## Qwen Intermediate-State Interface

- [x] `qwen_input` defined — `qwen_engine.h`
- [x] `qwen_intermediate_state` defined — `qwen_engine.h`
- [x] `forward_to_layer` implemented — `qwen_session_forward_to_layer()`,
      `qwen_engine.c` (Phase 0 accepts `stop_layer` 1..50)
- [x] H3 conditioning wrapper implemented — `qwen_session_get_h3_conditioning()`
      fixes `stop_layer = 50`
- [x] Text parity verified — bit-for-bit vs `h3_text_encode_bf16()`
- [x] Multimodal parity verified — bit-for-bit vs
      `h3_text_encode_multimodal_bf16()` (synthetic spliced vision rows)
- [x] Tags parity verified — presentation tags round-trip unchanged
- [x] Existing H3 generation regression verified — `h3_real_prompt_test`
      layer-50 BF16 hash `e007b3a5097af1bf` and `submissions == 51` unchanged

## Phase 0 acceptance (spec section 35)

- [x] Existing `h3_text_encode_bf16()` output unchanged
- [x] Existing multimodal encoder output unchanged
- [x] layer-49 state shape = `[N, 5120]`
- [x] dtype = BF16
- [x] no final RMSNorm applied (the value is the post-layer-49 residual stream;
      see `docs/qwen-intermediate-state.md`)
- [x] H3 tags preserved
- [x] video generation: no regression (legacy `h3_text_encode_*` entry points
      are untouched in behaviour; they now share one code path with the new
      interface via `h3_text_encode_layers_bf16()`)

## Phase 1 — Full 64-layer Chat LLM

- [x] Decoder layers 50..63 — `qwen_lm_decode_tail()` in `qwen_lm.c`, same
      per-layer recipe / epsilon / theta / mRoPE construction as layers 0..49
- [x] Final RMSNorm — `model.language_model.norm.weight`
- [x] LM head — `lm_head.weight` `[151936, 5120]` (`tie_word_embeddings: false`)
- [x] Full-vocabulary logits + CPU argmax — `qwen_logits`
- [x] `qwen_engine_forward_full()` — spec §11 (`forward_to_layer(50)` → tail)
- [x] `qwen_session_continue_from_intermediate()` — spec §12
- [x] Boundary decomposition verified — `forward_full()` is bit-for-bit
      `continue_from_intermediate(get_h3_conditioning())`
- [x] Determinism verified — two `forward_full()` runs are bit-for-bit equal
- [x] Smoke — "The capital of France is" greedily decodes to " Paris"
- [x] Layer-49 boundary regression — `phase0-parity` + `h3_real_prompt_test`
      hash unchanged
- [ ] Numeric parity vs an external MLX/HF logits reference — pending a
      `misc/fixtures` golden (test has an optional `x.logits` compare path)
- KV cache, HTTP, tool calling — not started (Phase 2+)

## Design notes

- Phase 0 keeps `qwen_engine` / `qwen_session` as thin handles. Both the legacy
  `h3_text_encode_*` functions and the new `qwen_session_*` functions funnel
  into the single 50-layer implementation in `h3_text_encoder.c`
  (`text_encode_bf16_impl`), so Chat/VLM and H3 conditioning are guaranteed to
  run identical GPU work. Weight residency and the per-layer KV cache are
  deferred to Phase 2 as the spec directs.
- `qwen_intermediate_state_into_h3_text_embedding()` is the bridge to the
  legacy `h3_text_embedding` type (spec sections 17 / 18); it moves buffer
  ownership and drops `gpu_stats`, which is diagnostics rather than contract.
- Phase 1's `qwen_lm.c` is a separate translation unit that consumes the
  Phase 0 boundary and re-implements the per-layer recipe for layers 50..63
  rather than sharing `encode_layer` (which is `static` and wired to the
  50-layer prefetch machinery). The recipe, epsilon, theta and mRoPE table
  construction are copied verbatim and must stay in sync with
  `h3_text_encoder.c`; `make phase0-parity` guards layers 0..49 and
  `make phase1-parity` guards the tail's boundary decomposition and
  determinism. It streams the 14 tail layers one at a time (no prefetch, no KV
  cache) — deliberately simple for Phase 1.

## Not started

KV cache, chat template, HTTP, tool calling, Responses API,
audio/image/video — see `TASKS.md`.
