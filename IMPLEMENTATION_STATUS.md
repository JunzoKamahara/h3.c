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

## Design notes

- Phase 0 keeps `qwen_engine` / `qwen_session` as thin handles. Both the legacy
  `h3_text_encode_*` functions and the new `qwen_session_*` functions funnel
  into the single 50-layer implementation in `h3_text_encoder.c`
  (`text_encode_bf16_impl`), so Chat/VLM and H3 conditioning are guaranteed to
  run identical GPU work. Weight residency, the per-layer KV cache and the
  layer-49 continuation path (`qwen_session_continue_from_intermediate`) are
  deferred to Phase 1 / Phase 2 as the spec directs.
- `qwen_intermediate_state_into_h3_text_embedding()` is the bridge to the
  legacy `h3_text_embedding` type (spec sections 17 / 18); it moves buffer
  ownership and drops `gpu_stats`, which is diagnostics rather than contract.

## Not started

Layers 50..63, final RMSNorm, LM head, logits, KV cache, chat template, HTTP,
tool calling, Responses API, audio/image/video — see `TASKS.md`.
