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

## Phase 2 — KV Cache

- [x] `h3_gqa_causal_kv_bf16` — new Metal kernel + `h3_gpu_gqa_causal_kv_bf16`
      wrapper. `query_rows` new queries attend a `kv_length`-row cache; query
      row i is at absolute position `kv_length - query_rows + i`. Reduces
      bit-for-bit to `h3_gqa_causal_bf16` when `query_rows == kv_length`.
- [x] `qwen_layers.c` — shared decoder-layer primitives (`qwen_layer_prep` /
      `qwen_layer_finish`, weight load/free, mRoPE table build with a position
      offset). Phase 1's `qwen_lm.c` was refactored onto it (no numeric
      change; `phase1-parity` still green).
- [x] Stateful `qwen_session` (`qwen_kv.c`): persistent GPU context + streamed
      weight store; resident embed / final-norm / lm_head; per-layer K/V caches
      held directly in GPU buffers sized to the session capacity
      (`H3_QWEN_KV_CAPACITY`, default 4096); token history; latest logits.
- [x] `qwen_session_eval()` — prefill (first call) and incremental decode
      (later calls): only the new tokens run projections/MLP; new RoPE'd K/V
      are appended to the cache and `h3_gpu_gqa_causal_kv_bf16` attends over it.
- [x] `qwen_session_sample()` greedy argmax; `qwen_session_logits()`,
      `qwen_session_length()`, `qwen_session_sync()`, `qwen_session_rewind()`.
- [x] Parity — `tests/test_qwen_kv.c` (`make phase2-parity`): prefill + greedy
      decode is bit-for-bit `qwen_engine_forward_full()` over the grown
      sequence (`max|dlogit| = 0`); chunked prefill == single-shot; rewind then
      re-eval reproduces earlier logits; two sessions deterministic.
- [x] Regressions — `phase0-parity`, `phase1-parity`, `h3_real_prompt_test`
      hash `e007b3a5097af1bf` / 51 submissions, `h3_tests` (1768) all green.
- [ ] Weight residency for the 64 decoder layers — still streamed per eval, so
      decode is correct and O(new tokens) in compute but not yet fast.
- Sampling beyond greedy, HTTP, tool calling — not started (Phase 3+).

## Phase 3 — Chat Template

- [x] `qwen_chat.c` — `qwen_chat_render()` builds the MiniMax-H3 / Qwen3-VL
      ChatML string, mirroring the non-tools path of `chat_template.json`:
      leading system turn, `user` / `assistant` turns, `tool` messages folded
      into one `<|im_start|>user` block of `<tool_response>` wrappers, optional
      trailing `<|im_start|>assistant\n` generation prompt.
- [x] `qwen_chat_tokenize()` = render + `h3_tokenizer_encode` (the tokenizer
      already maps `<|im_start|>` etc. to single ids).
- [x] Token constants `QWEN_TOKEN_IM_START/IM_END/ENDOFTEXT`; `<|im_end|>`
      (151645) is the turn/EOS stop.
- [x] Check — `tests/test_qwen_chat.c` (`make phase3-check`): exact-string
      render per role, tool folding, misplaced-system rejection, tokenization
      boundary counts + decode round trip, and one templated turn through the
      KV session ("What is the capital of France?" → "Paris", stops on
      `<|im_end|>`).
- [ ] `tools` system block + assistant `tool_calls` rendering — Phase 5.

## Phase 4 — Chat Completions API

- [x] `h3_json.c` — small read-only JSON parser (objects / arrays / strings
      with `\uXXXX` + surrogates / numbers / bool / null, depth-capped, parses
      a NUL-terminated copy) and `h3_json_escape()` for serialization.
- [x] `h3_http.c` — blocking HTTP/1.1: one request per connection, always
      `Connection: close`, 64 KiB header / 8 MiB body caps, `poll()` loop so
      SIGINT stops it within 0.5 s. Buffered (`h3_http_send`) and streaming
      (`h3_http_begin_stream` + `h3_http_write`) responders.
- [x] `qwen_server.c` — `GET /v1/models`; `POST /v1/chat/completions`. Parses
      OpenAI `messages` (string or `[{type:text,text}]` content), `stream`,
      `max_tokens`, `model`; renders with `qwen_chat_render`, runs a fresh
      `qwen_session` (prefill + greedy loop to `<|im_end|>` / `max_tokens`),
      incremental detokenization. Buffered reply is a `chat.completion` with
      `usage`; stream emits `chat.completion.chunk` SSE + `[DONE]`. One handler
      at a time (pthread mutex around the engine).
- [x] `h3_serve` binary + `docs` note; graceful shutdown.
- [x] Check — `tests/test_qwen_server.c` (`make phase4-check`): JSON units,
      HTTP loopback over a real socket, `/v1/models`, `/v1/chat/completions`
      buffered + streamed against the live runtime. Manual: `curl -N` streams
      "Tokyo" for "capital of Japan?" and stops with `finish_reason: stop`.
- Dependency direction holds (spec §38): `qwen_server` → `qwen_engine` /
  `h3_http` / `h3_json`; the engine has no HTTP or JSON dependency.
- [ ] Sampling params, `/v1/completions`, auth, request concurrency — later.

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
- `qwen_layers.c` holds the per-layer decoder recipe for layers 50..63 (Phase
  1) and every layer of the KV decoder (Phase 2). It does not share
  `h3_text_encoder.c`'s `encode_layer` (which is `static` and wired to the
  50-layer prefetch machinery); the recipe, epsilon, theta and mRoPE table
  construction are copied verbatim and must stay in sync. `make phase0-parity`
  guards layers 0..49, `phase1-parity` the tail boundary, `phase2-parity` that
  the KV path matches the full forward bit-for-bit.
- Phase 2's KV cache lives in GPU buffers (per layer, capacity-sized). New
  RoPE'd K/V rows are read back and written into the cache at their absolute
  offset each eval; attention uses `h3_gpu_gqa_causal_kv_bf16`. Two submits per
  layer per eval (prep, then append + attention + finish). Decoder-layer
  weights are still streamed per eval — residency is the outstanding Phase 2
  perf item; correctness and O(new tokens) compute are done.

## Not started

Tool calling, Responses API, audio/image/video, decoder-layer weight
residency, sampling beyond greedy — see `TASKS.md`.
