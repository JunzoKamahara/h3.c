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
- [x] **Decode GEMV kernel (chat-speedup step #1).** `h3_linear_gemv_bf16`
      (in `h3_shaders.metal`) replaces the MPSGraph batch-1 path for every
      `rows == 1` linear; ~190–250 GB/s vs ~110 on the wide shapes, decode
      **0.63 → 0.31 s/token (2.0×, ~3.2 tok/s)** steady state. Different
      reduction order from the tiled kernel → logits move ~1e-4 relative,
      argmax preserved; only single-token decode hits it, so the `rows > 1`
      parity gates stay bit-exact. `phase2-parity` compares decode on argmax
      + `rel_l2 < 3e-2`. `H3_DISABLE_GEMV=1` restores the tiled path. See
      `docs/chat-speedup.md`.
- [x] **`W4A16` decode weights + fusion (chat-speedup steps #2/#3).**
      `h3_linear_gemv_q4` (group-wise symmetric RTN, group 128) + `qwen_q4.{c,h}`
      + resident wiring, `H3_QWEN_Q4=1` (default off); the resident `qwen_kv.c`
      forward is one command buffer / one submit with an on-GPU K/V blit;
      `h3_qk_headnorm_rope_bf16` (Q/K head RMSNorm + RoPE, 3→1, `rows == 1`,
      ~1e-3 rel — prefill keeps the trio) and `h3_add_rms_norm_bf16` (residual
      add + RMSNorm, 2→1, bit-exact). The H3 path (`h3_text_encoder.c`) is
      separate BF16 code, so layer-49 parity is structurally untouched. **INT4
      fused decode 0.16 s/tok / 6.1 tok/s (M4 Max) — performance-qualified, not
      quality-qualified; opt-in only.** See `## Quantization Status` below and
      `docs/quantization-terminology.md` / `docs/chat-speedup.md` §3.1.
- [x] **Weight residency is the default.** All 64 decoder layers + embed /
      norm / lm_head are pinned in Unified Memory (~62 GB) on the first eval;
      decode ~0.31 s/token (with the step #1 GEMV kernel) vs ~13 s/token
      streaming (`make resident-check`, resident vs streaming bit-for-bit
      identical). If the allocation does not fit, the session
      falls back to streaming with a stderr note.
      `qwen_session_set_resident(session, 0)` / `H3_QWEN_RESIDENT=0` forces
      streaming; `= 1` forces resident (hard error if it will not fit).
      `h3_serve` loads it once at startup (warm-up eval) and reuses one
      persistent session rewound per request; `h3_serve --stream` opts out
      (and then recreates the session per request so per-eval Metal
      allocations do not pile up).
      Backed by a process-wide, reference-counted shared set in `qwen_kv.c`
      (`g_resident`, `resident_acquire()` / `resident_release()` under a
      mutex): the first resident session loads it, every other borrows it, so
      N sessions cost one copy (`make phase2-parity`, 3 sessions, one 62 GB
      load).
- [ ] Streaming decode (`--stream` / `H3_QWEN_RESIDENT=0`) is still ~13
      s/token (weight I/O bound; the GEMV kernel only helps the compute
      fraction). INT4 tail weights + per-layer submit fusion are the next
      work (chat-speedup steps #2/#3, see `docs/chat-speedup.md`).
- Sampling beyond greedy, tool calling — not started (Phase 5+).

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

## Phase 5 — Tool Calling

- [x] `qwen_tools.{c,h}` — `h3_tool_call { id, name, arguments }` (spec §20)
      and `qwen_tool_calls_parse()`: scans an assistant turn for
      `<tool_call>{json}</tool_call>` blocks, returns the leading text plus the
      parsed calls (object or string `arguments`, synthesised ids).
- [x] `qwen_chat.c` — `qwen_chat_render_tools()` / `qwen_chat_tokenize_tools()`
      add the `tools` system block (`# Tools` + `<tools>` signatures +
      `<tool_call>` instructions, per chat_template.json) and render
      `qwen_chat_message.tool_calls_json` as `<tool_call>` markup after the
      assistant content. `qwen_chat_render` / `qwen_chat_tokenize` are now thin
      wrappers with no behaviour change (Phase 3 test still exact).
- [x] `h3_json.c` — `h3_json_stringify()` (compact) to re-serialize tool
      definitions and `arguments` objects.
- [x] `qwen_server.c` — `/v1/chat/completions` accepts `tools`; after
      generation, `qwen_tool_calls_parse` splits the turn. Buffered response:
      `message.tool_calls` (+ `content: null`) and `finish_reason:
      "tool_calls"`. Streaming: content deltas stop once `<tool_call>` appears,
      then a `delta.tool_calls` chunk is emitted before the finish chunk.
      Assistant `tool_calls` in the request history feed back through the
      template.
- [x] Check — `tests/test_qwen_tools.c` (`make phase5-check`): stringify
      round-trip; parse (leading text, one/many blocks, object/string args,
      no-markup); `tools` render; live round trip -- "weather in Tokyo? call
      get_current_weather" -> `tool_calls:[{... "name":"get_current_weather",
      "arguments":"{\"location\":\"Tokyo\"}"}]`, `finish_reason: tool_calls`.
- [x] Incremental tool-call streaming — `qwen_stream.{c,h}` is a state machine
      fed the cumulative decoded text after every token; it emits leading-text
      deltas and, per `<tool_call>` block, a begin (name) / `arguments`
      fragments / end sequence with an incrementing index for parallel calls.
      Invariant (checked by `tests/test_qwen_stream.c`, `make stream-check`):
      the fragments concatenate to exactly the raw arguments value.
      `run_chat()` feeds a `qwen_stream`; `/v1/chat/completions` emits
      `delta.tool_calls` with `arguments` fragments, `/v1/responses` emits
      `response.function_call_arguments.delta` + `.done`.
- [ ] Tool-choice forcing.

## Server: session lifecycle

- The server keeps one `qwen_session`. In `--resident` mode it is rewound to
  empty per request and reuses the pinned weights. In streaming (non-resident)
  mode `server_reset_session()` recreates it per request, so per-eval Metal
  allocations from one request's decode do not carry into the next. A single
  very long non-resident generation can still exhaust Metal memory (freed
  weight buffers are not promptly returned to the OS across hundreds of
  load/free cycles); `--resident` is the path for sustained use, and the
  heavier server tests run resident.

## Phase 6 — Responses API

- [x] `run_chat()` in `qwen_server.c` — the generation core (tokenize -> prefill
      the persistent session -> greedy decode with an optional per-token text
      callback -> `qwen_tool_calls_parse`), now shared by
      `/v1/chat/completions` and `/v1/responses`; `handle_chat_completion` was
      re-pointed at it with no behaviour change (phase4/phase5 still pass).
- [x] `POST /v1/responses` (`handle_responses`): `input` as a string or an
      array of message / `function_call_output` / `function_call` items,
      `instructions` (-> leading system turn), `tools`, `stream`,
      `max_output_tokens`.
- [x] Buffered: `{object:"response", status:"completed", output:[…],
      output_text, usage:{input_tokens,output_tokens,total_tokens}}`. `output`
      is an assistant `message` item (`output_text` content part) plus a
      `function_call` item per tool call.
- [x] Streaming: typed SSE — `response.created`, `response.output_item.added`,
      `response.output_text.delta` (per token), `response.output_text.done`,
      `response.completed`; `response.failed` on error. `content_part.*` and
      `function_call_arguments.delta` are omitted (clients rebuild from
      `response.completed`).
- [x] Check — `tests/test_qwen_responses.c` (`make phase6-check`): buffered
      text, array input, tools -> `function_call`, streaming events. curl:
      "Capital of Germany?" -> `output_text: "Berlin"`.
- [ ] `previous_response_id` chaining / stored responses; granular
      `content_part` / arguments-delta events.

## Phase 7 — VLM

- [x] Multimodal → logits already flows through the Phase 0/1 boundary:
      `qwen_engine_forward_full()` given a `qwen_input` with `vision_spans` +
      axis-major `position_ids` + `tags` runs layers 0..49 (vision embedding
      splice, deepstack after layers 0/1/2, mRoPE) into the layer-49 state,
      then `qwen_lm_decode_tail()` runs layers 50..63 with the same mRoPE
      positions.
- [x] `qwen_session_continue_from_intermediate()` takes `position_ids` (NULL =
      sequential text). This is the explicit multimodal branch point: the same
      layer-49 state feeds H3 media generation and the Chat tail.
- [x] Check — `tests/test_qwen_vlm.c` (`make phase7-check`), synthetic vision
      rows through the real GPU path: the runtime's multimodal layer-49 state
      is bit-for-bit `h3_text_encode_multimodal_bf16()` (what H3 consumes);
      `continue_from_intermediate(state, positions)` is bit-for-bit
      `forward_full(multimodal input)`; deterministic.
- [ ] Front-end: `image_url` → pixels → `h3_vision_encode_bf16` →
      `qwen_vision_span`; `<|vision_start|>…<|vision_end|>` in the chat
      template; multimodal `qwen_session_eval`. `h3_multimodal.c` already has
      the FL2VA presentation builder used by H3.

## Quantization Status

Terminology: `docs/quantization-terminology.md`. Task tracker: `## Quantization`
(QINT-001+) in `TASKS.md`.

### BF16
Status: canonical / verified. The reference every quantized path is measured
against; layers 0..49 stay BF16 for H3 conditioning.

### INT4 (`W4A16`, group-wise symmetric RTN, group 128)
Status: **performance-qualified**, not quality-qualified.
- Decode: 0.16 s/token ≈ 6.1 tok/s on M4 Max 128 GB (`bench-chat`); 3.9× over
  BF16 tiled, 1.9× over BF16 GEMV.
- Kernel + fusion: complete (`h3_linear_gemv_q4`, `h3_qk_headnorm_rope_bf16`,
  `h3_add_rms_norm_bf16`, one-submit forward, on-GPU K/V append).
- Checks green: `q4-check`, `q4-decode-check`, `resident-check`, `real-parity`
  hash `e007b3a5097af1bf`, all phase gates (flag off).
- AWQ: not implemented. Quality validation (chat / Japanese / logit-topk /
  VLM / tool calling / layer-49 drift / H3 regression): pending (QINT-008+).
- Default: **no** — opt-in `H3_QWEN_Q4=1`. Naive RTN flips greedy tokens
  ~step 4 vs BF16.

### INT4-AWQ (`W4A16-AWQ`)
Status: **AWQ-lite implemented, insufficient.** `qwen_awq_calib` capture
(`H3_QWEN_AWQ_CALIB` / `make quant-calib`), `qwen_q4_quantize_awq`
(per-channel `s[j]=(act[j]/mean)^alpha`, alpha grid-searched on an
activation-weighted reconstruction *proxy*), `1/s` folded into the decode
GEMV x-load (decode still 0.16 s/tok). `H3_QWEN_Q4_AWQ=path`,
`make quant-eval-awq`.
- Result (`docs/quant-eval-baseline.md`): KL 0.078 → 0.054 (−31 %), rel-L2 and
  cosine improve, but **top-1 0.894 → 0.882 (flat/within noise)**.
- The diagonal proxy is not enough. Next: real activation-in-loss objective,
  clip search, or mixed precision (QINT-006 refinement).

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

VLM front-end (image_url decode), audio/image/video generation, sampling
beyond greedy — see `TASKS.md`.
