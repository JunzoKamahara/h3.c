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
benchmark reproduced) before it is **quality-qualified** (chat + Japanese +
logit/top-k vs BF16 + VLM + tool calling + layer-49 drift + H3 regression).
Default-on needs both.

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
- [ ] QINT-006 Implement AWQ calibration (per-channel activation-aware scale
      search)
- [ ] QINT-007 Generate `W4A16-AWQ` weights + scales (MLP `gate/up/down`
      first, then `q/o`, `k/v` last)
- [~] QINT-008 Chat-quality baseline recorded (`docs/quant-eval-baseline.md`):
      RTN `W4A16` = top-1 0.894, KL 0.078 vs BF16 decode; AWQ must beat this.
- [~] QINT-009 Japanese quality — baseline in QINT-008 (weakest: prompt 3
      KL 0.30). Re-run after AWQ.
- [ ] QINT-010 Evaluate VLM quality (image + text)
- [ ] QINT-011 Evaluate tool calling
- [ ] QINT-012 Measure layer-49 hidden drift (cosine / relative error)
- [ ] QINT-013 Evaluate H3 generation quality (conditioning cosine, prompt
      adherence, same-seed regression)
- [ ] QINT-014 Decide default quantization policy (per-tensor, per §11) —
      only if QINT-008..013 pass
- [ ] QINT-015 (later) Speculative decoding — draft/verify to change the
      "one 32B weight sweep per token" structure; higher expected value than
      further kernel fusion

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
