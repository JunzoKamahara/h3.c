# Qwen layer-49 intermediate state — exact semantics (P0-001 / P0-002)

This records what the released H3 conditioning tensor *is*, as implemented today
in `h3_text_encoder.c`, so the Phase 0 runtime boundary (`qwen_engine.h`) can be
held to it bit-for-bit.

## Principle (revised after QINT-012)

The layer-49 boundary is a **shared semantic interface** between Chat and H3
(`[N, 5120]` unnormalised hidden, tensor semantics, tokenizer / vision /
runtime architecture — all common). It is **not** a guarantee that the
layers-0..49 *numbers* are identical across execution policies:

- `QWEN_EXEC_BF16_CANONICAL` — layers 0..49 all BF16. The only state H3
  conditioning accepts (`h3_conditioning_accepts()`). In `--quality` mode the
  Chat path also uses this, so the 0..49 compute can be shared for a combined
  Chat + H3 request.
- `QWEN_EXEC_MIXED_W4_BF16` (`--mixed`) / `QWEN_EXEC_W4_FAST` (`--fast`) —
  Chat-decode-only. QINT-012 measured `mixed` at ~14 % relative / cos 0.991
  drift vs canonical, with 185/5120 channels off by >10 % RMS. Never fed to
  H3; H3 always runs its own BF16 0..49 (`h3_text_encoder.c`, streamed on
  demand). See `docs/quant-eval-baseline.md`.

## Entry points

| Function | Input | Layers run |
|---|---|---|
| `h3_text_encode_bf16()` | text token ids only | 0..49 |
| `h3_text_encode_multimodal_bf16()` | tokens + vision presentation spans + `position_ids` + `tags` | 0..49 |
| `h3_text_encode_layers_bf16()` (new) | text token ids only | `0..layer_count-1` |
| `h3_text_encode_multimodal_layers_bf16()` | multimodal | `0..layer_count-1` |

All four call one implementation, `text_encode_bf16_impl()`, with
`layer_count` (the full paths pass `TEXT_LAYERS == 50`).

## Backbone configuration

```
layers (released)      50          # of 64; layer 49 is the H3 cut
vocab                  151936
hidden_size            5120
mlp intermediate       25600
query heads            64
kv heads               8           # GQA
head dim               128
attention scale        1/sqrt(128)
RMSNorm epsilon        1e-6
RoPE theta             5_000_000
```

Weights (all BF16), from the `text_encoder` safetensors shards:

- `model.language_model.embed_tokens.weight` — `[151936, 5120]`
- `model.language_model.layers.<i>.input_layernorm.weight`
- `model.language_model.layers.<i>.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
- `model.language_model.layers.<i>.self_attn.{q_norm,k_norm}.weight` — `[128]`
- `model.language_model.layers.<i>.post_attention_layernorm.weight`
- `model.language_model.layers.<i>.mlp.{gate_proj,up_proj,down_proj}.weight`

## Per-layer computation (`encode_layer`)

1. `norm = RMSNorm(hidden, input_layernorm)`
2. `q = norm @ q_projᵀ`, `k = norm @ k_projᵀ`, `v = norm @ v_projᵀ`
3. per-head `RMSNorm(q, q_norm)`, `RMSNorm(k, k_norm)` (head dim 128)
4. RoPE applied to `q`, `k` (text mRoPE tables, see below)
5. causal GQA attention → `attn_heads`
6. `hidden += attn_heads @ o_projᵀ`   (residual)
7. `norm = RMSNorm(hidden, post_attention_layernorm)`
8. `gate = norm @ gate_projᵀ`, `up = norm @ up_projᵀ`
9. `act = silu(gate) * up`   (fused)
10. `hidden += act @ down_projᵀ`   (residual)

All activations are BF16. Attention/linears run through the same Metal kernels
as the rest of h3.c.

## Multimodal splicing

- After the embedding lookup, base token embeddings in each span's row range
  `[start, start + tokens)` are **overwritten** by the span's `embeddings`.
- `deepstack[0]`, `deepstack[1]`, `deepstack[2]` are **added** to `hidden`
  immediately after language layers 0, 1 and 2 respectively (only when spans are
  present).
- `tags` are carried straight through to the output (values `0..2`; `1` marks
  language rows, `0` marks a Qwen vision span including its boundary tokens).

## RoPE / mRoPE tables

Half dimension is 64. Inverse frequencies: `inv_freq[i] = 1 / theta^(2i/128)`.

Per position `p`, per index `i`:

- axis selection: if `position_ids` is provided **and** `i < 60` **and**
  `i % 3 == 1` → axis 1; `i % 3 == 2` → axis 2; otherwise axis 0.
- coordinate: `position_ids[axis * tokens + p]` when provided, else the plain
  sequential index `p`.
- `angle = coordinate * inv_freq[i]`, then `cos`/`sin`.
- **When `position_ids` is provided**, `cos`/`sin` are rounded to BF16 before
  being handed to the fused kernel; the text-only path keeps full F32.

Consequence: the text-only and `position_ids` code paths are *not*
interchangeable. `qwen_input` therefore requires `position_ids == NULL` and
`tags == NULL` for the text-only path and requires both for the multimodal
path — matching the two legacy entry points exactly.

## Output tensor — the intermediate state

```
values   BF16, row-major [tokens, 5120]   # hidden AFTER decoder layer 49
width    5120
tokens   token_count
tags     copy of input tags, [tokens], or NULL for text-only
```

- The value is the **unnormalized residual stream** after layer 49. The final
  language-model RMSNorm is **not** applied here (it belongs to Phase 1, before
  the LM head).
- GPU submission count for the full 50-layer run is exactly **51**
  (1 embedding + 50 layers). `h3_real_prompt_test` asserts this and the
  layer-50 BF16 hash `e007b3a5097af1bf` for the prompt
  "A red fox walking through snow"; both are release-blocking.

## Phase 0 mapping

| spec name | this repo |
|---|---|
| `qwen_intermediate_state` | `qwen_engine.h` (own type; `into_h3_text_embedding` bridges to `h3_text_embedding`) |
| `qwen_forward_to_layer(stop_layer)` | `qwen_session_forward_to_layer()`, `stop_layer` 1..50 |
| `qwen_get_h3_conditioning()` | `qwen_session_get_h3_conditioning()` — fixes `stop_layer = 50` |
| parity test | `tests/test_qwen_intermediate.c` → `make phase0-parity` |

## Phase 1 — layers 50..63 → logits (`qwen_lm.c`)

Continues from the layer-49 intermediate state:

```
hidden[N,5120] BF16  (layer 49, unnormalized)
      │
decoder layers 50..63          # 14 layers, same recipe as 0..49
      │
final RMSNorm                  # model.language_model.norm.weight, eps 1e-6
      │
lm_head.weight [151936,5120]   # tie_word_embeddings = false
      │
logits[N,151936]  → take last row → F32 → CPU argmax
```

- RoPE for 50..63 rebuilds the *same* mRoPE table as 0..49 (sequential text
  positions, or the input's `position_ids`); `mrope_section` is `[24,20,20]`
  interleaved, which the `index < 60 && index%3` axis split reproduces.
- `qwen_engine_forward_full()` = `forward_to_layer(50)` + this tail (spec §11).
  `qwen_session_continue_from_intermediate()` = this tail alone, assuming
  sequential positions (spec §12).
- No KV cache; full-prompt forward. Tail weights are streamed per layer.

| spec name | this repo |
|---|---|
| `qwen_engine_forward_full()` | `qwen_engine_forward_full()` in `qwen_engine.c` |
| `qwen_session_continue_from_intermediate()` | same name; core is `qwen_lm_decode_tail()` in `qwen_lm.c` |
| `qwen_logits` | `qwen_engine.h` — last-position `[vocab]` F32 + `argmax_token` |
| parity test | `tests/test_qwen_lm.c` → `make phase1-parity` |

## Phase 2 — KV cache (`qwen_kv.c`, `qwen_layers.c`, new Metal kernel)

Stateful `qwen_session` for multi-turn chat. The Phase 0/1 entry points stay
stateless; a KV context is created on the first `qwen_session_eval()`.

```
qwen_session_eval(tokens)      first call = prefill, later = incremental decode
  embedding(new tokens)
  for layer 0..63:
    qwen_layer_prep -> RoPE'd Q/K/V for the new rows   (positions past..past+m)
    append new K/V rows into the per-layer GPU cache at row `past`
    h3_gpu_gqa_causal_kv_bf16(Q_new, K_cache[0:past+m], V_cache[0:past+m])
    qwen_layer_finish
  final RMSNorm + lm_head on the last new row -> latest logits + argmax
qwen_session_sample()          greedy argmax of the latest logits
qwen_session_rewind(keep)      cache length / history / position -> keep
qwen_session_length(), qwen_session_logits(), qwen_session_sync()
```

- `h3_gpu_gqa_causal_kv_bf16` (kernel `h3_gqa_causal_kv_bf16`) is a copy of the
  plain causal GQA kernel with `key_count = (kv_length - query_rows) + i + 1`.
  With `query_rows == kv_length` (prefill, past 0) it is bit-identical, so a
  KV prefill + greedy decode reproduces `qwen_engine_forward_full()` over the
  grown sequence **bit-for-bit** — the `make phase2-parity` anchor.
- K/V caches are GPU buffers, one pair per layer, sized to the session
  capacity (`H3_QWEN_KV_CAPACITY`, default 4096). Rewind just moves `length`;
  stale rows are overwritten on the next eval.
- Decoder-layer weights: **resident by default** -- all 64 layers + embed /
  norm / lm_head pinned in Unified Memory (~62 GB), loaded on the first eval,
  decode ~0.7-1.4 s/token. If the allocation does not fit, the session falls
  back to streaming weights per eval (~14 s/token).
  `qwen_session_set_resident(session, 0)` / `H3_QWEN_RESIDENT=0` forces
  streaming; `= 1` forces resident (hard error if it will not fit). Both paths
  are bit-for-bit identical (`make resident-check`). The resident set is a
  process-wide, reference-counted singleton (`resident_acquire` /
  `resident_release` in `qwen_kv.c`): the first resident session loads it,
  others borrow the same copy, so N sessions do not cost N x 62 GB. `h3_serve`
  loads it once at startup and reuses one persistent session rewound per
  request; `h3_serve --stream` opts out.

| spec name | this repo |
|---|---|
| `qwen_layer_kv { k, v, capacity, length }` | per-layer GPU `k_cache[64]` / `v_cache[64]` + a single `length` in `struct qwen_kv_context` (`qwen_kv.c`) |
| `h3_session_eval` / `_sample` / `_rewind` / `_sync` / `_free` | `qwen_session_eval` / `_sample` / `_rewind` / `_sync` / `_free` |
| parity test | `tests/test_qwen_kv.c` → `make phase2-parity` |

## Phase 3 — chat template (`qwen_chat.c`)

`qwen_chat_render(messages, count, add_generation_prompt)` produces the
MiniMax-H3 ChatML string; `qwen_chat_tokenize()` then runs the tokenizer
(which already maps `<|im_start|>` and friends to single ids).

```
[system]     <|im_start|>system\n{content}<|im_end|>\n   (messages[0] only)
user         <|im_start|>user\n{content}<|im_end|>\n
assistant    <|im_start|>assistant\n{content}<|im_end|>\n
tool (run)   <|im_start|>user\n<tool_response>\n{c}\n</tool_response>
             [ \n<tool_response>\n{c}\n</tool_response> ]*   <|im_end|>\n
[gen prompt] <|im_start|>assistant\n
```

Stop token for an assistant turn is `<|im_end|>` (151645). The `tools` system
block and assistant `tool_calls` markup are Phase 5.

| spec name | this repo |
|---|---|
| roles system / user / assistant / tool | `qwen_role`, `qwen_chat_message` |
| chat template | `qwen_chat_render()` / `qwen_chat_tokenize()` |
| check | `tests/test_qwen_chat.c` → `make phase3-check` |

## Phase 4 — OpenAI-compatible server (`qwen_server.c`, `h3_http.c`, `h3_json.c`)

```
h3_serve --model MiniMax-H3 [--port 8080] [--host 127.0.0.1] [--stream]

GET  /v1/models              -> {"object":"list","data":[{"id":"minimax-h3",...}]}
POST /v1/chat/completions    body: {model?, messages[], stream?, max_tokens?}
     stream=false -> chat.completion { choices[0].message.content, usage }
     stream=true  -> SSE: chat.completion.chunk (role, then delta.content per
                     token, then finish_reason) then `data: [DONE]`
```

Per request: `qwen_chat_render(messages, gen_prompt=1)` -> tokenize -> a fresh
`qwen_session` -> prefill (`qwen_session_eval`) -> greedy loop
(`qwen_session_sample` / `qwen_session_eval`) until `<|im_end|>` or
`max_tokens`, detokenizing incrementally. Greedy only; `temperature` / `top_p`
/ tool calls / `/v1/responses` are later phases.

Dependency direction (spec §38): `qwen_server` depends on `qwen_engine` /
`h3_http` / `h3_json`; the runtime has no HTTP or JSON dependency.

| spec name | this repo |
|---|---|
| `GET /v1/models` | `handle_models()` |
| `POST /v1/chat/completions` + streaming | `handle_chat_completion()` |
| HTTP server / SSE | `h3_http.c` (`h3_http_send` / `h3_http_begin_stream`) |
| check | `tests/test_qwen_server.c` → `make phase4-check` |

## Phase 5 — tool calling (`qwen_tools.c`, `qwen_chat.c`, `qwen_server.c`)

```
request  {"tools":[{"type":"function","function":{...}}], "messages":[...]}
  qwen_chat_render_tools  -> system turn gets a # Tools block:
     <|im_start|>system\n{system}\n\n# Tools ... <tools>\n{tool_json}...\n</tools>
     \n\nFor each function call ... <tool_call>\n{...}\n</tool_call><|im_end|>\n
  decode as usual
  qwen_tool_calls_parse(assistant_text)  ->  leading content + h3_tool_call[]
     from <tool_call>\n{"name":..,"arguments":..}\n</tool_call> blocks
response {"choices":[{"message":{"content":null,
          "tool_calls":[{"id":"call_0001","type":"function",
            "function":{"name":"..","arguments":"<json string>"}}]},
          "finish_reason":"tool_calls"}]}
```

- `h3_tool_call { char *id; char *name; char *arguments; }` (spec §20); the
  parser accepts `arguments` as a JSON object (compacted with
  `h3_json_stringify`) or a raw string.
- assistant `tool_calls` in the request history are rendered back as
  `<tool_call>` markup via `qwen_chat_message.tool_calls_json`, so multi-turn
  function calling round-trips.
- streaming: `qwen_stream.c` splits the growing text incrementally -- content
  deltas until `<tool_call>`, then per call a begin chunk (name,
  `arguments:""`), `arguments` string fragments, and a close. `/v1/responses`
  maps these to `response.function_call_arguments.delta` / `.done`. Parallel
  calls carry an incrementing `index`. Fragments concatenate to exactly the
  raw arguments value (`make stream-check`).

| spec name | this repo |
|---|---|
| `h3_tool_call` IR | `qwen_tools.h` |
| Qwen markup parse | `qwen_tool_calls_parse()` |
| `tools` block + assistant tool_calls render | `qwen_chat_render_tools()` |
| OpenAI `tool_calls` serialization | `append_tool_calls_array()` in `qwen_server.c` |
| check | `tests/test_qwen_tools.c` → `make phase5-check` |

## Phase 6 — Responses API (`qwen_server.c`)

`POST /v1/responses`. The generation core is factored into `run_chat()`
(tokenize → prefill the persistent session → greedy decode with an optional
per-token text callback → `qwen_tool_calls_parse`), shared with
`/v1/chat/completions`.

```
request  {"instructions":"...", "input": "..." | [ {role,content} | 
          {type:"function_call_output",call_id,output} | {type:"function_call",...} ],
          "tools":[...], "stream":?, "max_output_tokens":N}
buffered {"object":"response","status":"completed",
          "output":[{"type":"message","role":"assistant",
                     "content":[{"type":"output_text","text":"..."}]},
                    {"type":"function_call","name":"..","arguments":"<json>"}],
          "output_text":"...","usage":{"input_tokens":,"output_tokens":,"total_tokens":}}
stream   event: response.created / response.output_item.added /
         response.output_text.delta (per token) / response.output_text.done /
         response.completed          (response.failed on error)
```

Omitted: `previous_response_id` chaining, stored responses, `content_part.*`
and `function_call_arguments.delta` granular events.

| spec name | this repo |
|---|---|
| `POST /v1/responses` | `handle_responses()` |
| generation core | `run_chat()` (also backs `/v1/chat/completions`) |
| response object | `append_response_object()` / `append_response_output()` |
| check | `tests/test_qwen_responses.c` → `make phase6-check` |

## Phase 7 — VLM (spec §22)

The multimodal path reuses the Phase 0/1 boundary end to end:

```
image -> vision encoder (h3_vision_encode_bf16) -> h3_vision_output
      -> presentation (h3_multimodal.c: <Picture n>, <|vision_start|>...)
      -> qwen_input { token_ids, vision_spans, position_ids (mRoPE), tags }
      -> layers 0..49  (vision embedding splice, deepstack after 0/1/2, mRoPE)
      -> layer-49 intermediate state  [N,5120] BF16
                 /                         \
        H3 media generation        qwen_lm_decode_tail(state, position_ids)
                                   layers 50..63 (mRoPE) -> final norm -> lm_head
                                   -> logits -> Chat / VLM
```

`qwen_engine_forward_full()` does image+text -> logits when the `qwen_input`
carries vision spans; `qwen_session_continue_from_intermediate(state,
position_ids, ...)` is the explicit branch point on the shared state.
`make phase7-check` proves the multimodal layer-49 state is bit-for-bit what
H3 consumes and that the Chat tail on it matches a one-shot forward.

The `image_url` decode front-end (pixels -> `qwen_vision_span`, the
`<|vision_start|>` chat-template slot) is P7-004, not yet wired.

| spec name | this repo |
|---|---|
| vision encoder | `h3_vision_encode_bf16()` (`h3_vision_encoder.c`) |
| FL2VA presentation | `h3_multimodal_encode_fl2va_bf16()` |
| multimodal -> logits | `qwen_engine_forward_full()` (vision spans in `qwen_input`) |
| shared-state branch | `qwen_session_continue_from_intermediate(state, position_ids)` |
| check | `tests/test_qwen_vlm.c` → `make phase7-check` |
