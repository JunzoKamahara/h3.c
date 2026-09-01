# Qwen layer-49 intermediate state — exact semantics (P0-001 / P0-002)

This records what the released H3 conditioning tensor *is*, as implemented today
in `h3_text_encoder.c`, so the Phase 0 runtime boundary (`qwen_engine.h`) can be
held to it bit-for-bit.

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
- Decoder-layer weights: **streamed per eval by default** (~14 s/token), or
  **resident** with `qwen_session_set_resident()` / `H3_QWEN_RESIDENT=1` --
  all 64 layers pinned in Unified Memory (~62 GB), decode ~0.75 s/token,
  bit-for-bit identical (`make resident-check`). embed / final-norm / lm_head
  are always resident in the context. `h3_serve --resident` loads once and
  reuses one persistent session (rewound per request).

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
h3_serve --model MiniMax-H3 [--port 8080] [--host 127.0.0.1] [--resident]

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
