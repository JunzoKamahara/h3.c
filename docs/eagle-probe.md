# QINT-015h-1a — EAGLE-3 checkpoint compatibility probe

`qwen_eagle_probe.{c,h}` + `tests/probe_qwen_eagle.c` → `h3_qwen_eagle_probe`.

A **static** check: given a candidate draft-head checkpoint directory, read only
`config.json` and the `*.safetensors` header(s) (the JSON tensor map — never
any tensor bytes) and decide whether the checkpoint can drive speculative
decoding for the current Qwen3-VL-32B target. Nothing is allocated per weight,
nothing touches the GPU, no network. The verdict gates whether QINT-015h-1b
(the real tensor loader) is worth building for that checkpoint. **Performance
is not considered here** — that is QINT-015i.

## Use

```
h3_qwen_eagle_probe <checkpoint_dir> [<target_text_config.json>]
h3_qwen_eagle_probe --selftest          # make spec-eagle-probe-check
```

Exit code = verdict: `0` COMPATIBLE, `1` INCOMPATIBLE, `2` PROBE_ERROR
(checkpoint unreadable — not a verdict). With no target config the built-in H3
backbone constants are used (hidden 5120, vocab 151936, 64/8 heads, head_dim
128, intermediate 25600, rope_theta 5e6, mrope `[24,20,20]` interleaved).

## What it inspects

From `config.json` (trying the outer object and nested `model` /
`draft_model` / `eagle_config`, with alternate key spellings):
`architectures[0]`, `model_type`, `hidden_size`, `vocab_size` /
`draft_vocab_size`, `num_hidden_layers`, head counts + `head_dim`,
`intermediate_size`, `rope_theta`, `rope_scaling.mrope_section` /
`mrope_interleaved`, `torch_dtype`, `quantization_config.quant_method`, and any
declared aux-hidden count.

From the safetensors header(s): every tensor's name, dtype and declared shape.
Derived — `fc.weight` shape → fusion output dim and how many `hidden_size`
blocks it fuses; `midlayer.*` / `layers.N.*` → draft decoder depth; `d2t` /
`t2d` presence and length; `lm_head.weight` shape; `embed_tokens.weight`
presence; whether the drafter's own weights are GPTQ/AWQ-packed
(`*.qweight` / `*.scales` / `*.qzeros` / `*.g_idx`, or integer-typed proj
weights).

## Verdict rules

INCOMPATIBLE if **any** of (all listed, not just the first):

- not an EAGLE-3 draft head at all — no `fc` fusion tensor, no `eagle` in the
  architecture, and not a 1–2-layer head with a vocab map (catches "this is
  the full target model");
- `hidden_size` (config or `fc.weight` rows) ≠ target hidden;
- `fc.weight` fusion input not a whole multiple of the target hidden;
- config's target `vocab_size` ≠ target vocab;
- reduced draft vocab (`draft_vocab_size` ≠ target, or `lm_head` rows ≠ target
  vocab) but `d2t` / `t2d` mapping tensors absent;
- the drafter's own weights are quantized (the 015h-1b C loader is bf16/fp16
  only);
- safetensors header unreadable.

`rope_theta` and `mrope_section` mismatches are **reported, not gated** — the
loader reproduces whatever the checkpoint was trained with. Head counts on the
draft side are informational (EAGLE-3's single decoder layer is self-contained).

COMPATIBLE ⇒ 015h-1b may be built. INCOMPATIBLE ⇒ reselect the checkpoint.

## Status

`make spec-eagle-probe-check` (model-free, 5 synthetic miniature checkpoints):
compatible EAGLE-3 → COMPATIBLE; hidden mismatch / missing d2t·t2d / AWQ-packed
drafter → INCOMPATIBLE with the right reasons; missing `config.json` →
PROBE_ERROR. Smoke: the real 32B target `text_encoder/` (1058 tensors across 14
shards, headers read in ~10 ms) → INCOMPATIBLE "not an EAGLE-3 draft head".

Real run: `h3_qwen_eagle_probe ~/models/mattbucci-eagle3
~/models/MiniMax-H3/text_encoder/config.json` → **COMPATIBLE**. `hidden_size
5120` matches; `fc.weight [5120, 15360]` fuses 3 × 5120; drafter weights are
BF16 / non-quantized (the "AWQ" in the name is the *target* it was trained
against, not the drafter); `d2t [32000]` + `t2d [151936]` both present; 15
tensors. The 8B negatives (`AQ-MedAI/…-eagle3`,
`taobao-mnn/…-Eagle3`, `hidden_size 4096`) are still worth running as the
`4096 != 5120` negative test when fetched.

---

# QINT-015h-1b — loader

`qwen_eagle3.{c,h}` + `tests/test_qwen_eagle3.c` → `h3_qwen_eagle3_test`
(`make spec-eagle3-load-check`).

Loads `<dir>/config.json` + `model.safetensors` into the C runtime, validating
every tensor name / dtype / shape, and exposes a **CPU reference forward**.
Not wired to the coordinator; not a parity claim (that is 1c). Weights are
converted bf16 → f32 at load (~3.3 GB resident for the mattbucci checkpoint);
`matvec` is a plain double-accumulate loop — the fast path is 015h-2.

## Confirmed for `mattbucci/Qwen3-VL-32B-AWQ-EAGLE3` (real load)

| field | value | note |
|---|---|---|
| architecture / model_type | `LlamaForCausalLMEagle3` / `llama` | plain 1-D RoPE, **not** mRoPE |
| hidden_size | 5120 | matches target |
| head_dim | **128 (explicit)** | `q_dim = 32·128 = 4096 ≠ hidden` — never derive it |
| q/k/v input width | **10240 = 2·hidden** | EAGLE-3 norms the embedding and the fused hidden separately, then concatenates |
| fusion (`fc.weight`) | `[5120, 15360]` = fuse 3 × 5120 → 5120, no bias | |
| intermediate_size | 32768 | SwiGLU MLP on the 5120 residual |
| rms_norm_eps | **1e-5** | the draft's own — target uses 1e-6 |
| rope_theta | 5e6 | `rope_type: "default"` |
| draft_vocab / target_vocab | 32000 / 151936 | |
| `d2t` (i64 [32000]) | **delta**: `target = draft + d2t[draft]` | proven — `{draft + d2t[draft]}` equals exactly the 32000 target ids whose `t2d` bit is set; `d2t[0..~thousands] = 0` (common tokens keep their id), last entry → 151645 (EOS) |
| `t2d` (bool [151936]) | mask, 32000 set | |
| biases | none (`attention_bias`/`mlp_bias` false, no `*.bias`) | |

## Reference forward structure (per EAGLE-3 / SpecForge — 1c verifies)

```
aux hidden low/mid/high  →  concat [15360]  →  fc  →  fused [5120]
prev token  →  target embed (SHARED, never copied)  →  emb [5120]
x = concat( RMSNorm(emb, input_layernorm),
            RMSNorm(fused, hidden_norm) )              [10240]
q,k,v = q/k/v_proj(x)   →  RoPE(theta 5e6)  →  GQA (32/8 heads, hd 128)
attn = o_proj(attn_heads)
res  = fused + attn                     (residual is the FUSED hidden)
res += down_proj( silu(gate_proj(RMSNorm(res, post_attention_layernorm)))
                  * up_proj(...) )
draft_logits = lm_head( RMSNorm(res, norm) )           [32000]
draft_target = draft_id + d2t[draft_id]
```

The single-position reference collapses self-attention (softmax over one key);
multi-token draft chains with a real KV cache are 015h-2.

## Status

`spec-eagle3-load-check` (model-free, miniature EAGLE-3 in a temp dir): loads,
config fields as declared, `d2t` maps in range, reference forward → finite
logits; rejects an unknown tensor / a missing required tensor / a 2-layer
config, each with a reason. Real load of `~/models/mattbucci-eagle3`: all 15
tensors validated, forward → finite logits, `d2t`/`t2d` consistent on a sample,
~0.7 s including the 3.3 GB f32 conversion.

**Open for 1c** (forward parity vs Python/SGLang, staged: fc output → decoder
hidden → draft logits top-k): the norm↔stream pairing and concat order
(`[norm(emb), norm(fused)]` assumed), whether RoPE positions are absolute
target positions or draft-relative, and the exact aux-hidden layer ids the
reference captures.

---

# QINT-015h-1c — staged forward parity (PASSES; see the -2a section for the current schema)

Loader is done; the risk is now the EAGLE-3 forward *semantics* (norm↔stream
pairing, concat order, RoPE position convention, which target layers feed the
3 aux hidden states). A 1-token step has a degenerate softmax, so the final
logits alone cannot verify the Q/K projections or RoPE — the pre/post-RoPE
Q and K are compared directly.

## Pieces

- **Deterministic fixture** — `h3_qwen_eagle3_test gen-fixture <out.json>
  [hidden] [token] [position]`. Self-contained: 3 aux hidden rows + the token's
  target embedding + token id + a non-zero position (default 37), all from a
  seeded splitmix64. The baked-in embedding means a Python reference never
  loads the 32B target.
- **C staged trace** — `h3_qwen_eagle3_test dump <ckpt> <fixture.json>
  <c_trace.json>`. Runs the reference forward and dumps every stage:
  `aux_concat → fc_out → embed_norm → hidden_normed → qkv_in →
  q_pre_rope / k_pre_rope / v → q_post_rope / k_post_rope → attn_heads →
  attn_out → post_attn_norm → mlp_out → final_hidden → draft_logits`, plus
  `draft_top1 / draft_top5 / target_top1`.
- **Reference trace** — `scripts/eagle3_reference.py <ckpt> <fixture.json>
  <ref_trace.json>` (numpy, float32). A *self-contained* reference; the
  authority is SGLang `llama_eagle3.py` / SpecForge
  `eagle3/model.py`. Each step is annotated with its assumption so it can be
  checked against that source, or replace it by running SpecForge/SGLang's
  `LlamaForCausalLMEagle3` with forward hooks that dump the same JSON keys.
- **Compare** — `scripts/eagle3_compare.py <c_trace.json> <ref_trace.json>`:
  per-stage cosine / max|diff| / relative-L2, and the gate.

`make spec-eagle3-1c-trace` runs the two C steps and prints the two Python
commands. `make EAGLE_CKPT=/path/to/other-eagle3 spec-eagle3-1c-trace` for a
different checkpoint.

## Gate (initial — tighten once the observed error is known)

| stage | gate |
|---|---|
| `fc_out`, `*_norm`, `qkv_in`, `q_*`, `k_*`, `v`, `*_hidden`, `draft_logits` | cosine ≥ 0.99999 |
| `draft_top1` | exact |
| `draft_top5` | ≥ 4 / 5 common |
| `target_top1` (draft_top1 after d2t) | exact |

The Python side widens the bf16 weights to float32 and computes in float32, to
line up with the C CPU reference (double-accumulate matvec).

## Status — staged parity PASSES

`h3_qwen_eagle3_test dump ~/models/mattbucci-eagle3` vs
`scripts/eagle3_reference.py` (numpy float32), fixture token 1234 / position 37:

| | result |
|---|---|
| all 16 stages (`aux_concat` … `draft_logits`) | **cosine 1.00000000**, max\|diff\| 5e-10 – 4e-6, relL2 ≤ 4e-7 |
| `q_pre_rope` / `q_post_rope` / `k_pre_rope` / `k_post_rope` | cosine 1.0 — RoPE, the 2·hidden concat, and the Q/K projections are right |
| `draft_top1` / `draft_top5` / `target_top1` | exact / 5-of-5 / exact (624 → 624) |

So the C reference forward and an independent float32 reimplementation agree to
float-rounding across every stage: fusion (concat low→mid→high, no bias),
`input_layernorm`↔embedding and `hidden_norm`↔fused with concat
`[norm(emb), norm(fused)]`, RoPE (plain 1-D, θ 5e6, HF rotate-half, absolute
position), GQA 32/8, residual on the **fused** hidden, SwiGLU MLP, final norm,
`lm_head`, and the `d2t` delta.

**Still open** (neither impl can settle these alone, and both are the same
author's reading of the EAGLE-3 spec):

1. A cross-check against SGLang / SpecForge `LlamaForCausalLMEagle3` for a
   *shared* blind spot — run it with forward hooks dumping the same JSON keys
   and re-run `eagle3_compare.py`.
2. **Which 3 target decoder layers feed `aux_hidden_low/mid/high`.** The
   fixture uses random aux, so 1c parity holds regardless; the ids matter when
   `qwen_session_aux_hidden` (015h-0) is wired to the EAGLE input in 015h-2.
   SpecForge's EAGLE-3 default is roughly `{low ≈ 1, mid ≈ N/2, high ≈ N−4}`;
   confirm from the checkpoint's training config.
3. A 2-token fixture so RoPE / GQA / the causal mask / attention scaling go
   through a real (non-degenerate) softmax — best done in 015h-2 where the
   multi-token forward is built.

By the 1c gate (Q/K pre/post-RoPE parity + top-1/top-5 + cos ≈ 1), the forward
semantics are validated enough to build 015h-2.

---

# QINT-015h-2a — multi-token / KV-cache correctness

1c validated the forward math with a degenerate 1-key softmax. 2a exercises a
real causal softmax, GQA, `1/sqrt(head_dim)` scaling, RoPE at consecutive
positions, and the K/V cache.

- `qwen_eagle3_forward_seq(e, T, aux[T·3], tokens, positions, ...)` — one-shot
  causal forward, query i attends key/value rows 0..i.
- `qwen_eagle3_kv_new` / `_kv_step` / `_kv_free` — incremental variant; each
  `_kv_step` appends this token's K/V and attends over all appended positions.
- `qwen_eagle3_step_ref` (1c) is now `forward_seq` with T=1.
- The fixture schema gained `num_tokens` / `token_ids` / `positions`; arrays are
  flat `[T·hidden]`. `h3_qwen_eagle3_test dump` runs **both** paths and
  requires the step-wise `_kv_step` logits to equal the `forward_seq` logits
  per token (cosine ≥ 0.9999999), then writes per-token `t{i}_<stage>` keys.
- `make spec-eagle3-parity-trace` builds a 1-token and a 2-token fixture +
  C traces.

## Status — 2a PASSES

`h3_qwen_eagle3_test dump ~/models/mattbucci-eagle3 <2-token fixture>`:
**step-wise KV vs batch, worst per-token draft-logit cosine = 1.0000000000.**

vs `scripts/eagle3_reference.py` (numpy float32, causal), 2 tokens
(positions 37, 38):

| | result |
|---|---|
| every stage, both tokens | cosine 1.00000000, max\|diff\| ≤ 3e-6 |
| token 1 `attn_heads` / `attn_out` (real 2-key softmax) | cosine 1.0 |
| `draft_top1` / `draft_top5` / `target_top1`, both tokens | exact / 5-of-5 / exact |

So the KV cache layout, causal masking, GQA grouping, attention scaling, and
RoPE at consecutive positions are all correct, and the incremental and
one-shot paths agree. Ready for 015h-2b (the autoregressive `qwen_draft_eagle`
backend).

The aux-layer convention is pinned: `qwen_session_set_aux_layers` captures the
**output** of each named decoder layer (`h_k_out == h_{k+1}_in`, so no +1
shift), and the SpecForge EAGLE-3 default for a 64-layer target is
`{1, 32, 60}` — `QWEN_EAGLE3_AUX_LAYERS_DEFAULT` /
`qwen_eagle3_default_aux_layers()`. Confirm against real acceptance in 015i.

---

# QINT-015h-2b — the `qwen_draft_eagle` backend (CPU, 2b-0)

`qwen_eagle3_chain()` + `qwen_draft_eagle.c` (`qwen_draft_eagle_new`, a
`qwen_draft_backend`).

The autoregressive draft chain:

- **step 0** fuses the 3 aux hidden (the target residual at the frontier
  position) with `Emb(anchor_token)` at `position = history_length` — the
  `hidden(t) + Emb(token t+1)` one-token shift; the anchor is the token the
  target has decided comes next, not the last committed one.
- **step j > 0** feeds EAGLE's own previous output hidden (recurrent — the 3
  aux are **not** re-fused) with `Emb(previous draft token, mapped through
  d2t)` at `position = history_length + j`.
- each step's K/V is appended to the single decoder layer's cache; step j's
  query attends rows 0..j.
- returns `k = width-1` draft tokens: draft-vocab argmax → `d2t` → target vocab.

`qwen_draft_eagle_new(dir, embed, embed_ctx)` loads the checkpoint and wires
`propose()` to read `qwen_draft_context.aux_hidden` (n_aux == 3, bf16) +
`anchor_token` + `history_length`. `n_aux < 3` or a hidden-size mismatch →
`count 0` (the coordinator falls back to a scalar step). `embed` is the
target's token-embedding accessor; the checkpoint has no embed table.

**2b-0 scope:** a fresh per-cycle draft KV (no draft prefill over the
committed context) and no post-verify catch-up. Both are 2b-2.

## Status

`spec-eagle3-load-check` (`--selftest`, model-free) now also asserts:
step 0 embeds the anchor and step 1 embeds `d2t(step-0 draft argmax)` (the
one-token-shift recurrence); the chain is deterministic; the backend
`propose()` returns `k` target-vocab tokens, and `count 0` when `n_aux == 0`.

`make spec-eagle3-chain-smoke` — one real chain on
`~/models/mattbucci-eagle3` with a random-aux fixture: deterministic, 4 draft
tokens, **T_draft ≈ 380 ms/step** on the plain double-accumulate `matvec`
(~1.6 GB f32 weights per step). That is the CPU-reference cost — 015i
measures whether it needs Metal; the 2b goal is semantics, not speed.

## 2b-1 — live wiring (done)

`qwen_session_embedding_row_f32(session, token_id, dst, dst_count)` (public;
`qwen_kv_embedding_row_f32` + `h3_gpu_tensor_read_bf16_range`): widens one
5120-value slice of the resident BF16 `embed_tokens.weight` into a
caller-owned f32 buffer -- bounds-checked, no GPU work, no allocation, no
full-table copy. The draft backend takes it as its `embed` accessor.

`make spec-eagle-live-check` (`test_qwen_spec.c eagle-live`): a live session
with `set_aux_layers({1,32,60})`, 5 real decode steps, then

```
L=27  anchor=279  aux_ids={1,32,60}  aux_row_source=26  eagle_pos=26  embed_token=279
proposal (target vocab) = 279 279 279 279  (== direct chain @ L-1)
```

Confirms: real `{1,32,60}` aux (captured as layer *outputs*), aux row at
`L-1`, anchor at `L`, EAGLE step 0 at `L-1`, step 0 embeds the anchor, the
embedding accessor rejects an out-of-range token and a wrong `dst_count`, the
backend is deterministic, and **`propose()` == a direct `qwen_eagle3_chain()`
at `start_pos = L-1`**.

The `279 279 279 279` proposal is degenerate **as expected**: a fresh
per-cycle draft KV has no prefix to attend, so the recurrent hidden barely
moves and the draft head repeats. Acceptance / τ are *not* evaluated here.
2b-2 adds the prefix draft KV; if the output is still degenerate then, it is a
real problem.

## Next

- **2b-2** — build a prefix draft KV (draft-extend over the committed context
  with the one-token shift `hidden(x[t]) + Emb(x[t+1])`), seed the recurrent
  hidden from its last output, and after each verify rewind/extend it to the
  accepted target prefix. **Then A/B the step-0 position L-1 vs L on real
  acceptance** (2a parity cannot see a shared off-by-one).
- **2b-3 / 015i** — hook into the coordinator and measure τ. τ ≈ 2–2.5 with
  T_draft large → semantics right, only speed remains → Metal. τ ≈ 1.1 →
  re-check aux ids / anchor alignment / positions / the recurrent-hidden
  handoff before optimising.
