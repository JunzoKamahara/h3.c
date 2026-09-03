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
