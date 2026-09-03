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

**Not yet run against a real EAGLE-3 checkpoint** — needs
`mattbucci/Qwen3-VL-32B-AWQ-EAGLE3` (primary candidate) and the 8B negatives
(`AQ-MedAI/Qwen3-VL-8B-Instruct-eagle3`, `taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3`,
expected INCOMPATIBLE on `hidden_size 4096 != 5120`) fetched locally. The
probe is ready; point it at the directory.
