#!/usr/bin/env python3
"""QINT-015i-c (②-a, CUDA side) -- capture Qwen3-VL-32B *AWQ* decoder-layer
OUTPUT hiddens for the token ids in a BF16 fixture, so the public EAGLE-3
head can be scored on the hidden distribution it was actually trained on.

Runs on a CUDA box (24 GB is enough: seq ~40, batch 1, base transformer
only, no lm_head). Prefer the training-time stack -- Transformers + an AWQ
backend close to what SpecForge used; the goal is hidden reproducibility,
not speed.

Writes an AWQ fixture in the SAME schema as `h3_qwen_spec_test
eagle-i-fixture` (token_ids / positions / expected_next / score_from
copied verbatim; aux_hidden_{low,mid,high} and embedding from the AWQ
target). Then, on any machine:

  python scripts/eagle3_specforge_accuracy.py <eagle_ckpt> bf16_fix.json awq_fix.json

usage:
  python scripts/awq_target_hidden.py <bf16_fixture.json> <awq_fixture_out.json> \
      [--model mattbucci/Qwen3-VL-32B-AWQ] [--layers 1,31,60]
"""
import sys
import json
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("bf16_fixture")
ap.add_argument("out")
ap.add_argument("--model", default="mattbucci/Qwen3-VL-32B-AWQ")
ap.add_argument("--layers", default=None,
                help="override aux layers, e.g. 1,31,60 (default: from fixture)")
args = ap.parse_args()

fx = json.load(open(args.bf16_fixture))
H = fx["hidden_size"]
token_ids = fx["token_ids"]
T = fx["num_tokens"]
assert len(token_ids) == T
# `full_ids` is x[0..N-1] (N = T+1). EAGLE one-token shift: draft position t
# uses the target residual at seq position t (after x[0..t]) and embeds
# x[t+1]. So feed the target full_ids and read decoder-layer outputs at
# positions 0..T-1.
tok_seq = fx["full_ids"]
assert len(tok_seq) == T + 1, (len(tok_seq), T)

layers = ([int(v) for v in args.layers.split(",")] if args.layers
          else fx.get("aux_layers", [1, 31, 60]))

print(f"model={args.model}  seq_len={len(tok_seq)}  aux layers={layers}")

from transformers import AutoModelForImageTextToText, AutoConfig  # noqa: E402

cfg = AutoConfig.from_pretrained(args.model)
print(f"  model_type={cfg.model_type}  quant={getattr(cfg, 'quantization_config', None)}")
model = AutoModelForImageTextToText.from_pretrained(
    args.model, device_map="cuda", attn_implementation="eager")
model.eval()


def find_layers(m):
    for path in ("model.language_model.layers",
                 "model.model.language_model.layers",
                 "language_model.model.layers",
                 "model.model.layers"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                return obj, path
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


dec, path = find_layers(model)
print(f"  decoder layers at {path}  (n={len(dec)})")

captured = {}


def mk_hook(idx):
    def hook(_mod, _inp, out):
        captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


handles = [dec[L].register_forward_hook(mk_hook(L)) for L in layers]

ids = torch.tensor([tok_seq], device="cuda")
with torch.no_grad():
    model.model(input_ids=ids, use_cache=False)  # base transformer, no lm_head
for h in handles:
    h.remove()

# decoder-layer output at seq positions 0..T-1 -> aux for draft positions 0..T-1
aux = {L: captured[L][0, :T].float().cpu() for L in layers}
for L in layers:
    assert aux[L].shape == (T, H), (L, aux[L].shape)

# embeddings for token_ids from the AWQ model's own table
emb_mod = model.get_input_embeddings()
emb = emb_mod(torch.tensor([token_ids], device="cuda"))[0].float().cpu()

out = dict(fx)  # copy token_ids / positions / expected_next / score_from ...
out["source"] = f"awq:{args.model}"
out["hidden_size"] = H
out["num_tokens"] = T
out["aux_layers"] = list(layers)
out["aux_hidden_low"] = aux[layers[0]].flatten().tolist()
out["aux_hidden_mid"] = aux[layers[1]].flatten().tolist()
out["aux_hidden_high"] = aux[layers[2]].flatten().tolist()
out["embedding"] = emb.flatten().tolist()
json.dump(out, open(args.out, "w"))
print(f"wrote {args.out}")
