#!/usr/bin/env python3
"""QINT-015i-a -- target decoder-hidden parity: h3.c vs Transformers.

Reads the binary dump written by `h3_qwen_spec_test eagle-target-dump`
(raw token ids + h3.c's greedy argmax + the decoder-layer OUTPUT hiddens
at the EAGLE aux layers), runs an independent Transformers Qwen3-VL-32B
BF16 forward on the identical ids, and compares the layer outputs
position by position (cosine / relL2 / max|diff|).

The point is an INDEPENDENT implementation: if these hiddens agree, the
target runtime is exonerated and any EAGLE acceptance problem is on the
draft side. If they disagree, fix the target capture before anything else
-- training a drafter on wrong hiddens would bake in the error.

usage: target_hidden_parity.py <dump.bin> [<model_dir>]
  model_dir defaults to the MiniMax-H3 text tower, which is byte-identical
  to official Qwen/Qwen3-VL-32B-Instruct (verified in cfddd6a).
"""
import sys
import struct
import numpy as np
import torch

DUMP = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else (
    "/Users/kamahara/models/MiniMax-H3/FL2VA/text_encoder")

with open(DUMP, "rb") as f:
    assert f.read(4) == b"H3TD", "bad magic"
    n, nl, hid = struct.unpack("<III", f.read(12))
    lids = list(struct.unpack("<3i", f.read(12)))
    ids = np.frombuffer(f.read(4 * n), dtype=np.uint32).astype(np.int64)
    argmax = np.frombuffer(f.read(4 * n), dtype=np.uint32).astype(np.int64)
    aux = np.frombuffer(
        f.read(4 * 3 * n * hid), dtype=np.float32).reshape(3, n, hid)

print(f"dump: {n} tokens, aux layers {lids}, hidden {hid}")
print(f"first ids: {ids[:12].tolist()}")

from transformers import AutoModelForImageTextToText, AutoConfig  # noqa: E402

cfg = AutoConfig.from_pretrained(MODEL)
print(f"model_type={cfg.model_type}  arch={cfg.architectures}")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
model.eval()

input_ids = torch.tensor(ids[None, :], dtype=torch.long)
with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

hs = out.hidden_states  # len = num_layers + 1; hs[0] = embeddings,
#                          hs[k] = OUTPUT of decoder layer (k-1)
print(f"hidden_states: {len(hs)} tensors, shape {tuple(hs[0].shape)}")
ref = {L: hs[L + 1][0].float().numpy() for L in lids}

ref_logits = out.logits[0].float().numpy()
ref_argmax = ref_logits.argmax(-1)


def stats(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na and nb else 0.0
    rel = float(np.linalg.norm(a - b) / na) if na else 0.0
    mx = float(np.max(np.abs(a - b)))
    return cos, rel, mx


print(f"\n{'layer':>6} {'pos':>4} {'cos':>13} {'relL2':>11} {'max|d|':>11}")
for si, L in enumerate(lids):
    r = ref[L]
    per = [stats(aux[si, t], r[t]) for t in range(n)]
    show = sorted(set(list(range(min(4, n))) +
                      [int(np.argmax([p[1] for p in per]))] + [n - 1]))
    for t in show:
        c, rl, mx = per[t]
        print(f"{L:>6} {t:>4} {c:>13.9f} {rl:>11.3e} {mx:>11.3e}")
    cmin = min(p[0] for p in per)
    rmax = max(p[1] for p in per)
    mmax = max(p[2] for p in per)
    print(f"  layer {L}: worst cos {cmin:.9f}  worst relL2 {rmax:.3e}  "
          f"worst max|d| {mmax:.3e}")

# greedy argmax cross-check (prediction of token t+1 from tokens 0..t)
agree = int((argmax[:n - 1] == ref_argmax[:n - 1]).sum())
print(f"\ngreedy argmax agreement h3.c vs transformers: {agree}/{n - 1}")
mism = [(t, int(argmax[t]), int(ref_argmax[t]))
        for t in range(n - 1) if argmax[t] != ref_argmax[t]]
if mism:
    print(f"  mismatches (pos, h3, ref): {mism[:10]}")
