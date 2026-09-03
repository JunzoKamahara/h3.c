#!/usr/bin/env python3
"""QINT-015i-c (②-a) -- teacher-forced 1-step accuracy of the PUBLIC EAGLE-3
head under the authoritative SpecForge f7245ad forward, for one or two
target-hidden fixtures (BF16 vs AWQ).

The forward math is vendored verbatim from SpecForge
`specforge/modeling/draft/llama3_eagle.py` (sdpa path). Fixtures come from
`h3_qwen_spec_test eagle-i-fixture` (BF16 / h3.c) and
`scripts/awq_target_hidden.py` (AWQ / CUDA) -- SAME token_ids, positions,
expected_next; only the aux hiddens (and embeddings) differ.

Decisive question: does feeding AWQ-teacher hiddens recover the head's
1-step accuracy (BF16 ~9%)?  AWQ high => the head is adapted to the AWQ
distribution.  AWQ also low => suspect the checkpoint itself.

usage:
  eagle3_specforge_accuracy.py <ckpt_dir> <fixture_a.json> [<fixture_b.json>]
"""
import sys
import json

import numpy as np
import torch

torch.compile = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
torch.set_grad_enabled(False)

CKPT = sys.argv[1]
FIXES = sys.argv[2:]
assert 1 <= len(FIXES) <= 2, __doc__


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(h, n):
    b, kvh, s, hd = h.shape
    if n == 1:
        return h
    return h[:, :, None].expand(b, kvh, n, s, hd).reshape(b, kvh * n, s, hd)


def rmsnorm(x, w, eps):
    d = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return w * x.to(d)


cfg = json.load(open(f"{CKPT}/config.json"))
H = cfg["hidden_size"]
NH, NKV, HD = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
              cfg["head_dim"])
EPS = cfg["rms_norm_eps"]
THETA = cfg["rope_parameters"]["rope_theta"]
QD, KVD = NH * HD, NKV * HD

from safetensors import safe_open  # noqa: E402

W = {}
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()
d2t = W["d2t"].long()

inv_freq = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
_t = torch.arange(cfg["max_position_embeddings"] + 20, dtype=torch.float32)
_emb = torch.cat((torch.outer(_t, inv_freq),) * 2, dim=-1)
COS, SIN = _emb.cos()[None, None], _emb.sin()[None, None]


def lin(x, name):
    return torch.nn.functional.linear(x, W[name])


def eagle_forward(cat3, inputs_embeds, positions):
    """SpecForge LlamaForCausalLMEagle3.forward + midlayer, sdpa path."""
    T = cat3.shape[1]
    fc_out = lin(cat3, "fc.weight")
    residual = fc_out
    hs = rmsnorm(fc_out, W["midlayer.hidden_norm.weight"], EPS)
    ie = rmsnorm(inputs_embeds, W["midlayer.input_layernorm.weight"], EPS)
    x = torch.cat((ie, hs), dim=-1)
    q = lin(x, "midlayer.self_attn.q_proj.weight").view(1, T, NH, HD).transpose(1, 2)
    k = lin(x, "midlayer.self_attn.k_proj.weight").view(1, T, NKV, HD).transpose(1, 2)
    v = lin(x, "midlayer.self_attn.v_proj.weight").view(1, T, NKV, HD).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, COS[:, :, :T], SIN[:, :, :T], positions)
    k, v = repeat_kv(k, NH // NKV), repeat_kv(v, NH // NKV)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attn = attn.transpose(1, 2).reshape(1, T, QD)
    attn = lin(attn, "midlayer.self_attn.o_proj.weight")
    hs = residual + attn
    residual = hs
    y = rmsnorm(hs, W["midlayer.post_attention_layernorm.weight"], EPS)
    g = lin(y, "midlayer.mlp.gate_proj.weight")
    u = lin(y, "midlayer.mlp.up_proj.weight")
    hs = residual + lin(torch.nn.functional.silu(g) * u,
                        "midlayer.mlp.down_proj.weight")
    final = rmsnorm(hs, W["norm.weight"], EPS)
    return lin(final, "lm_head.weight")[0]  # [T, draft_vocab]


def load_fixture(path):
    fx = json.load(open(path))
    T = fx["num_tokens"]
    def vec(n):
        return torch.tensor(fx[n], dtype=torch.float32).reshape(T, H)
    cat3 = torch.cat([vec("aux_hidden_low"), vec("aux_hidden_mid"),
                      vec("aux_hidden_high")], dim=-1)[None]
    ie = vec("embedding")[None]
    pos = torch.tensor(fx["positions"], dtype=torch.long).view(1, -1)
    exp = fx["expected_next"]
    sf = fx.get("score_from", 0)
    return fx, T, cat3, ie, pos, exp, sf


results = {}
for path in FIXES:
    fx, T, cat3, ie, pos, exp, sf = load_fixture(path)
    logits = eagle_forward(cat3, ie, pos)
    top1 = logits.argmax(-1).tolist()
    tgt = [int(d2t[i]) + i for i in top1]
    scored = [t for t in range(T) if t >= sf and t + 2 <= T + 1 and exp[t] != 0]
    hit = sum(int(tgt[t] == exp[t]) for t in scored)
    acc = hit / len(scored) if scored else 0.0
    results[path] = dict(fx=fx, tgt=tgt, acc=acc, hit=hit, n=len(scored))
    print(f"{path}")
    print(f"  source={fx.get('source')}  T={T}  scored={len(scored)}  "
          f"teacher-forced 1-step accuracy = {acc:.3f}  ({hit}/{len(scored)})")

if len(FIXES) == 2:
    a, b = (results[p]["fx"] for p in FIXES)
    print("\nper-position target-hidden drift  A -> B  "
          f"({FIXES[0]} vs {FIXES[1]})")
    T = a["num_tokens"]
    for slot in ("aux_hidden_low", "aux_hidden_mid", "aux_hidden_high"):
        A = np.array(a[slot], dtype=np.float64).reshape(T, H)
        B = np.array(b[slot], dtype=np.float64).reshape(T, H)
        cs = []
        for t in range(T):
            na, nb = np.linalg.norm(A[t]), np.linalg.norm(B[t])
            cos = float(A[t] @ B[t] / (na * nb)) if na and nb else 1.0
            rel = float(np.linalg.norm(A[t] - B[t]) / na) if na else 0.0
            cs.append((cos, rel, float(np.max(np.abs(A[t] - B[t])))))
        print(f"  {slot:>16}: worst cos {min(c for c,_,_ in cs):.6f}  "
              f"worst relL2 {max(r for _,r,_ in cs):.3e}  "
              f"worst max|d| {max(m for _,_,m in cs):.3e}")
    print(f"\n  1-step accuracy:  {FIXES[0]} = {results[FIXES[0]]['acc']:.3f}   "
          f"{FIXES[1]} = {results[FIXES[1]]['acc']:.3f}")
