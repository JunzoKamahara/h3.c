#!/usr/bin/env python3
"""QINT-015i-b -- authoritative EAGLE-3 one-step parity: h3.c vs SpecForge.

Vendors the *exact* forward math from SpecForge `f7245ad`
`specforge/modeling/draft/llama3_eagle.py` (sdpa path only -- flash / flex
/ usp / distributed stripped, they do not change the math) and runs it on
the fixture written by `h3_qwen_spec_test eagle-b2-fixture`, then compares
stage by stage against the h3.c trace from `h3_qwen_eagle3_test dump`.

This is an INDEPENDENT author's implementation -- the training-time one
mattbucci used -- so it catches a shared blind-spot the same-author numpy
reference (1c / 2a) could not.

usage: eagle3_specforge_parity.py <fixture.json> <c_trace.json> <ckpt_dir>
"""
import sys
import json
import math

import numpy as np
import torch

torch.compile = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
torch.set_grad_enabled(False)

FIX, CTRACE, CKPT = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- vendored from SpecForge f7245ad llama3_eagle.py -----------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    b, kvh, s, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kvh, n_rep, s, hd)
    return hidden_states.reshape(b, kvh * n_rep, s, hd)


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings + 20,
                                torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)

    def forward(self, x, seq_len=None):
        return (self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(x.dtype))


# ---- fixture + config ----------------------------------------------------

fx = json.load(open(FIX))
H = fx["hidden_size"]
T = fx["num_tokens"]
tok_ids = fx["token_ids"]
positions = torch.tensor(fx["positions"], dtype=torch.long).view(1, -1)  # [1,T]


def vec(name, n):
    return torch.tensor(fx[name], dtype=torch.float32).reshape(T, n)


aux = torch.stack([vec("aux_hidden_low", H), vec("aux_hidden_mid", H),
                   vec("aux_hidden_high", H)], dim=0)     # [3, T, H]
inputs_embeds = vec("embedding", H)[None]                 # [1, T, H]
cat3 = torch.cat([aux[0], aux[1], aux[2]], dim=-1)[None]  # [1, T, 3H]

cfg = json.load(open(f"{CKPT}/config.json"))
n_heads = cfg["num_attention_heads"]
n_kv = cfg["num_key_value_heads"]
head_dim = cfg["head_dim"]
eps = cfg["rms_norm_eps"]
theta = cfg["rope_parameters"]["rope_theta"]
q_dim = n_heads * head_dim
kv_dim = n_kv * head_dim

# ---- load checkpoint weights (as float32 -- matches h3.c) ---------------
from safetensors import safe_open  # noqa: E402

W = {}
with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()


def lin(x, wname):
    return torch.nn.functional.linear(x, W[wname])


in_ln = LlamaRMSNorm(H, eps)
in_ln.weight.data = W["midlayer.input_layernorm.weight"]
hid_ln = LlamaRMSNorm(H, eps)
hid_ln.weight.data = W["midlayer.hidden_norm.weight"]
post_ln = LlamaRMSNorm(H, eps)
post_ln.weight.data = W["midlayer.post_attention_layernorm.weight"]
fin_ln = LlamaRMSNorm(H, eps)
fin_ln.weight.data = W["norm.weight"]
rotary = LlamaRotaryEmbedding(head_dim, cfg["max_position_embeddings"], theta)

# ---- forward, capturing every stage (LlamaForCausalLMEagle3 + midlayer) --
S = {}
fc_out = lin(cat3, "fc.weight")                       # [1,T,H]
S["fc_out"] = fc_out[0]
residual = fc_out
hs = hid_ln(fc_out)
S["hidden_normed"] = hs[0]
ie = in_ln(inputs_embeds)
S["embed_norm"] = ie[0]
x = torch.cat((ie, hs), dim=-1)                       # [1,T,2H]  [emb, hidden]
S["qkv_in"] = x[0]

q = lin(x, "midlayer.self_attn.q_proj.weight")        # [1,T,q_dim]
k = lin(x, "midlayer.self_attn.k_proj.weight")
v = lin(x, "midlayer.self_attn.v_proj.weight")
S["q_pre_rope"] = q[0]
S["k_pre_rope"] = k[0]
S["v"] = v[0]

q = q.view(1, T, n_heads, head_dim).transpose(1, 2)   # [1,nh,T,hd]
k = k.view(1, T, n_kv, head_dim).transpose(1, 2)
v = v.view(1, T, n_kv, head_dim).transpose(1, 2)
cos, sin = rotary(q, seq_len=T)
q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
S["q_post_rope"] = q.transpose(1, 2).reshape(T, q_dim)
S["k_post_rope"] = k.transpose(1, 2).reshape(T, kv_dim)

k = repeat_kv(k, n_heads // n_kv)
v = repeat_kv(v, n_heads // n_kv)
mask = torch.full((T, T), float("-inf")).triu(1)[None, None]  # causal float mask
attn = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, dropout_p=0.0)           # [1,nh,T,hd]
attn = attn.transpose(1, 2).reshape(1, T, q_dim)
S["attn_heads"] = attn[0]
attn = lin(attn, "midlayer.self_attn.o_proj.weight")
S["attn_out"] = attn[0]

hs = residual + attn
residual = hs
y = post_ln(hs)
S["post_attn_norm"] = y[0]
g = lin(y, "midlayer.mlp.gate_proj.weight")
u = lin(y, "midlayer.mlp.up_proj.weight")
mlp = lin(torch.nn.functional.silu(g) * u, "midlayer.mlp.down_proj.weight")
S["mlp_out"] = mlp[0]
hs = residual + mlp
final = fin_ln(hs)
S["final_hidden"] = final[0]
logits = lin(final, "lm_head.weight")[0]              # [T, draft_vocab]

d2t = W["d2t"].long()

# ---- compare stage by stage vs the h3.c trace --------------------------
ct = json.load(open(CTRACE))
STAGES = ["fc_out", "embed_norm", "hidden_normed", "qkv_in", "q_pre_rope",
          "k_pre_rope", "v", "q_post_rope", "k_post_rope", "attn_heads",
          "attn_out", "post_attn_norm", "mlp_out", "final_hidden"]


def stats(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na and nb else 1.0
    rel = float(np.linalg.norm(a - b) / na) if na else 0.0
    return cos, rel, float(np.max(np.abs(a - b)))


print(f"T={T}  n_heads={n_heads}/{n_kv}  head_dim={head_dim}  eps={eps}  "
      f"theta={theta}")
print(f"\n{'stage':>16}  {'worst cos':>13} {'worst relL2':>12} "
      f"{'worst max|d|':>12}")
worst_overall = 1.0
for st in STAGES:
    cs = []
    for t in range(T):
        cref = np.array(ct[f"t{t}_{st}"], dtype=np.float64)
        cs.append(stats(S[st][t].numpy(), cref))
    cmin = min(c for c, _, _ in cs)
    rmax = max(r for _, r, _ in cs)
    mmax = max(m for _, _, m in cs)
    worst_overall = min(worst_overall, cmin)
    flag = "" if cmin > 0.9999 else ("  <-- DIVERGES" if cmin < 0.99 else "  <- soft")
    print(f"{st:>16}  {cmin:>13.9f} {rmax:>12.3e} {mmax:>12.3e}{flag}")

# draft top-1 / d2t agreement
c_top1 = [ct[f"t{t}_draft_top1"] for t in range(T)]
py_top1 = logits.argmax(-1).tolist()
agree = sum(int(a == b) for a, b in zip(c_top1, py_top1))
print(f"\ndraft top-1 agreement (h3.c vs SpecForge): {agree}/{T}")
tgt_c = [ct[f"t{t}_target_top1"] for t in range(T)]
tgt_py = [int(d2t[i]) + i for i in py_top1]
agree_t = sum(int(a == b) for a, b in zip(tgt_c, tgt_py))
print(f"post-d2t target-id agreement: {agree_t}/{T}")
mism = [(t, c_top1[t], py_top1[t]) for t in range(T) if c_top1[t] != py_top1[t]]
if mism:
    print(f"  top-1 mismatches (pos, h3, sf): {mism[:12]}")
print(f"\nWORST stage cosine overall: {worst_overall:.9f}")
