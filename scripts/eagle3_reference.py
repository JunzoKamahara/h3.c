#!/usr/bin/env python3
"""QINT-015h-1c / -2a -- EAGLE-3 draft-head reference forward (float32, staged,
multi-token causal).

Reads <checkpoint_dir>/config.json + model.safetensors and a fixture produced
by `h3_qwen_eagle3_test gen-fixture` (per token: 3 aux hidden rows + the token's
target embedding baked in, so the 32B target is never loaded). Runs a causal
EAGLE-3 forward over all `num_tokens` positions and writes ref_trace.json with
per-token stage keys `t{i}_<stage>`, matching the C dump.

This is a *self-contained* reference. The authority is SGLang
`python/sglang/srt/models/llama_eagle3.py` / SpecForge
`specforge/algorithms/eagle3/model.py`; every step is annotated with its
assumption so it can be checked against that source, or replaced by running
their `LlamaForCausalLMEagle3` with forward hooks dumping the same keys.

    python3 scripts/eagle3_reference.py <checkpoint_dir> <fixture.json> <ref_trace.json>

numpy only.
"""
import json
import struct
import sys

import numpy as np


def load_safetensors_f32(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            b0, b1 = meta["data_offsets"]
            f.seek(base + b0)
            raw = f.read(b1 - b0)
            dt, shape = meta["dtype"], meta["shape"]
            if dt == "BF16":
                arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, "<f2").astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, "<f4")
            elif dt in ("I64", "U64"):
                arr = np.frombuffer(raw, "<i8")
            elif dt in ("I32", "U32"):
                arr = np.frombuffer(raw, "<i4")
            elif dt in ("BOOL", "I8", "U8"):
                arr = np.frombuffer(raw, "u1")
            else:
                raise SystemExit(f"unhandled dtype {dt} for {name}")
            out[name] = np.ascontiguousarray(arr.reshape(shape))
        return out


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    var = np.mean(x * x, dtype=np.float32)
    return (x * np.float32(1.0 / np.sqrt(var + eps)) * w.astype(np.float32)).astype(np.float32)


def rope(vec, n_heads, head_dim, position, theta):
    """HF 'default' rotary; matches the C reference's per-half rotation."""
    v = vec.reshape(n_heads, head_dim).astype(np.float32)
    half = head_dim // 2
    inv_freq = theta ** (-2.0 * np.arange(half, dtype=np.float64) / head_dim)
    ang = position * inv_freq
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = v[:, :half], v[:, half:]
    out = np.empty_like(v)
    out[:, :half] = a * cos - b * sin
    out[:, half:] = a * sin + b * cos
    return out.reshape(-1)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    ckpt, fixpath, outpath = sys.argv[1:4]
    cfg = json.load(open(f"{ckpt}/config.json"))
    W = load_safetensors_f32(f"{ckpt}/model.safetensors")
    fx = json.load(open(fixpath))

    H = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])            # explicit -- never derive H/n_heads
    group = n_heads // n_kv
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rp = cfg.get("rope_parameters", {})
    theta = float(rp.get("rope_theta", cfg.get("rope_theta", 5e6)))
    draft_vocab = int(cfg["draft_vocab_size"])
    d2t = W["d2t"].astype(np.int64)
    scale = np.float32(1.0 / np.sqrt(head_dim))

    assert int(fx["hidden_size"]) == H
    T = int(fx.get("num_tokens", 1))
    tokens = [int(x) for x in fx["token_ids"]]
    positions = [int(x) for x in fx["positions"]]
    low = np.asarray(fx["aux_hidden_low"], np.float32).reshape(T, H)
    mid = np.asarray(fx["aux_hidden_mid"], np.float32).reshape(T, H)
    high = np.asarray(fx["aux_hidden_high"], np.float32).reshape(T, H)
    emb_all = np.asarray(fx["embedding"], np.float32).reshape(T, H)

    fc_w = W["fc.weight"].astype(np.float32)
    q_w = W["midlayer.self_attn.q_proj.weight"].astype(np.float32)
    k_w = W["midlayer.self_attn.k_proj.weight"].astype(np.float32)
    v_w = W["midlayer.self_attn.v_proj.weight"].astype(np.float32)
    o_w = W["midlayer.self_attn.o_proj.weight"].astype(np.float32)
    gate_w = W["midlayer.mlp.gate_proj.weight"].astype(np.float32)
    up_w = W["midlayer.mlp.up_proj.weight"].astype(np.float32)
    down_w = W["midlayer.mlp.down_proj.weight"].astype(np.float32)
    lm_w = W["lm_head.weight"].astype(np.float32)
    in_ln = W["midlayer.input_layernorm.weight"]
    hid_norm = W["midlayer.hidden_norm.weight"]
    post_ln = W["midlayer.post_attention_layernorm.weight"]
    fin_norm = W["norm.weight"]

    out = {"source": "python-numpy-reference", "num_tokens": T,
           "hidden_size": H, "draft_vocab_size": draft_vocab,
           "positions": positions}
    k_cache, v_cache = [], []  # each [kv_dim] (post-RoPE for k, raw for v)

    for i in range(T):
        pos = positions[i]
        aux_concat = np.concatenate([low[i], mid[i], high[i]]).astype(np.float32)
        fc_out = (fc_w @ aux_concat).astype(np.float32)

        embed_norm = rmsnorm(emb_all[i], in_ln, eps)
        hidden_normed = rmsnorm(fc_out, hid_norm, eps)
        qkv_in = np.concatenate([embed_norm, hidden_normed]).astype(np.float32)

        q = (q_w @ qkv_in).astype(np.float32)
        k = (k_w @ qkv_in).astype(np.float32)
        v = (v_w @ qkv_in).astype(np.float32)
        q_r = rope(q, n_heads, head_dim, pos, theta)
        k_r = rope(k, n_kv, head_dim, pos, theta)
        k_cache.append(k_r)
        v_cache.append(v)

        # causal GQA over positions 0..i
        K = np.stack(k_cache).reshape(i + 1, n_kv, head_dim)
        V = np.stack(v_cache).reshape(i + 1, n_kv, head_dim)
        qh = q_r.reshape(n_heads, head_dim)
        heads = np.empty((n_heads, head_dim), np.float32)
        for h in range(n_heads):
            kv = h // group
            s = (K[:, kv, :] @ qh[h]).astype(np.float32) * scale
            s = s - s.max()
            w = np.exp(s).astype(np.float32)
            w = w / w.sum()
            heads[h] = (w[:, None] * V[:, kv, :]).sum(axis=0)
        heads = heads.reshape(-1)
        attn_out = (o_w @ heads).astype(np.float32)

        res = (fc_out + attn_out).astype(np.float32)
        y = rmsnorm(res, post_ln, eps)
        g = (gate_w @ y).astype(np.float32)
        u = (up_w @ y).astype(np.float32)
        act = (g / (1.0 + np.exp(-g))).astype(np.float32) * u
        mlp_out = (down_w @ act).astype(np.float32)
        res = (res + mlp_out).astype(np.float32)
        final_hidden = rmsnorm(res, fin_norm, eps)
        logits = (lm_w @ final_hidden).astype(np.float32)

        order = np.argsort(-logits)
        top5 = [int(x) for x in order[:5]]
        stages = {
            "aux_concat": aux_concat, "fc_out": fc_out, "embed_norm": embed_norm,
            "hidden_normed": hidden_normed, "qkv_in": qkv_in,
            "q_pre_rope": q, "k_pre_rope": k, "v": v,
            "q_post_rope": q_r, "k_post_rope": k_r,
            "attn_heads": heads, "attn_out": attn_out, "post_attn_norm": y,
            "mlp_out": mlp_out, "final_hidden": final_hidden,
            "draft_logits": logits,
        }
        for name, arr in stages.items():
            out[f"t{i}_{name}"] = [float(x) for x in np.asarray(arr).ravel()]
        out[f"t{i}_draft_top1"] = top5[0]
        out[f"t{i}_draft_top5"] = top5
        out[f"t{i}_target_top1"] = int(top5[0] + d2t[top5[0]])  # d2t is a DELTA

    json.dump(out, open(outpath, "w"))
    tops = [out[f"t{i}_draft_top1"] for i in range(T)]
    tgts = [out[f"t{i}_target_top1"] for i in range(T)]
    print(f"wrote {outpath}  T={T}  draft_top1={tops}  target_top1={tgts}")


if __name__ == "__main__":
    main()
