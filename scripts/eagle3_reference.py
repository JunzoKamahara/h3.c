#!/usr/bin/env python3
"""QINT-015h-1c -- EAGLE-3 draft-head reference forward (float32, staged).

Reads <checkpoint_dir>/config.json + model.safetensors and a fixture produced
by `h3_qwen_eagle3_test gen-fixture` (3 aux hidden rows + the token's target
embedding baked in, so the 32B target is never loaded). Runs one EAGLE-3 step
and writes ref_trace.json with the same stage keys as the C dump, for
scripts/eagle3_compare.py.

This is a *self-contained* reference. The authority is SGLang's
`python/sglang/srt/models/llama_eagle3.py` / SpecForge
`specforge/algorithms/eagle3/model.py`; every step below is annotated with the
assumption it makes so it can be checked against that source. If you have
SpecForge or SGLang installed, prefer running their `LlamaForCausalLMEagle3`
with forward hooks and dumping the same keys -- the trace JSON is the contract.

    python3 scripts/eagle3_reference.py <checkpoint_dir> <fixture.json> <ref_trace.json>

Needs numpy only (`pip install numpy`).
"""
import json
import struct
import sys

import numpy as np


def load_safetensors_f32(path):
    """name -> float32 ndarray. bf16/f16 are widened to f32; ints kept as-is."""
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
            dt = meta["dtype"]
            shape = meta["shape"]
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
                arr = (u16 << 16).view(np.float32).reshape(shape)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
            elif dt in ("I64", "U64"):
                arr = np.frombuffer(raw, dtype="<i8").reshape(shape)
            elif dt in ("I32", "U32"):
                arr = np.frombuffer(raw, dtype="<i4").reshape(shape)
            elif dt in ("BOOL", "I8", "U8"):
                arr = np.frombuffer(raw, dtype="u1").reshape(shape)
            else:
                raise SystemExit(f"unhandled dtype {dt} for {name}")
            out[name] = np.ascontiguousarray(arr)
        return out


def rmsnorm(x, w, eps):
    # x * rsqrt(mean(x^2) + eps) * w, all in float32.
    x = x.astype(np.float32)
    var = np.mean(x * x, dtype=np.float32)
    return (x * np.float32(1.0 / np.sqrt(var + eps)) * w.astype(np.float32)).astype(np.float32)


def rope(vec, n_heads, head_dim, position, theta):
    """HF 'default' rotary: q_embed = q*cos + rotate_half(q)*sin, cos/sin tiled
    over the two halves. Matches the C reference's per-half rotation."""
    v = vec.reshape(n_heads, head_dim).astype(np.float32)
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = theta ** (-2.0 * i / head_dim)
    ang = position * inv_freq
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    a = v[:, :half]
    b = v[:, half:]
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
    head_dim = int(cfg["head_dim"])            # explicit -- do NOT derive H/n_heads
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rp = cfg.get("rope_parameters", {})
    theta = float(rp.get("rope_theta", cfg.get("rope_theta", 5e6)))
    draft_vocab = int(cfg["draft_vocab_size"])
    assert fx["hidden_size"] == H

    token = int(fx["token_id"])
    position = int(fx["position"])
    low = np.asarray(fx["aux_hidden_low"], dtype=np.float32)
    mid = np.asarray(fx["aux_hidden_mid"], dtype=np.float32)
    high = np.asarray(fx["aux_hidden_high"], dtype=np.float32)
    emb = np.asarray(fx["embedding"], dtype=np.float32)

    tr = {}

    # 1. fuse the 3 aux hidden states. ASSUMPTION: concat order low, mid, high;
    #    fc.weight is [H, 3H], no bias.
    aux_concat = np.concatenate([low, mid, high]).astype(np.float32)
    tr["aux_concat"] = aux_concat
    fc_out = (W["fc.weight"].astype(np.float32) @ aux_concat).astype(np.float32)
    tr["fc_out"] = fc_out

    # 2. EAGLE-3 norms the embedding and the fused hidden separately, then
    #    concatenates. ASSUMPTION: input_layernorm on the embedding, hidden_norm
    #    on the fused hidden; concat = [norm(emb), norm(fused)].
    embed_norm = rmsnorm(emb, W["midlayer.input_layernorm.weight"], eps)
    hidden_normed = rmsnorm(fc_out, W["midlayer.hidden_norm.weight"], eps)
    tr["embed_norm"] = embed_norm
    tr["hidden_normed"] = hidden_normed
    qkv_in = np.concatenate([embed_norm, hidden_normed]).astype(np.float32)
    tr["qkv_in"] = qkv_in

    q = (W["midlayer.self_attn.q_proj.weight"].astype(np.float32) @ qkv_in).astype(np.float32)
    k = (W["midlayer.self_attn.k_proj.weight"].astype(np.float32) @ qkv_in).astype(np.float32)
    v = (W["midlayer.self_attn.v_proj.weight"].astype(np.float32) @ qkv_in).astype(np.float32)
    tr["q_pre_rope"] = q
    tr["k_pre_rope"] = k
    tr["v"] = v

    # 3. RoPE at the fixture position. ASSUMPTION: absolute position, plain 1-D
    #    rotary from the draft config (NOT the target's mRoPE).
    q_r = rope(q, n_heads, head_dim, position, theta)
    k_r = rope(k, n_kv, head_dim, position, theta)
    tr["q_post_rope"] = q_r
    tr["k_post_rope"] = k_r

    # 4. single position -> softmax over one key = 1 -> attn head = grouped v.
    group = n_heads // n_kv
    heads = np.empty(n_heads * head_dim, dtype=np.float32)
    vv = v.reshape(n_kv, head_dim)
    for hh in range(n_heads):
        heads[hh * head_dim:(hh + 1) * head_dim] = vv[hh // group]
    tr["attn_heads"] = heads
    attn_out = (W["midlayer.self_attn.o_proj.weight"].astype(np.float32) @ heads).astype(np.float32)
    tr["attn_out"] = attn_out

    # 5. residual is the FUSED hidden (not the embedding).
    res = (fc_out + attn_out).astype(np.float32)
    y = rmsnorm(res, W["midlayer.post_attention_layernorm.weight"], eps)
    tr["post_attn_norm"] = y
    g = (W["midlayer.mlp.gate_proj.weight"].astype(np.float32) @ y).astype(np.float32)
    u = (W["midlayer.mlp.up_proj.weight"].astype(np.float32) @ y).astype(np.float32)
    act = (g / (1.0 + np.exp(-g))).astype(np.float32) * u
    mlp_out = (W["midlayer.mlp.down_proj.weight"].astype(np.float32) @ act).astype(np.float32)
    tr["mlp_out"] = mlp_out
    res = (res + mlp_out).astype(np.float32)

    final_hidden = rmsnorm(res, W["norm.weight"], eps)
    tr["final_hidden"] = final_hidden
    logits = (W["lm_head.weight"].astype(np.float32) @ final_hidden).astype(np.float32)
    tr["draft_logits"] = logits

    order = np.argsort(-logits)
    top5 = [int(x) for x in order[:5]]
    d2t = W["d2t"].astype(np.int64)
    target_top1 = int(top5[0] + d2t[top5[0]])  # d2t is a DELTA

    out = {
        "source": "python-numpy-reference",
        "token_id": token,
        "position": position,
        "hidden_size": H,
        "draft_vocab_size": draft_vocab,
        "draft_top1": top5[0],
        "draft_top5": top5,
        "target_top1": target_top1,
    }
    for kk, vvv in tr.items():
        out[kk] = [float(x) for x in np.asarray(vvv).ravel()]
    json.dump(out, open(outpath, "w"))
    print(f"wrote {outpath}  draft top1={top5[0]} -> target {target_top1}  top5={top5}")


if __name__ == "__main__":
    main()
