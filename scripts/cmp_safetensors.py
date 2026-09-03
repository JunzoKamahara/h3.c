#!/usr/bin/env python3
"""Compare named tensors between two safetensors shards (BF16-aware).

Used by QINT-015h-2b-3-fix to check whether the MiniMax-H3 text tower is
byte-identical to official Qwen3-VL-32B-Instruct (it is). Parses the
safetensors header by hand -- no `safetensors` package needed, just numpy.

usage: cmp_safetensors.py <fileA> <fileB> <tensor_name> [<tensor_name> ...]

Prints, per tensor: exact byte equality, a short SHA-256 of each side, and
cosine / relative-L2 / max|diff| over the BF16->f32 values.
"""
import sys, json, struct, hashlib
import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_tensor(path, hdr, base, name):
    if name not in hdr:
        return None, None
    meta = hdr[name]
    a, b = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dt in ("F16", "FP16"):
        f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt in ("F32", "FP32"):
        f32 = np.frombuffer(raw, dtype=np.float32)
    else:
        raise SystemExit(f"unhandled dtype {dt} for {name}")
    return raw, (f32, meta["shape"], dt)


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    ha, ba = read_header(fa)
    hb, bb = read_header(fb)
    print(f"A = {fa}")
    print(f"B = {fb}")
    print(f"{'tensor':52} {'shape':16} {'dt':5} {'exact':6} "
          f"{'cos':>12} {'relL2':>11} {'max|d|':>11}")
    for name in names:
        rawa, ta = load_tensor(fa, ha, ba, name)
        rawb, tb = load_tensor(fb, hb, bb, name)
        if ta is None or tb is None:
            print(f"{name:52} MISSING  A={ta is not None} B={tb is not None}")
            continue
        (a, sha, dta) = ta
        (b, shb, dtb) = tb
        exact = (rawa == rawb)
        sa = hashlib.sha256(rawa).hexdigest()[:16]
        sb = hashlib.sha256(rawb).hexdigest()[:16]
        if a.shape != b.shape:
            print(f"{name:52} SHAPE A={sha} B={shb}")
            continue
        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        na = np.linalg.norm(a64)
        nb = np.linalg.norm(b64)
        cos = float(a64 @ b64 / (na * nb)) if na and nb else float(a is b)
        rel = float(np.linalg.norm(a64 - b64) / na) if na else 0.0
        mx = float(np.max(np.abs(a64 - b64)))
        tag = "YES" if exact else "no"
        print(f"{name:52} {str(sha):16} {dta:5} {tag:6} "
              f"{cos:12.8f} {rel:11.3e} {mx:11.3e}   shaA={sa} shaB={sb}")


if __name__ == "__main__":
    main()
