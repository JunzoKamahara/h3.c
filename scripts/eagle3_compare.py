#!/usr/bin/env python3
"""QINT-015h-1c / -2a -- staged parity between the C EAGLE-3 trace and a
reference trace (numpy reference, or SpecForge/SGLang with hooks dumping the
same keys).

    python3 scripts/eagle3_compare.py <c_trace.json> <ref_trace.json>

Both traces carry `num_tokens` T and per-token keys `t{i}_<stage>`. Per stage:
cosine, max |diff|, relative L2. Gate (initial -- tighten once the observed
error is known):

    fc_out / *_norm / qkv_in / q_* / k_* / v / *_hidden / draft_logits
        cosine >= 0.99999   (every token)
    draft_top1   exact   (every token)
    draft_top5   >= 4 / 5 in common
    target_top1  exact

Exit 0 iff every gate passes for every token.
"""
import json
import sys

STAGES = [
    "aux_concat", "fc_out", "embed_norm", "hidden_normed", "qkv_in",
    "q_pre_rope", "k_pre_rope", "v", "q_post_rope", "k_post_rope",
    "attn_heads", "attn_out", "post_attn_norm", "mlp_out", "final_hidden",
    "draft_logits",
]
COS_GATE = 0.99999


def stats(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    dp = sum(x * y for x, y in zip(a, b))
    cos = 1.0 if (da == 0 and db == 0) else (dp / (da * db) if da and db else 0.0)
    mad = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    rl2 = (sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5) / (da or 1.0)
    return cos, mad, rl2


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    c = json.load(open(sys.argv[1]))
    r = json.load(open(sys.argv[2]))
    T = min(int(c.get("num_tokens", 1)), int(r.get("num_tokens", 1)))
    print(f"C   : {c.get('source')}   ref : {r.get('source')}   num_tokens={T}")

    ok = True
    for i in range(T):
        print(f"\n-- token {i}  (pos C={c.get('positions',[None]*T)[i]} "
              f"ref={r.get('positions',[None]*T)[i]}) "
              f"------------------------------------")
        print(f"{'stage':<16} {'len':>7} {'cosine':>12} {'max|dif|':>11} {'relL2':>11}  gate")
        for s in STAGES:
            ck, rk = f"t{i}_{s}", f"t{i}_{s}"
            if ck not in c or rk not in r:
                print(f"{s:<16} {'--':>7}   (missing)")
                continue
            co, mad, rl2 = stats(c[ck], r[rk])
            g = co >= COS_GATE
            ok = ok and g
            print(f"{s:<16} {len(c[ck]):>7} {co:>12.8f} {mad:>11.3e} {rl2:>11.3e}  "
                  f"{'ok' if g else 'FAIL'}")

        ct1, rt1 = c.get(f"t{i}_draft_top1"), r.get(f"t{i}_draft_top1")
        c5, r5 = set(c.get(f"t{i}_draft_top5", [])), set(r.get(f"t{i}_draft_top5", []))
        inter = len(c5 & r5)
        ctt, rtt = c.get(f"t{i}_target_top1"), r.get(f"t{i}_target_top1")
        t1ok, t5ok, ttok = ct1 == rt1, inter >= 4, ctt == rtt
        ok = ok and t1ok and t5ok and ttok
        print(f"draft_top1   C={ct1} ref={rt1}   {'ok' if t1ok else 'FAIL'}")
        print(f"draft_top5   common={inter}/5   {'ok' if t5ok else 'FAIL'}")
        print(f"target_top1  C={ctt} ref={rtt}   {'ok' if ttok else 'FAIL'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
