#!/usr/bin/env python3
"""QINT-015h-1c -- staged parity check between the C EAGLE-3 reference trace and
a reference trace (numpy reference, or SpecForge/SGLang with hooks dumping the
same keys).

    python3 scripts/eagle3_compare.py <c_trace.json> <ref_trace.json>

Per stage: cosine similarity, max |diff|, relative L2. Gate (initial -- tighten
once the observed error is known):

    fc_out / *_norm / qkv_in / q_* / k_* / v / *_hidden / draft_logits
        cosine >= 0.99999
    draft_top1        exact
    draft_top5        >= 4 / 5 in common
    target_top1       exact  (draft_top1 after d2t)

Exit 0 iff every gate passes.
"""
import json
import math
import sys

STAGES = [
    "aux_concat", "fc_out", "embed_norm", "hidden_normed", "qkv_in",
    "q_pre_rope", "k_pre_rope", "v", "q_post_rope", "k_post_rope",
    "attn_heads", "attn_out", "post_attn_norm", "mlp_out", "final_hidden",
    "draft_logits",
]
COS_GATE = 0.99999


def cos(a, b):
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 1.0 if da == db else 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def stats(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mad = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    num = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    den = sum(x * x for x in a) ** 0.5 or 1.0
    return cos(a, b), mad, num / den


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    c = json.load(open(sys.argv[1]))
    r = json.load(open(sys.argv[2]))

    print(f"C   : {c.get('source')}  token={c.get('token_id')} pos={c.get('position')}")
    print(f"ref : {r.get('source')}  token={r.get('token_id')} pos={r.get('position')}")
    print(f"{'stage':<16} {'len':>7} {'cosine':>12} {'max|dif|':>12} {'relL2':>12}  gate")
    ok = True
    for s in STAGES:
        if s not in c or s not in r:
            print(f"{s:<16} {'--':>7}   (missing in one trace)")
            continue
        co, mad, rl2 = stats(c[s], r[s])
        g = co >= COS_GATE
        ok = ok and g
        print(f"{s:<16} {len(c[s]):>7} {co:>12.8f} {mad:>12.3e} {rl2:>12.3e}  "
              f"{'ok' if g else 'FAIL'}")

    t1 = c.get("draft_top1") == r.get("draft_top1")
    c5, r5 = set(c.get("draft_top5", [])), set(r.get("draft_top5", []))
    inter = len(c5 & r5)
    tt1 = c.get("target_top1") == r.get("target_top1")
    print()
    print(f"draft_top1   C={c.get('draft_top1')}  ref={r.get('draft_top1')}   "
          f"{'ok' if t1 else 'FAIL'}")
    print(f"draft_top5   C={sorted(c5)}  ref={sorted(r5)}   common={inter}/5   "
          f"{'ok' if inter >= 4 else 'FAIL'}")
    print(f"target_top1  C={c.get('target_top1')}  ref={r.get('target_top1')}   "
          f"{'ok' if tt1 else 'FAIL'}")

    ok = ok and t1 and (inter >= 4) and tt1
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
