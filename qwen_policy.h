#ifndef QWEN_POLICY_H
#define QWEN_POLICY_H

/* QINT-015a -- execution geometry and precision policy are separate concerns.
 *
 * Before speculative decoding, `rows == 1` in the KV path meant BOTH "this is
 * an autoregressive decode step" AND "use the INT4 GEMV". Speculative
 * verification needs `rows = 2..N` while still running the *same* precision
 * policy as scalar decode, so the two ideas are split:
 *
 *   qwen_eval_kind      -- what shape of work this is; picks kernel geometry
 *   qwen_decode_policy  -- which weights are W4 vs BF16; independent of rows
 *
 * The policy is resolved once from the environment (H3_QWEN_Q4, which
 * `h3_serve` writes for its --fast / --quality flags) and is the single place
 * that mapping lives. */

typedef enum {
    QWEN_EVAL_PREFILL, /* rows > 1, bulk prompt ingest (BF16 parity anchor)  */
    QWEN_EVAL_DECODE,  /* rows == 1, autoregressive head                     */
    QWEN_EVAL_VERIFY   /* rows 2..QWEN_SPEC_MAX, speculative candidate block */
} qwen_eval_kind;

typedef enum {
    QWEN_DECODE_POLICY_MIXED = 0, /* Mixed-W4/BF16 -- the default            */
    QWEN_DECODE_POLICY_FAST,      /* Pure W4A16                              */
    QWEN_DECODE_POLICY_QUALITY    /* BF16                                    */
} qwen_decode_policy;

/* Resolve H3_QWEN_Q4:
 *   unset / "0" / "bf16"  -> QUALITY (no quantised weights)
 *   "mixed"               -> MIXED
 *   anything else ("1")   -> FAST (every projection W4)
 * This never consults `rows`. */
qwen_decode_policy qwen_decode_policy_current(void);

const char *qwen_decode_policy_name(qwen_decode_policy policy);

/* 1 iff the policy quantises any projection to W4 (MIXED or FAST). Equivalent
 * to the historical qwen_q4_enabled(). */
int qwen_policy_uses_q4(qwen_decode_policy policy);

#endif
