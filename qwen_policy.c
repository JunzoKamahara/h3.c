#include "qwen_policy.h"

#include <stdlib.h>
#include <string.h>

qwen_decode_policy qwen_decode_policy_current(void) {
    const char *v = getenv("H3_QWEN_Q4");
    if (!v || !v[0]) return QWEN_DECODE_POLICY_QUALITY;
    if (strcmp(v, "0") == 0 || strcmp(v, "bf16") == 0)
        return QWEN_DECODE_POLICY_QUALITY;
    if (strcmp(v, "mixed") == 0) return QWEN_DECODE_POLICY_MIXED;
    return QWEN_DECODE_POLICY_FAST;
}

const char *qwen_decode_policy_name(qwen_decode_policy policy) {
    switch (policy) {
    case QWEN_DECODE_POLICY_MIXED:   return "Mixed-W4/BF16";
    case QWEN_DECODE_POLICY_FAST:    return "Pure W4A16";
    case QWEN_DECODE_POLICY_QUALITY: return "BF16";
    }
    return "?";
}

int qwen_policy_uses_q4(qwen_decode_policy policy) {
    return policy != QWEN_DECODE_POLICY_QUALITY;
}
