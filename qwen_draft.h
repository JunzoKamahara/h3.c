#ifndef QWEN_DRAFT_H
#define QWEN_DRAFT_H

/* QINT-015 -- speculative-decoding draft backend interface.
 *
 * A draft backend cheaply proposes a short block of continuation tokens that
 * the Qwen3-VL target then verifies. The coordinator (qwen_spec.c) never
 * assumes a particular backend, so n-gram, a small LM, or a learned
 * DFlash/EAGLE head can be swapped in without touching the verifier.
 *
 * Contract:
 *  - `propose` sees the full token history (prompt + everything generated) and
 *    fills up to `max_tokens` (<= QWEN_DRAFT_MAX) speculative tokens.
 *  - Returning count 0 is legal: the coordinator then just takes the target's
 *    own next token for that cycle (no speculation lost, no correctness risk).
 *  - The backend must NOT be given, and must not require, the target's next
 *    token up front -- a future learned draft may generate D0..Dk directly
 *    from the frontier hidden state.
 *  - `propose` returns 1 on success, 0 on error (treated as "no proposal").
 */

#include <stddef.h>
#include <stdint.h>

#define QWEN_DRAFT_MAX 8u

typedef struct {
    uint32_t tokens[QWEN_DRAFT_MAX];
    size_t count;
} qwen_draft_proposal;

typedef struct qwen_draft_backend qwen_draft_backend;

struct qwen_draft_backend {
    const char *name;
    int (*propose)(qwen_draft_backend *self, const uint32_t *history,
                   size_t history_length, size_t max_tokens,
                   qwen_draft_proposal *out);
    void (*reset)(qwen_draft_backend *self); /* optional; new sequence */
    void (*destroy)(qwen_draft_backend *self); /* optional; frees self */
    void *state;
};

static inline int qwen_draft_propose(qwen_draft_backend *backend,
                                     const uint32_t *history,
                                     size_t history_length, size_t max_tokens,
                                     qwen_draft_proposal *out) {
    out->count = 0;
    if (!backend || !backend->propose) return 1;
    if (max_tokens > QWEN_DRAFT_MAX) max_tokens = QWEN_DRAFT_MAX;
    if (max_tokens == 0) return 1;
    return backend->propose(backend, history, history_length, max_tokens, out);
}

static inline void qwen_draft_reset(qwen_draft_backend *backend) {
    if (backend && backend->reset) backend->reset(backend);
}

static inline void qwen_draft_destroy(qwen_draft_backend *backend) {
    if (backend && backend->destroy) backend->destroy(backend);
}

/* --- test-only oracle draft (QINT-015b) --------------------------------- *
 * Replays a pre-recorded token stream (the known-correct greedy
 * continuation): with an aligned history it yields 100% acceptance, which is
 * the parity anchor for `spec-oracle-check`. `corrupt_at` (SIZE_MAX = never)
 * forces one wrong token at that absolute history index so the rejection /
 * rewind paths can be exercised deterministically. The stream is copied. */
qwen_draft_backend *qwen_draft_oracle_new(const uint32_t *stream, size_t count,
                                          size_t corrupt_at,
                                          uint32_t corrupt_token);

/* --- n-gram / prompt-lookup draft (QINT-015c) -------------------------- *
 * Longest-suffix match over the token history; proposes the continuation of
 * the most recent earlier occurrence. Stateless, no model. */
qwen_draft_backend *qwen_draft_ngram_new(void);

#endif
