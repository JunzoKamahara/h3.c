#ifndef QWEN_DRAFT_H
#define QWEN_DRAFT_H

/* QINT-015 -- speculative-decoding draft backend interface.
 *
 * A draft backend cheaply proposes a short run of continuation tokens that the
 * Qwen3-VL target then verifies in one batch. The coordinator (qwen_spec.c)
 * never assumes a particular backend, so an n-gram, a small language model, or
 * a learned DFlash / EAGLE head can be swapped in without touching the
 * verifier.
 *
 * The coordinator always knows the *anchor*: the single token the target has
 * already decided comes next (the argmax of the current frontier logits, or a
 * pending anchor carried over from a previous cycle -- see QINT-015d-2). The
 * anchor is passed to the backend as an OPTION:
 *   - A backend MAY use it (an n-gram treats history + anchor as the logical
 *     suffix to match; an oracle checks the anchor lines up with its stream).
 *   - A backend MAY ignore it (a learned draft that generates D1..Dk directly
 *     from a frontier hidden state).
 *   - A backend MUST NOT require it.
 *
 * `propose` fills up to `max_tokens` (<= QWEN_DRAFT_MAX) tokens that are the
 * predicted continuation AFTER the anchor. Returning count 0 is legal: the
 * coordinator then falls back to a single scalar step for that cycle.
 * `propose` returns 1 on success, 0 on error (treated as "no proposal").
 */

#include <stddef.h>
#include <stdint.h>

#define QWEN_DRAFT_MAX 8u

typedef struct {
    uint32_t tokens[QWEN_DRAFT_MAX];
    size_t count;
} qwen_draft_proposal;

/* What the backend sees each cycle. `history` is every token already committed
 * (prompt + emitted output), length `history_length`. `anchor_token` is the
 * target's guaranteed next token when `have_anchor` is set. `frontier_hidden`
 * is reserved for a learned draft (QINT-015h/i) and is NULL / 0 for now. */
typedef struct {
    const uint32_t *history;
    size_t history_length;
    int have_anchor;
    uint32_t anchor_token;
    const uint16_t *frontier_hidden;
    size_t frontier_hidden_size;
} qwen_draft_context;

typedef struct qwen_draft_backend qwen_draft_backend;

struct qwen_draft_backend {
    const char *name;
    int (*propose)(qwen_draft_backend *self, const qwen_draft_context *ctx,
                   size_t max_tokens, qwen_draft_proposal *out);
    void (*reset)(qwen_draft_backend *self);   /* optional; new sequence */
    void (*destroy)(qwen_draft_backend *self); /* optional; frees self */
    void *state;
};

static inline int qwen_draft_propose(qwen_draft_backend *backend,
                                     const qwen_draft_context *ctx,
                                     size_t max_tokens,
                                     qwen_draft_proposal *out) {
    out->count = 0;
    if (!backend || !backend->propose) return 1;
    if (max_tokens > QWEN_DRAFT_MAX) max_tokens = QWEN_DRAFT_MAX;
    if (max_tokens == 0) return 1;
    return backend->propose(backend, ctx, max_tokens, out);
}

static inline void qwen_draft_reset(qwen_draft_backend *backend) {
    if (backend && backend->reset) backend->reset(backend);
}

static inline void qwen_draft_destroy(qwen_draft_backend *backend) {
    if (backend && backend->destroy) backend->destroy(backend);
}

/* --- test-only oracle draft (QINT-015b) --------------------------------- *
 * Replays a pre-recorded token stream (the known-correct greedy
 * continuation). With an aligned history + anchor it yields 100 % acceptance,
 * the parity anchor for `spec-pending-oracle-check`. `corrupt_at`
 * (SIZE_MAX = never) forces one wrong token at that absolute stream index so
 * the rejection / rewind paths can be exercised deterministically. The stream
 * is copied. */
qwen_draft_backend *qwen_draft_oracle_new(const uint32_t *stream, size_t count,
                                          size_t corrupt_at,
                                          uint32_t corrupt_token);

/* --- n-gram / prompt-lookup draft (QINT-015c) -------------------------- *
 * Longest-suffix match over history + anchor; proposes the continuation of
 * the most recent earlier occurrence. Stateless, no model. */
qwen_draft_backend *qwen_draft_ngram_new(void);

#endif
