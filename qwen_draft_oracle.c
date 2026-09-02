/* QINT-015b -- test-only oracle draft backend. See qwen_draft.h. */

#include "qwen_draft.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *stream;
    size_t count;
    size_t corrupt_at;    /* absolute history index to poison, or SIZE_MAX */
    uint32_t corrupt_token;
} oracle_state;

static int oracle_propose(qwen_draft_backend *self, const uint32_t *history,
                          size_t history_length, size_t max_tokens,
                          qwen_draft_proposal *out) {
    (void)history;
    oracle_state *st = self->state;
    out->count = 0;
    if (history_length >= st->count) return 1; /* nothing left to propose */
    size_t avail = st->count - history_length;
    size_t k = avail < max_tokens ? avail : max_tokens;
    for (size_t i = 0; i < k; i++) {
        size_t idx = history_length + i;
        out->tokens[i] =
            (idx == st->corrupt_at) ? st->corrupt_token : st->stream[idx];
    }
    out->count = k;
    return 1;
}

static void oracle_destroy(qwen_draft_backend *self) {
    if (!self) return;
    oracle_state *st = self->state;
    if (st) {
        free(st->stream);
        free(st);
    }
    free(self);
}

qwen_draft_backend *qwen_draft_oracle_new(const uint32_t *stream, size_t count,
                                          size_t corrupt_at,
                                          uint32_t corrupt_token) {
    qwen_draft_backend *b = calloc(1, sizeof(*b));
    oracle_state *st = calloc(1, sizeof(*st));
    uint32_t *copy = count ? malloc(count * sizeof(*copy)) : NULL;
    if (!b || !st || (count && !copy)) {
        free(b);
        free(st);
        free(copy);
        return NULL;
    }
    if (count) memcpy(copy, stream, count * sizeof(*copy));
    st->stream = copy;
    st->count = count;
    st->corrupt_at = corrupt_at;
    st->corrupt_token = corrupt_token;
    b->name = "oracle";
    b->propose = oracle_propose;
    b->reset = NULL;
    b->destroy = oracle_destroy;
    b->state = st;
    return b;
}
