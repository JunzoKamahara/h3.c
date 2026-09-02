/* QINT-015b -- test-only oracle draft backend. See qwen_draft.h. */

#include "qwen_draft.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *stream;
    size_t count;
    size_t corrupt_at;    /* absolute stream index to poison, or SIZE_MAX */
    uint32_t corrupt_token;
} oracle_state;

static int oracle_propose(qwen_draft_backend *self,
                          const qwen_draft_context *ctx, size_t max_tokens,
                          qwen_draft_proposal *out) {
    oracle_state *st = self->state;
    out->count = 0;

    /* The stream is prompt ++ reference; index `ctx->history_length` is where
     * the coordinator's anchor should sit. If the coordinator's actual path
     * has drifted from the recorded stream by a decode near-tie, re-sync by
     * looking for the anchor in a small window; otherwise give up for this
     * cycle (the coordinator falls back to a single scalar step). */
    size_t at = ctx->history_length;
    if (ctx->have_anchor) {
        if (at >= st->count) return 1;
        if (st->stream[at] != ctx->anchor_token) {
            size_t lo = at > 4 ? at - 4 : 0;
            size_t hi = at + 4 < st->count ? at + 4 : st->count;
            size_t found = st->count;
            for (size_t i = lo; i < hi; i++)
                if (st->stream[i] == ctx->anchor_token) { found = i; break; }
            if (found == st->count) return 1;
            at = found;
        }
        at += 1; /* propose what follows the anchor */
    }
    if (at >= st->count) return 1;

    size_t avail = st->count - at;
    size_t k = avail < max_tokens ? avail : max_tokens;
    for (size_t i = 0; i < k; i++) {
        size_t idx = at + i;
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
