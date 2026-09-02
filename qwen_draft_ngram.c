/* QINT-015c -- n-gram ("prompt lookup") draft backend. See qwen_draft.h.
 *
 * Stateless: builds the logical suffix (history + anchor when present), finds
 * its longest earlier occurrence, and proposes the tokens that followed that
 * occurrence. No model, no GPU, no tokeniser coupling -- the point is to
 * measure acceptance behaviour, which is strong on code / JSON / repetition
 * and weak on free-form prose. */

#include "qwen_draft.h"

#include <stdlib.h>
#include <string.h>

#define NGRAM_MAX_MATCH 32u  /* cap on the suffix length we try to match */
#define NGRAM_MAX_CTX   256u /* logical-history window we search             */

static int ngram_propose(qwen_draft_backend *self,
                         const qwen_draft_context *ctx, size_t max_tokens,
                         qwen_draft_proposal *out) {
    (void)self;
    out->count = 0;
    if (!ctx->history || ctx->history_length == 0) return 1;

    /* Logical history = the tail of `history` plus the anchor token. */
    uint32_t logical[NGRAM_MAX_CTX];
    size_t n = 0;
    size_t take = ctx->history_length;
    if (take > NGRAM_MAX_CTX - 1) take = NGRAM_MAX_CTX - 1;
    memcpy(logical, ctx->history + (ctx->history_length - take),
           take * sizeof(uint32_t));
    n = take;
    if (ctx->have_anchor) logical[n++] = ctx->anchor_token;
    if (n < 2) return 1;

    size_t max_match = n - 1;
    if (max_match > NGRAM_MAX_MATCH) max_match = NGRAM_MAX_MATCH;

    for (size_t L = max_match; L >= 1; L--) {
        const uint32_t *suffix = logical + (n - L);
        for (size_t p = n - L; p-- > 0;) {
            if (p + L >= n) continue; /* no continuation after the match */
            int eq = 1;
            for (size_t k = 0; k < L; k++)
                if (logical[p + k] != suffix[k]) { eq = 0; break; }
            if (!eq) continue;
            size_t avail = n - (p + L);
            size_t k = avail < max_tokens ? avail : max_tokens;
            for (size_t i = 0; i < k; i++) out->tokens[i] = logical[p + L + i];
            out->count = k;
            return 1;
        }
    }
    return 1;
}

qwen_draft_backend *qwen_draft_ngram_new(void) {
    qwen_draft_backend *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->name = "ngram";
    b->propose = ngram_propose;
    b->reset = NULL;
    b->destroy = (void (*)(qwen_draft_backend *))free;
    b->state = NULL;
    return b;
}
