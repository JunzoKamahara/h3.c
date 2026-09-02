/* QINT-015c -- n-gram ("prompt lookup") draft backend. See qwen_draft.h.
 *
 * Stateless: on each proposal it finds the longest suffix of the history that
 * also occurs earlier, and proposes the tokens that followed that earlier
 * occurrence. No model, no GPU, no tokeniser coupling -- the point is to
 * measure acceptance behaviour, which is strong on code / JSON / repetition
 * and weak on free-form prose. */

#include "qwen_draft.h"

#include <stdlib.h>

#define NGRAM_MAX_MATCH 32u   /* cap on the suffix length we try to match */

static int ngram_propose(qwen_draft_backend *self, const uint32_t *history,
                         size_t history_length, size_t max_tokens,
                         qwen_draft_proposal *out) {
    (void)self;
    out->count = 0;
    if (!history || history_length < 2) return 1;

    size_t n = history_length;
    size_t max_match = n - 1;
    if (max_match > NGRAM_MAX_MATCH) max_match = NGRAM_MAX_MATCH;

    for (size_t L = max_match; L >= 1; L--) {
        const uint32_t *suffix = history + (n - L);
        /* Most recent earlier occurrence first: better locality, and matches
         * how repetition actually continues. */
        for (size_t p = n - L; p-- > 0;) {
            if (p + L >= n) continue; /* no continuation after the match */
            int eq = 1;
            for (size_t k = 0; k < L; k++)
                if (history[p + k] != suffix[k]) { eq = 0; break; }
            if (!eq) continue;
            size_t avail = n - (p + L);
            size_t take = avail < max_tokens ? avail : max_tokens;
            for (size_t k = 0; k < take; k++)
                out->tokens[k] = history[p + L + k];
            out->count = take;
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
