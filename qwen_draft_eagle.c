/* QINT-015h-2b -- EAGLE-3 learned draft backend.
 *
 * Wires qwen_eagle3_chain() (the recurrent autoregressive draft) behind the
 * qwen_draft_backend vtable. Lifecycle:
 *
 *   qwen_draft_eagle_prime()  -- once per request: build the draft prefix K/V
 *                                over the whole committed context (2b-2a).
 *   propose()                 -- one speculative chain from the frontier aux
 *                                + anchor; appends its K/V to the same cache.
 *   sync()                    -- after the target verify: roll the draft K/V
 *                                back to base and re-extend it authoritatively
 *                                for the committed prefix (2b-2b).
 *   reset()                   -- new sequence.
 *
 * Invariant asserted at the start of every primed propose():
 *   draft_kv_len == history_length - 1        (rows 0..L-2; row L-1 is step 0)
 *
 * CPU reference speed -- QINT-015i measures T_draft; Metal only if it
 * dominates. Correctness-first: sync() drops ALL speculative K/V and rebuilds
 * the accepted prefix from the VERIFY aux; no speculative-K/V reuse yet.
 */

#include "qwen_draft.h"
#include "qwen_eagle3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    qwen_eagle3 *e;
    qwen_eagle3_kv *kv;
    qwen_draft_embed_fn embed;
    void *embed_ctx;
    int H;

    float *fr_f32;         /* [3 * H] scratch: frontier aux as f32 for the chain */
    const float *fr_ptr[3];

    int primed;
    int unsynced;          /* a sync inconsistency -> decline until reset */

    /* saved at propose() for the following sync() */
    int have_cycle;
    size_t cyc_L;                 /* history_length at propose() */
    uint32_t cyc_anchor;
    uint16_t *cyc_frontier_aux;   /* [3 * H] bf16 == h[L-1] */
    uint32_t cyc_proposals[QWEN_DRAFT_MAX]; /* target-vocab */
    size_t cyc_n;
} eagle_state;

static float bf16f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Append `n` prefix rows to the kv: row t = aux(t) + Emb(pair[t]) @ pos0+t,
 * where aux(t) for slot a is `aux_bf16[(a*aux_rows + t)*H]` (aux-major). K/V
 * only. Converts the needed rows bf16 -> f32 through a temp buffer. */
static int append_prefix(eagle_state *st, const uint16_t *aux_bf16,
                         size_t aux_rows, const uint32_t *pair, int n, int pos0,
                         char *err, size_t errn) {
    if (n <= 0) return 1;
    int H = st->H;
    float *f = malloc((size_t)3 * (size_t)n * H * sizeof(float));
    if (!f) {
        snprintf(err, errn, "out of memory in eagle prefix");
        return 0;
    }
    for (int a = 0; a < 3; a++)
        for (int t = 0; t < n; t++) {
            const uint16_t *src = aux_bf16 + ((size_t)a * aux_rows + (size_t)t) * H;
            float *dst = f + ((size_t)a * n + t) * H;
            for (int i = 0; i < H; i++) dst[i] = bf16f(src[i]);
        }
    const float *ap[3] = {f, f + (size_t)n * H, f + (size_t)2 * n * H};
    int ok = qwen_eagle3_kv_prefix_extend(st->e, st->kv, ap, pair, n, pos0,
                                          (qwen_eagle3_embed_fn)st->embed,
                                          st->embed_ctx, err, errn);
    free(f);
    return ok;
}

int qwen_draft_eagle_prime(qwen_draft_backend *b, const uint16_t *aux_all,
                           size_t all_rows, size_t n_aux, size_t hidden,
                           const uint32_t *history, size_t L, char *err,
                           size_t errn) {
    if (!b || !b->state || !aux_all || !history) {
        if (err && errn) snprintf(err, errn, "qwen_draft_eagle_prime: bad args");
        return 0;
    }
    eagle_state *st = b->state;
    if (n_aux < 3 || hidden != (size_t)st->H || all_rows != L || L < 1) {
        snprintf(err, errn,
                 "prime: need n_aux>=3, hidden==%d, all_rows==history_length>=1",
                 st->H);
        return 0;
    }
    qwen_eagle3_kv_reset(st->kv);
    /* rows t = 0 .. L-2 : pair token is history[t+1]. */
    int n = (int)L - 1;
    int ok = 1;
    if (n > 0) {
        uint32_t *pair = malloc((size_t)n * sizeof(uint32_t));
        if (!pair) {
            snprintf(err, errn, "out of memory");
            return 0;
        }
        for (int t = 0; t < n; t++) pair[t] = history[t + 1];
        ok = append_prefix(st, aux_all, all_rows, pair, n, 0, err, errn);
        free(pair);
    }
    if (!ok) return 0;
    if (qwen_eagle3_kv_len(st->kv) != n) {
        snprintf(err, errn, "prime: draft_kv_len %d != history_length-1 %d",
                 qwen_eagle3_kv_len(st->kv), n);
        return 0;
    }
    st->primed = 1;
    st->unsynced = 0;
    st->have_cycle = 0;
    return 1;
}

static int eagle_propose(qwen_draft_backend *self, const qwen_draft_context *ctx,
                         size_t max_tokens, qwen_draft_proposal *out) {
    eagle_state *st = self->state;
    out->count = 0;
    if (st->unsynced) return 1;
    if (!ctx || !ctx->have_anchor || ctx->n_aux < 3 ||
        (int)ctx->hidden_size != st->H)
        return 1;
    int H = st->H;
    for (int a = 0; a < 3; a++) {
        const uint16_t *src = ctx->aux_hidden[a];
        if (!src) return 1;
        float *dst = st->fr_f32 + (size_t)a * H;
        for (int i = 0; i < H; i++) dst[i] = bf16f(src[i]);
        st->fr_ptr[a] = dst;
    }

    size_t L = ctx->history_length;
    int base = L > 0 ? (int)L - 1 : 0;

    if (st->primed) {
        if (qwen_eagle3_kv_len(st->kv) != base) {
            st->unsynced = 1; /* fail-closed: invariant broken */
            return 1;
        }
    } else {
        qwen_eagle3_kv_reset(st->kv); /* 2b-0 fallback: no prefix (degenerate) */
    }

    /* save the cycle state for sync(). */
    st->cyc_L = L;
    st->cyc_anchor = ctx->anchor_token;
    for (int a = 0; a < 3; a++)
        memcpy(st->cyc_frontier_aux + (size_t)a * H, ctx->aux_hidden[a],
               (size_t)H * sizeof(uint16_t));

    int k = (int)max_tokens;
    if (k > (int)QWEN_DRAFT_MAX) k = (int)QWEN_DRAFT_MAX;
    if (k < 1) return 1;

    uint32_t draft_ids[QWEN_DRAFT_MAX];
    char err[256];
    if (!qwen_eagle3_chain(st->e, st->kv, st->fr_ptr, ctx->anchor_token, base, k,
                           (qwen_eagle3_embed_fn)st->embed, st->embed_ctx,
                           draft_ids, err, sizeof(err)))
        return 1;

    for (int j = 0; j < k; j++) {
        out->tokens[j] = qwen_eagle3_d2t(st->e, draft_ids[j]);
        st->cyc_proposals[j] = out->tokens[j];
    }
    st->cyc_n = (size_t)k;
    st->have_cycle = 1;
    out->count = (size_t)k;
    return 1;
}

static int fail_closed(eagle_state *st) {
    st->unsynced = 1;
    return 0;
}

static int eagle_sync(qwen_draft_backend *self,
                      const qwen_draft_sync_context *s) {
    eagle_state *st = self->state;
    if (!s || st->unsynced || !st->primed || !st->have_cycle) return 0;

    int H = st->H;
    size_t C = s->n_committed, L = st->cyc_L;
    if (C < 1 || s->n_aux < 3 || s->hidden_size != (size_t)H ||
        s->verify_rows < C || s->new_history_length != L + C)
        return fail_closed(st);
    if (s->committed_tokens[0] != st->cyc_anchor) return fail_closed(st);
    for (size_t j = 1; j < C; j++)
        if (s->committed_tokens[j] != st->cyc_proposals[j - 1])
            return fail_closed(st);

    char err[256];
    qwen_eagle3_kv_truncate(st->kv, (int)L - 1);

    /* row L-1 : the SAVED frontier aux h[L-1] + Emb(committed[0]=old anchor). */
    if (!append_prefix(st, st->cyc_frontier_aux, 1, &s->committed_tokens[0], 1,
                       (int)L - 1, err, sizeof(err)))
        return fail_closed(st);

    /* rows L .. L'-2 : VERIFY aux rows 0 .. C-2 + Emb(committed[1..C-1]).
     * VERIFY aux row (C-1) is h[L'-1] -- next cycle's frontier, NOT stored. */
    if (C > 1 &&
        !append_prefix(st, s->verify_aux, s->verify_rows, &s->committed_tokens[1],
                       (int)C - 1, (int)L, err, sizeof(err)))
        return fail_closed(st);

    if (qwen_eagle3_kv_len(st->kv) != (int)(s->new_history_length) - 1)
        return fail_closed(st);

    st->have_cycle = 0;
    return 1;
}

static void eagle_reset(qwen_draft_backend *self) {
    if (!self || !self->state) return;
    eagle_state *st = self->state;
    qwen_eagle3_kv_reset(st->kv);
    st->primed = 0;
    st->unsynced = 0;
    st->have_cycle = 0;
}

static void eagle_destroy(qwen_draft_backend *self) {
    if (!self) return;
    eagle_state *st = self->state;
    if (st) {
        qwen_eagle3_kv_free(st->kv);
        qwen_eagle3_free(st->e);
        free(st->fr_f32);
        free(st->cyc_frontier_aux);
        free(st);
    }
    free(self);
}

qwen_draft_backend *qwen_draft_eagle_new(const char *dir,
                                         qwen_draft_embed_fn embed,
                                         void *embed_ctx, char *error,
                                         size_t errn) {
    if (!dir || !embed) {
        if (error && errn) snprintf(error, errn, "qwen_draft_eagle_new: bad args");
        return NULL;
    }
    qwen_eagle3 *e = NULL;
    if (!qwen_eagle3_load(dir, &e, error, errn)) return NULL;
    const qwen_eagle3_config *c = qwen_eagle3_config_of(e);
    int H = c->hidden_size;

    qwen_eagle3_kv *kv = NULL;
    eagle_state *st = calloc(1, sizeof(*st));
    qwen_draft_backend *b = calloc(1, sizeof(*b));
    float *fr = malloc((size_t)3 * H * sizeof(float));
    uint16_t *cfa = malloc((size_t)3 * H * sizeof(uint16_t));
    if (!st || !b || !fr || !cfa || !qwen_eagle3_kv_new(e, &kv, error, errn)) {
        free(st);
        free(b);
        free(fr);
        free(cfa);
        qwen_eagle3_kv_free(kv);
        qwen_eagle3_free(e);
        if (error && errn && !error[0]) snprintf(error, errn, "out of memory");
        return NULL;
    }
    st->e = e;
    st->kv = kv;
    st->embed = embed;
    st->embed_ctx = embed_ctx;
    st->H = H;
    st->fr_f32 = fr;
    st->cyc_frontier_aux = cfa;
    b->name = "eagle3";
    b->propose = eagle_propose;
    b->sync = eagle_sync;
    b->reset = eagle_reset;
    b->destroy = eagle_destroy;
    b->state = st;
    return b;
}
