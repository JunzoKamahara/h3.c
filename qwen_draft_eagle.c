/* QINT-015h-2b -- EAGLE-3 learned draft backend.
 *
 * Wires qwen_eagle3_chain() (the recurrent autoregressive draft, QINT-015h-2a)
 * behind the qwen_draft_backend vtable so the speculative coordinator can use
 * it. Reads the target's 3 aux hidden states from qwen_draft_context, fuses
 * them for step 0 with the anchor token, then recurs on EAGLE's own hidden.
 * CPU reference speed -- QINT-015i measures T_draft; Metal only if it
 * dominates.
 *
 * NOT yet a drop-in for long contexts: 2b-0 uses a fresh per-cycle draft KV
 * (no draft prefill over the committed context) and does not catch the draft
 * KV up to the accepted target prefix after a verify -- that is QINT-015h-2b-2.
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
    float *aux_f32;            /* scratch [3 * hidden_size] */
    const float *aux_ptr[3];
    int hidden_size;
} eagle_state;

static float bf16f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int eagle_propose(qwen_draft_backend *self, const qwen_draft_context *ctx,
                         size_t max_tokens, qwen_draft_proposal *out) {
    eagle_state *st = self->state;
    out->count = 0;
    if (!ctx || !ctx->have_anchor || ctx->n_aux < 3 ||
        (int)ctx->hidden_size != st->hidden_size)
        return 1; /* not configured for EAGLE -> scalar fallback */

    int H = st->hidden_size;
    for (int a = 0; a < 3; a++) {
        const uint16_t *src = ctx->aux_hidden[a];
        if (!src) return 1;
        float *dst = st->aux_f32 + (size_t)a * H;
        for (int i = 0; i < H; i++) dst[i] = bf16f(src[i]);
        st->aux_ptr[a] = dst;
    }

    int k = (int)max_tokens;
    if (k > (int)QWEN_DRAFT_MAX) k = (int)QWEN_DRAFT_MAX;
    if (k < 1) return 1;

    /* 2b-0: fresh per-cycle draft KV. */
    qwen_eagle3_kv_reset(st->kv);
    int start_pos = (int)ctx->history_length; /* the anchor's position */

    uint32_t draft_ids[QWEN_DRAFT_MAX];
    char err[256];
    if (!qwen_eagle3_chain(st->e, st->kv, st->aux_ptr, ctx->anchor_token,
                           start_pos, k, (qwen_eagle3_embed_fn)st->embed,
                           st->embed_ctx, draft_ids, err, sizeof(err)))
        return 1; /* soft failure -> scalar fallback for this cycle */

    for (int j = 0; j < k; j++)
        out->tokens[j] = qwen_eagle3_d2t(st->e, draft_ids[j]); /* -> target vocab */
    out->count = (size_t)k;
    return 1;
}

static void eagle_reset(qwen_draft_backend *self) {
    if (self && self->state) qwen_eagle3_kv_reset(((eagle_state *)self->state)->kv);
}

static void eagle_destroy(qwen_draft_backend *self) {
    if (!self) return;
    eagle_state *st = self->state;
    if (st) {
        qwen_eagle3_kv_free(st->kv);
        qwen_eagle3_free(st->e);
        free(st->aux_f32);
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

    qwen_eagle3_kv *kv = NULL;
    eagle_state *st = calloc(1, sizeof(*st));
    qwen_draft_backend *b = calloc(1, sizeof(*b));
    float *aux = malloc((size_t)3 * c->hidden_size * sizeof(float));
    if (!st || !b || !aux || !qwen_eagle3_kv_new(e, &kv, error, errn)) {
        free(st);
        free(b);
        free(aux);
        qwen_eagle3_kv_free(kv);
        qwen_eagle3_free(e);
        if (error && errn && !error[0]) snprintf(error, errn, "out of memory");
        return NULL;
    }
    st->e = e;
    st->kv = kv;
    st->embed = embed;
    st->embed_ctx = embed_ctx;
    st->aux_f32 = aux;
    st->hidden_size = c->hidden_size;
    b->name = "eagle3";
    b->propose = eagle_propose;
    b->reset = eagle_reset;
    b->destroy = eagle_destroy;
    b->state = st;
    return b;
}
