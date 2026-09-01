#include "qwen_engine_internal.h"
#include "qwen_layers.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 2 -- Qwen3-VL KV-cache chat session (spec sections 7.2, 13, 14).
 *
 * The context owns a persistent GPU device, the streamed weight store, resident
 * embedding / final-norm / lm_head weights, and a per-layer K/V cache that
 * lives directly in GPU buffers sized to the session capacity. Each eval runs
 * the full 64-layer forward on only the new tokens: qwen_layer_prep() produces
 * their RoPE'd Q/K/V, the new K/V rows are appended to the cache at their
 * absolute offset, and h3_gpu_gqa_causal_kv_bf16() attends the new queries over
 * the whole cache. When the first eval covers the whole prompt (past = 0) the
 * cached attention reduces bit-for-bit to the Phase 1 plain causal path, which
 * is the parity anchor for `make phase2-parity`.
 *
 * Decoder-layer weights are still streamed per layer per eval; residency is a
 * later phase. */

#define KV_DEFAULT_CAPACITY 4096u

struct qwen_kv_context {
    h3_gpu *gpu;
    h3_weight_store *store;

    h3_gpu_tensor *embed_weight;
    h3_gpu_tensor *final_norm_weight;
    h3_gpu_tensor *lm_head_weight;

    /* Per-layer K/V cache, each [capacity, QWEN_LM_KV_DIM] BF16 in a GPU
     * buffer; only the first `length` rows are live. */
    h3_gpu_tensor *k_cache[QWEN_LM_TOTAL_LAYERS];
    h3_gpu_tensor *v_cache[QWEN_LM_TOTAL_LAYERS];

    uint32_t capacity;
    uint32_t length;

    uint32_t *history; /* [capacity] token ids */

    qwen_logits logits; /* latest next-token logits (last position) */
    int have_logits;

    /* Approach B: when non-NULL, all 64 decoder layers are held resident in
     * Unified Memory instead of being streamed from disk on every eval. */
    qwen_layer_weights *resident_layers;
};

static void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

void qwen_kv_context_free(qwen_kv_context *kv) {
    if (!kv) return;
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        h3_gpu_tensor_free(kv->k_cache[layer]);
        h3_gpu_tensor_free(kv->v_cache[layer]);
        if (kv->resident_layers)
            qwen_layer_weights_free(&kv->resident_layers[layer]);
    }
    free(kv->resident_layers);
    h3_gpu_tensor_free(kv->embed_weight);
    h3_gpu_tensor_free(kv->final_norm_weight);
    h3_gpu_tensor_free(kv->lm_head_weight);
    free(kv->history);
    qwen_logits_free(&kv->logits);
    if (kv->gpu) h3_gpu_free(kv->gpu);
    if (kv->store) h3_weight_store_free(kv->store);
    free(kv);
}

static uint32_t kv_capacity_from_env(void) {
    const char *value = getenv("H3_QWEN_KV_CAPACITY");
    if (!value || !*value) return KV_DEFAULT_CAPACITY;
    char *tail = NULL;
    long parsed = strtol(value, &tail, 10);
    if (!tail || *tail || parsed < 1 || parsed > 1 << 20)
        return KV_DEFAULT_CAPACITY;
    return (uint32_t)parsed;
}

static int kv_resident_from_env(void) {
    const char *value = getenv("H3_QWEN_RESIDENT");
    return value && *value && strcmp(value, "0") != 0;
}

/* Approach B: pin all 64 decoder layers in Unified Memory (~62 GB BF16). */
static int context_load_resident(qwen_kv_context *kv, char *error,
                                 size_t error_size) {
    kv->resident_layers =
        calloc(QWEN_LM_TOTAL_LAYERS, sizeof(*kv->resident_layers));
    if (!kv->resident_layers) {
        set_error(error, error_size, "out of memory for resident weight table");
        return 0;
    }
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        if (!qwen_layer_weights_load(kv->store, kv->gpu, layer,
                                     &kv->resident_layers[layer], error,
                                     error_size))
            return 0;
        if ((layer + 1) % 8 == 0 || layer + 1 == QWEN_LM_TOTAL_LAYERS)
            fprintf(stderr, "Qwen resident weights: %d/%d layers\n", layer + 1,
                    QWEN_LM_TOTAL_LAYERS);
    }
    return 1;
}

static int context_create(struct qwen_session *session, char *error,
                          size_t error_size) {
    qwen_kv_context *kv = calloc(1, sizeof(*kv));
    if (!kv) {
        set_error(error, error_size, "out of memory allocating KV context");
        return 0;
    }
    kv->capacity = kv_capacity_from_env();

    kv->store = h3_weight_store_open(session->engine->weight_directory, error,
                                    error_size);
    if (!kv->store) {
        qwen_kv_context_free(kv);
        return 0;
    }
    kv->gpu = h3_gpu_create(session->engine->shader_source_path, error,
                            error_size);
    if (!kv->gpu) {
        qwen_kv_context_free(kv);
        return 0;
    }
    h3_gpu_profile_set_label(kv->gpu, "Qwen KV session");

    uint64_t embed_shape[] = {QWEN_LM_VOCAB, QWEN_LM_HIDDEN};
    uint64_t norm_shape[] = {QWEN_LM_HIDDEN};
    kv->embed_weight = h3_weight_load_bf16(
        kv->store, kv->gpu, "model.language_model.embed_tokens.weight", 2,
        embed_shape, error, error_size);
    kv->final_norm_weight = h3_weight_load_bf16(
        kv->store, kv->gpu, "model.language_model.norm.weight", 1, norm_shape,
        error, error_size);
    kv->lm_head_weight = h3_weight_load_bf16(kv->store, kv->gpu,
                                             "lm_head.weight", 2, embed_shape,
                                             error, error_size);
    if (!kv->embed_weight || !kv->final_norm_weight || !kv->lm_head_weight) {
        qwen_kv_context_free(kv);
        return 0;
    }

    size_t cache_elements = (size_t)kv->capacity * QWEN_LM_KV_DIM;
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        kv->k_cache[layer] = h3_gpu_tensor_new_bf16(kv->gpu, cache_elements);
        kv->v_cache[layer] = h3_gpu_tensor_new_bf16(kv->gpu, cache_elements);
        if (!kv->k_cache[layer] || !kv->v_cache[layer]) {
            set_error(error, error_size,
                      "cannot allocate KV cache for %u tokens: %s",
                      kv->capacity, h3_gpu_error(kv->gpu));
            qwen_kv_context_free(kv);
            return 0;
        }
    }

    kv->history = malloc((size_t)kv->capacity * sizeof(*kv->history));
    if (!kv->history) {
        set_error(error, error_size, "out of memory allocating token history");
        qwen_kv_context_free(kv);
        return 0;
    }

    if ((session->resident_requested || kv_resident_from_env()) &&
        !context_load_resident(kv, error, error_size)) {
        qwen_kv_context_free(kv);
        return 0;
    }

    session->kv = kv;
    return 1;
}

typedef struct {
    h3_gpu_tensor *ids;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key_new;
    h3_gpu_tensor *value_new;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *logits;
    float *cosines;
    float *sines;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    uint16_t *kv_scratch; /* [m, QWEN_LM_KV_DIM] host readback */
    uint16_t *logits_host;
} eval_scratch;

static void eval_scratch_free(eval_scratch *s) {
    h3_gpu_tensor_free(s->ids);
    h3_gpu_tensor_free(s->hidden);
    h3_gpu_tensor_free(s->norm);
    h3_gpu_tensor_free(s->query);
    h3_gpu_tensor_free(s->key_new);
    h3_gpu_tensor_free(s->value_new);
    h3_gpu_tensor_free(s->attention_heads);
    h3_gpu_tensor_free(s->attention_output);
    h3_gpu_tensor_free(s->gate);
    h3_gpu_tensor_free(s->up);
    h3_gpu_tensor_free(s->mlp_output);
    h3_gpu_tensor_free(s->logits);
    h3_gpu_tensor_free(s->rope_cos);
    h3_gpu_tensor_free(s->rope_sin);
    free(s->cosines);
    free(s->sines);
    free(s->kv_scratch);
    free(s->logits_host);
    memset(s, 0, sizeof(*s));
}

static int gpu_ok(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *what) {
    if (ok) return 1;
    set_error(error, error_size, "Qwen KV %s failed: %s", what,
              h3_gpu_error(gpu));
    return 0;
}

int qwen_kv_eval(struct qwen_session *session, const uint32_t *token_ids,
                 size_t token_count, char *error, size_t error_size) {
    if (!session || !session->engine || !token_ids || !token_count) {
        set_error(error, error_size, "qwen_kv_eval requires a session and "
                  "tokens");
        return 0;
    }
    if (!session->kv && !context_create(session, error, error_size)) return 0;
    qwen_kv_context *kv = session->kv;

    if (token_count > UINT32_MAX ||
        token_count > (size_t)(kv->capacity - kv->length)) {
        set_error(error, error_size,
                  "KV context is full: %u/%u tokens, cannot add %zu "
                  "(raise H3_QWEN_KV_CAPACITY)",
                  kv->length, kv->capacity, token_count);
        return 0;
    }
    for (size_t index = 0; index < token_count; index++) {
        if (token_ids[index] >= QWEN_LM_VOCAB) {
            set_error(error, error_size,
                      "token id %u is outside the vocabulary",
                      token_ids[index]);
            return 0;
        }
    }

    h3_gpu *gpu = kv->gpu;
    uint32_t past = kv->length;
    uint32_t m = (uint32_t)token_count;
    uint32_t total = past + m;
    size_t hidden_elements = (size_t)m * QWEN_LM_HIDDEN;

    eval_scratch s;
    memset(&s, 0, sizeof(s));
    qwen_layer_weights local_weights;
    memset(&local_weights, 0, sizeof(local_weights));
    int have_local = 0;
    int ok = 0;

    if (!qwen_build_rope_tables(m, past, NULL, &s.cosines, &s.sines, error,
                                error_size))
        goto done;

    s.ids = h3_gpu_tensor_from_u32(gpu, token_ids, m);
    s.rope_cos = h3_gpu_tensor_from_f32(gpu, s.cosines,
                                        (size_t)m * QWEN_LM_ROPE_HALF);
    s.rope_sin = h3_gpu_tensor_from_f32(gpu, s.sines,
                                        (size_t)m * QWEN_LM_ROPE_HALF);
    s.hidden = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    s.norm = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    s.query = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_QUERY_DIM);
    s.key_new = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_KV_DIM);
    s.value_new = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_KV_DIM);
    s.attention_heads =
        h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_QUERY_DIM);
    s.attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    s.gate = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_INTERMEDIATE);
    s.up = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_INTERMEDIATE);
    s.mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    s.logits = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_VOCAB);
    s.kv_scratch = malloc((size_t)m * QWEN_LM_KV_DIM * sizeof(*s.kv_scratch));
    s.logits_host = malloc((size_t)m * QWEN_LM_VOCAB * sizeof(*s.logits_host));
    if (!s.ids || !s.rope_cos || !s.rope_sin || !s.hidden || !s.norm ||
        !s.query || !s.key_new || !s.value_new || !s.attention_heads ||
        !s.attention_output || !s.gate || !s.up || !s.mlp_output || !s.logits ||
        !s.kv_scratch || !s.logits_host) {
        set_error(error, error_size, "cannot allocate KV eval scratch: %s",
                  h3_gpu_error(gpu));
        goto done;
    }

    if (!gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size, "embedding begin") ||
        !gpu_ok(gpu, h3_gpu_embedding_bf16(gpu, s.hidden, kv->embed_weight,
                                           s.ids, m, QWEN_LM_VOCAB,
                                           QWEN_LM_HIDDEN),
                error, error_size, "embedding") ||
        !gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size, "embedding submit"))
        goto done;

    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        const qwen_layer_weights *w;
        if (kv->resident_layers) {
            w = &kv->resident_layers[layer];
        } else {
            if (!qwen_layer_weights_load(kv->store, gpu, layer, &local_weights,
                                         error, error_size))
                goto done;
            have_local = 1;
            w = &local_weights;
        }

        int prep_ok =
            gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size, "prep begin") &&
            qwen_layer_prep(gpu, w, m, s.hidden, s.norm, s.query, s.key_new,
                            s.value_new, s.rope_cos, s.rope_sin, layer, error,
                            error_size) &&
            gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size, "prep submit");
        if (!prep_ok) goto done;

        /* Append the new RoPE'd K/V rows to the cache at absolute offset. */
        size_t new_elements = (size_t)m * QWEN_LM_KV_DIM;
        size_t offset = (size_t)past * QWEN_LM_KV_DIM;
        if (!h3_gpu_tensor_read_bf16(s.key_new, s.kv_scratch, new_elements) ||
            !h3_gpu_tensor_write_bf16_range(kv->k_cache[layer], offset,
                                            s.kv_scratch, new_elements) ||
            !h3_gpu_tensor_read_bf16(s.value_new, s.kv_scratch, new_elements) ||
            !h3_gpu_tensor_write_bf16_range(kv->v_cache[layer], offset,
                                            s.kv_scratch, new_elements)) {
            set_error(error, error_size,
                      "cannot append layer %d K/V to the cache", layer);
            goto done;
        }

        int body_ok =
            gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size, "body begin") &&
            gpu_ok(gpu, h3_gpu_gqa_causal_kv_bf16(
                            gpu, s.attention_heads, s.query,
                            kv->k_cache[layer], kv->v_cache[layer], m, total,
                            QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                            QWEN_LM_HEAD_DIM, qwen_lm_attention_scale()),
                   error, error_size, "cached GQA") &&
            qwen_layer_finish(gpu, w, m, s.hidden, s.attention_heads, s.norm,
                              s.attention_output, s.gate, s.up, s.mlp_output,
                              layer, error, error_size) &&
            gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size, "body submit");
        if (have_local) {
            qwen_layer_weights_free(&local_weights);
            have_local = 0;
        }
        if (!body_ok) goto done;
    }

    if (!gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size, "head begin") ||
        !gpu_ok(gpu, h3_gpu_rms_norm_bf16(gpu, s.norm, s.hidden,
                                          kv->final_norm_weight, m,
                                          QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
                error, error_size, "final RMSNorm") ||
        !gpu_ok(gpu, h3_gpu_linear_bf16(gpu, s.logits, s.norm,
                                        kv->lm_head_weight, NULL, m,
                                        QWEN_LM_HIDDEN, QWEN_LM_VOCAB),
                error, error_size, "lm_head") ||
        !gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size, "head submit"))
        goto done;

    if (!h3_gpu_tensor_read_bf16(s.logits, s.logits_host,
                                 (size_t)m * QWEN_LM_VOCAB)) {
        set_error(error, error_size, "cannot read KV logits: %s",
                  h3_gpu_error(gpu));
        goto done;
    }

    if (!kv->logits.values) {
        kv->logits.values = malloc(QWEN_LM_VOCAB * sizeof(*kv->logits.values));
        if (!kv->logits.values) {
            set_error(error, error_size, "out of memory storing KV logits");
            goto done;
        }
    }
    kv->logits.vocab = QWEN_LM_VOCAB;
    {
        const uint16_t *last = s.logits_host + (size_t)(m - 1) * QWEN_LM_VOCAB;
        float best = -INFINITY;
        uint32_t best_index = 0;
        for (size_t index = 0; index < QWEN_LM_VOCAB; index++) {
            float value_f = bf16_to_f32(last[index]);
            kv->logits.values[index] = value_f;
            if (value_f > best) {
                best = value_f;
                best_index = (uint32_t)index;
            }
        }
        kv->logits.argmax_token = best_index;
    }
    kv->have_logits = 1;

    memcpy(kv->history + past, token_ids, m * sizeof(*kv->history));
    kv->length = total;
    ok = 1;

done:
    if (have_local) qwen_layer_weights_free(&local_weights);
    eval_scratch_free(&s);
    return ok;
}

int qwen_kv_rewind(struct qwen_session *session, size_t keep, char *error,
                   size_t error_size) {
    if (!session) {
        set_error(error, error_size, "qwen_kv_rewind requires a session");
        return 0;
    }
    qwen_kv_context *kv = session->kv;
    if (!kv) {
        if (keep != 0) {
            set_error(error, error_size,
                      "cannot rewind to %zu tokens: session has no context",
                      keep);
            return 0;
        }
        return 1;
    }
    if (keep > kv->length) {
        set_error(error, error_size,
                  "cannot rewind to %zu tokens: context holds %u", keep,
                  kv->length);
        return 0;
    }
    kv->length = (uint32_t)keep;
    /* Stale rows beyond `keep` stay in the GPU buffers; the next eval
     * overwrites them at the correct offset. Latest logits no longer describe
     * the truncated context. */
    kv->have_logits = 0;
    return 1;
}

const qwen_logits *qwen_kv_latest_logits(const struct qwen_session *session) {
    if (!session || !session->kv || !session->kv->have_logits) return NULL;
    return &session->kv->logits;
}

size_t qwen_kv_length(const struct qwen_session *session) {
    return (session && session->kv) ? session->kv->length : 0;
}
