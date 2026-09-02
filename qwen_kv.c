#include "qwen_engine_internal.h"
#include "qwen_layers.h"
#include "qwen_policy.h"

#include <math.h>
#include <pthread.h>
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
    qwen_q4_weight lm_head_q4;   /* borrowed with the resident set; may be empty */

    /* Per-layer K/V cache, each [capacity, QWEN_LM_KV_DIM] BF16 in a GPU
     * buffer; only the first `length` rows are live. */
    h3_gpu_tensor *k_cache[QWEN_LM_TOTAL_LAYERS];
    h3_gpu_tensor *v_cache[QWEN_LM_TOTAL_LAYERS];

    uint32_t capacity;
    uint32_t length;

    uint32_t *history; /* [capacity] token ids */

    qwen_logits logits; /* latest next-token logits (last position) */
    int have_logits;

    /* Approach B: when set, the decoder-layer weights, embed / norm / lm_head
     * and the GPU come from the process-wide shared resident set below, which
     * loads them once no matter how many sessions ask. `gpu`, `store`,
     * `embed_weight`, `final_norm_weight`, `lm_head_weight` and
     * `resident_layers` are then borrowed, not owned. */
    int holds_resident;
    const qwen_layer_weights *resident_layers;

    /* AWQ calibration capture (H3_QWEN_AWQ_CALIB=path): accumulates mean |x|
     * per projection input over every token evalled, written on free. */
    qwen_awq_calib *awq_calib;
    char *awq_calib_path;

    /* QINT-012: H3_QWEN_DUMP_L49=path appends the layer-49 hidden state (BF16
     * [m, QWEN_LM_HIDDEN]) after every eval, for drift measurement. */
    char *l49_path;

    /* P7-005: after a multimodal prefill the mRoPE position of the next text
     * token is not `length` (the vision block advances the grid), so decode
     * rope uses `mrope_next` instead. 0 = plain sequential (text-only). */
    uint32_t mrope_next;
    uint32_t mrope_base_len; /* token count right after the multimodal prefill */
    uint32_t mrope_base_pos; /* mrope_next right after that prefill            */
};

/* ---- process-wide shared resident weights (Approach B) ------------------- *
 * One 64-layer set (~62 GB) plus embed / norm / lm_head and a GPU, loaded on
 * the first request and reference-counted. Every resident session borrows it,
 * so N sessions cost one copy, not N. */
static pthread_mutex_t g_resident_lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
    int refcount;
    char *weight_directory;
    h3_gpu *gpu;
    h3_weight_store *store;
    h3_gpu_tensor *embed_weight;
    h3_gpu_tensor *final_norm_weight;
    h3_gpu_tensor *lm_head_weight;
    qwen_q4_weight lm_head_q4;
    qwen_layer_weights layers[QWEN_LM_TOTAL_LAYERS];
} g_resident;

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

static void resident_release(void);

/* Ablation knobs (QINT-016):
 *   H3_QWEN_Q4_BF16_PROJ=kv,down,qo,gateup,mlp,attn  -> those classes BF16
 *   H3_QWEN_Q4_BF16_LAYERS=50-63,0-3                 -> those layers all BF16
 * Returns the projection mask to quantise for `layer` ({q,k,v,o,gate,up,down}
 * bits; 0 = whole layer BF16). */
static int q4_mixed_preset(void) {
    return qwen_decode_policy_current() == QWEN_DECODE_POLICY_MIXED;
}

static uint32_t q4_proj_mask_for_layer(int layer) {
    uint32_t mask = QWEN_Q4_PROJ_ALL;
    /* "mixed" preset (QINT-016 winner): BF16 chat tail (layers 50..63, which
     * never feed H3) + BF16 K/V on the rest; W4 elsewhere. top-1 0.953,
     * KL 0.033 vs 0.894/0.078 for pure W4, at ~0.20 vs 0.16 s/tok. Explicit
     * H3_QWEN_Q4_BF16_* still override. */
    if (q4_mixed_preset()) {
        if (layer >= QWEN_LM_RELEASED_LAYERS) return 0;
        mask &= ~(QWEN_Q4_PROJ_K | QWEN_Q4_PROJ_V);
    }
    const char *proj = getenv("H3_QWEN_Q4_BF16_PROJ");
    if (proj && *proj) {
        struct { const char *name; uint32_t bits; } cls[] = {
            {"q", QWEN_Q4_PROJ_Q}, {"k", QWEN_Q4_PROJ_K}, {"v", QWEN_Q4_PROJ_V},
            {"o", QWEN_Q4_PROJ_O}, {"kv", QWEN_Q4_PROJ_K | QWEN_Q4_PROJ_V},
            {"qo", QWEN_Q4_PROJ_Q | QWEN_Q4_PROJ_O},
            {"gate", QWEN_Q4_PROJ_GATE}, {"up", QWEN_Q4_PROJ_UP},
            {"gateup", QWEN_Q4_PROJ_GATE | QWEN_Q4_PROJ_UP},
            {"down", QWEN_Q4_PROJ_DOWN},
            {"mlp", QWEN_Q4_PROJ_GATE | QWEN_Q4_PROJ_UP | QWEN_Q4_PROJ_DOWN},
            {"attn", QWEN_Q4_PROJ_Q | QWEN_Q4_PROJ_K | QWEN_Q4_PROJ_V |
                     QWEN_Q4_PROJ_O},
        };
        for (size_t i = 0; i < sizeof(cls) / sizeof(cls[0]); i++) {
            const char *p = proj;
            size_t n = strlen(cls[i].name);
            while ((p = strstr(p, cls[i].name))) {
                int lb = p == proj || p[-1] == ',';
                int rb = p[n] == '\0' || p[n] == ',';
                if (lb && rb) { mask &= ~cls[i].bits; break; }
                p += n;
            }
        }
    }
    const char *lr = getenv("H3_QWEN_Q4_BF16_LAYERS");
    for (const char *p = lr; p && *p;) {
        int a = atoi(p), b = a;
        const char *dash = strchr(p, '-');
        const char *comma = strchr(p, ',');
        if (dash && (!comma || dash < comma)) b = atoi(dash + 1);
        if (layer >= a && layer <= b) mask = 0;
        p = comma ? comma + 1 : NULL;
    }
    return mask;
}

void qwen_kv_context_free(qwen_kv_context *kv) {
    if (!kv) return;
    /* K/V caches and history are always this session's own. */
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        h3_gpu_tensor_free(kv->k_cache[layer]);
        h3_gpu_tensor_free(kv->v_cache[layer]);
    }
    free(kv->history);
    qwen_logits_free(&kv->logits);
    if (kv->awq_calib) {
        char error[256];
        error[0] = '\0';
        if (kv->awq_calib_path &&
            qwen_awq_calib_write(kv->awq_calib, kv->awq_calib_path, error,
                                 sizeof(error)))
            fprintf(stderr, "Qwen AWQ calibration written -> %s\n",
                    kv->awq_calib_path);
        else if (kv->awq_calib_path)
            fprintf(stderr, "Qwen AWQ calibration write failed: %s\n", error);
        qwen_awq_calib_free(kv->awq_calib);
        free(kv->awq_calib_path);
    }
    free(kv->l49_path);
    if (kv->holds_resident) {
        /* gpu / store / embed / norm / lm_head / layers are borrowed. */
        resident_release();
    } else {
        h3_gpu_tensor_free(kv->embed_weight);
        h3_gpu_tensor_free(kv->final_norm_weight);
        h3_gpu_tensor_free(kv->lm_head_weight);
        if (kv->gpu) h3_gpu_free(kv->gpu);
        if (kv->store) h3_weight_store_free(kv->store);
    }
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

/* Resident weights are the default. Returns 1 to force resident, -1 to force
 * streaming, 0 for "default" (resident, with a fall-back to streaming if the
 * ~62 GB allocation fails). Session flag wins over the environment. */
static int resident_mode(const struct qwen_session *session) {
    if (session->resident_mode) return session->resident_mode;
    const char *value = getenv("H3_QWEN_RESIDENT");
    if (value && *value) return strcmp(value, "0") == 0 ? -1 : 1;
    return 0;
}

/* Free the shared resident set (must hold g_resident_lock, refcount 0). */
static void resident_teardown_locked(void) {
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++)
        qwen_layer_weights_free(&g_resident.layers[layer]);
    qwen_q4_weight_free(&g_resident.lm_head_q4);
    h3_gpu_tensor_free(g_resident.embed_weight);
    h3_gpu_tensor_free(g_resident.final_norm_weight);
    h3_gpu_tensor_free(g_resident.lm_head_weight);
    if (g_resident.gpu) h3_gpu_free(g_resident.gpu);
    if (g_resident.store) h3_weight_store_free(g_resident.store);
    free(g_resident.weight_directory);
    memset(&g_resident, 0, sizeof(g_resident));
}

/* Acquire a reference to the process-wide resident weight set, loading it
 * (~62 GB, all 64 decoder layers + embed / norm / lm_head) on first use.
 * Returns 0 and fills `error` on failure. */
static int resident_acquire(const char *weight_directory,
                            const char *shader_source_path, char *error,
                            size_t error_size) {
    pthread_mutex_lock(&g_resident_lock);
    if (g_resident.refcount > 0) {
        if (strcmp(g_resident.weight_directory, weight_directory) != 0) {
            set_error(error, error_size,
                      "resident weights already loaded for a different model "
                      "(%s); cannot also hold %s",
                      g_resident.weight_directory, weight_directory);
            pthread_mutex_unlock(&g_resident_lock);
            return 0;
        }
        g_resident.refcount++;
        pthread_mutex_unlock(&g_resident_lock);
        return 1;
    }

    g_resident.weight_directory = strdup(weight_directory);
    g_resident.store =
        h3_weight_store_open(weight_directory, error, error_size);
    g_resident.gpu = g_resident.store
        ? h3_gpu_create(shader_source_path, error, error_size)
        : NULL;
    if (!g_resident.weight_directory || !g_resident.store || !g_resident.gpu) {
        resident_teardown_locked();
        pthread_mutex_unlock(&g_resident_lock);
        return 0;
    }
    h3_gpu_profile_set_label(g_resident.gpu, "Qwen resident weights");

    uint64_t embed_shape[] = {QWEN_LM_VOCAB, QWEN_LM_HIDDEN};
    uint64_t norm_shape[] = {QWEN_LM_HIDDEN};
    g_resident.embed_weight = h3_weight_load_bf16(
        g_resident.store, g_resident.gpu,
        "model.language_model.embed_tokens.weight", 2, embed_shape, error,
        error_size);
    g_resident.final_norm_weight = h3_weight_load_bf16(
        g_resident.store, g_resident.gpu, "model.language_model.norm.weight", 1,
        norm_shape, error, error_size);
    g_resident.lm_head_weight =
        h3_weight_load_bf16(g_resident.store, g_resident.gpu, "lm_head.weight",
                            2, embed_shape, error, error_size);
    if (!g_resident.embed_weight || !g_resident.final_norm_weight ||
        !g_resident.lm_head_weight) {
        resident_teardown_locked();
        pthread_mutex_unlock(&g_resident_lock);
        return 0;
    }
    for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
        if (!qwen_layer_weights_load(g_resident.store, g_resident.gpu, layer,
                                     &g_resident.layers[layer], error,
                                     error_size)) {
            resident_teardown_locked();
            pthread_mutex_unlock(&g_resident_lock);
            return 0;
        }
        if ((layer + 1) % 8 == 0 || layer + 1 == QWEN_LM_TOTAL_LAYERS)
            fprintf(stderr, "Qwen resident weights: %d/%d layers\n", layer + 1,
                    QWEN_LM_TOTAL_LAYERS);
    }

    /* Chat-speedup step #2: quantise the resident projections + lm_head to
     * group-wise INT4 for the rows==1 decode GEMV. Prefill (rows>1), the
     * streaming session and the H3 path all stay BF16. A quantiser OOM is not
     * fatal -- keep the BF16 resident set and note it. */
    if (qwen_q4_enabled()) {
        char q4_error[256];
        q4_error[0] = '\0';
        const char *awq_path = getenv("H3_QWEN_Q4_AWQ");
        if (awq_path && !*awq_path) awq_path = NULL;
        int q4_ok = 1, q4_bf16_layers = 0;
        for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS && q4_ok; layer++) {
            uint32_t mask = q4_proj_mask_for_layer(layer);
            if (mask != QWEN_Q4_PROJ_ALL) q4_bf16_layers++;
            q4_ok = qwen_layer_weights_quantize(&g_resident.layers[layer],
                                                g_resident.gpu, layer, awq_path,
                                                mask, q4_error,
                                                sizeof(q4_error));
        }
        if (q4_bf16_layers)
            fprintf(stderr, "Qwen resident weights: INT4 ablation active "
                    "(%d layer(s) partially/fully BF16)\n", q4_bf16_layers);
        const char *q4_head = getenv("H3_QWEN_Q4_HEAD");
        if (q4_ok && q4_head && q4_head[0] == '1')
            q4_ok = qwen_q4_quantize(g_resident.gpu, g_resident.lm_head_weight,
                                     QWEN_LM_VOCAB, QWEN_LM_HIDDEN,
                                     &g_resident.lm_head_q4, q4_error,
                                     sizeof(q4_error));
        if (q4_ok) {
            fprintf(stderr, "Qwen resident weights: INT4 decode copy ready "
                    "(policy %s, group %u%s)\n",
                    qwen_decode_policy_name(qwen_decode_policy_current()),
                    QWEN_Q4_GROUP, awq_path ? ", AWQ" : ", RTN");
        } else {
            fprintf(stderr, "Qwen resident weights: INT4 quantise skipped "
                    "(%s); decode stays BF16\n", q4_error);
            for (int layer = 0; layer < QWEN_LM_TOTAL_LAYERS; layer++) {
                qwen_q4_weight_free(&g_resident.layers[layer].q4_query);
                qwen_q4_weight_free(&g_resident.layers[layer].q4_key);
                qwen_q4_weight_free(&g_resident.layers[layer].q4_value);
                qwen_q4_weight_free(
                    &g_resident.layers[layer].q4_attention_output);
                qwen_q4_weight_free(&g_resident.layers[layer].q4_gate);
                qwen_q4_weight_free(&g_resident.layers[layer].q4_up);
                qwen_q4_weight_free(&g_resident.layers[layer].q4_down);
                g_resident.layers[layer].has_q4 = 0;
            }
            qwen_q4_weight_free(&g_resident.lm_head_q4);
        }
    }

    g_resident.refcount = 1;
    pthread_mutex_unlock(&g_resident_lock);
    return 1;
}

static void resident_release(void) {
    pthread_mutex_lock(&g_resident_lock);
    if (g_resident.refcount > 0 && --g_resident.refcount == 0)
        resident_teardown_locked();
    pthread_mutex_unlock(&g_resident_lock);
}

static int context_create(struct qwen_session *session, char *error,
                          size_t error_size) {
    qwen_kv_context *kv = calloc(1, sizeof(*kv));
    if (!kv) {
        set_error(error, error_size, "out of memory allocating KV context");
        return 0;
    }
    kv->capacity = kv_capacity_from_env();

    const char *calib_path = getenv("H3_QWEN_AWQ_CALIB");
    if (calib_path && *calib_path) {
        kv->awq_calib = qwen_awq_calib_new();
        kv->awq_calib_path = strdup(calib_path);
        if (!kv->awq_calib || !kv->awq_calib_path) {
            set_error(error, error_size, "out of memory for AWQ calibration");
            qwen_kv_context_free(kv);
            return 0;
        }
    }
    const char *l49 = getenv("H3_QWEN_DUMP_L49");
    if (l49 && *l49 && !(kv->l49_path = strdup(l49))) {
        set_error(error, error_size, "out of memory");
        qwen_kv_context_free(kv);
        return 0;
    }

    int mode = resident_mode(session);
    if (mode >= 0 &&
        resident_acquire(session->engine->weight_directory,
                         session->engine->shader_source_path, error,
                         error_size)) {
        kv->holds_resident = 1;
        kv->gpu = g_resident.gpu;                 /* borrowed */
        kv->store = NULL;                         /* not needed */
        kv->embed_weight = g_resident.embed_weight;
        kv->final_norm_weight = g_resident.final_norm_weight;
        kv->lm_head_weight = g_resident.lm_head_weight;
        kv->lm_head_q4 = g_resident.lm_head_q4;   /* borrowed; may be empty */
        kv->resident_layers = g_resident.layers;
    } else if (mode > 0) {
        /* Resident was explicitly required; do not silently stream. */
        qwen_kv_context_free(kv);
        return 0;
    } else {
        if (mode == 0)
            fprintf(stderr, "Qwen: resident weights unavailable (%s); "
                    "streaming per eval\n", error);
        if (error && error_size) error[0] = '\0';
        kv->store = h3_weight_store_open(session->engine->weight_directory,
                                        error, error_size);
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
            kv->store, kv->gpu, "model.language_model.norm.weight", 1,
            norm_shape, error, error_size);
        kv->lm_head_weight = h3_weight_load_bf16(
            kv->store, kv->gpu, "lm_head.weight", 2, embed_shape, error,
            error_size);
        if (!kv->embed_weight || !kv->final_norm_weight ||
            !kv->lm_head_weight) {
            qwen_kv_context_free(kv);
            return 0;
        }
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
    h3_gpu_tensor *l49_dump;   /* QINT-012: layer-49 hidden snapshot, or NULL */
    h3_gpu_tensor *deepstack[3]; /* P7-005: multimodal prefill only, else NULL */
    h3_gpu_tensor *logits;
    float *cosines;
    float *sines;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
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
    h3_gpu_tensor_free(s->l49_dump);
    h3_gpu_tensor_free(s->deepstack[0]);
    h3_gpu_tensor_free(s->deepstack[1]);
    h3_gpu_tensor_free(s->deepstack[2]);
    h3_gpu_tensor_free(s->logits);
    h3_gpu_tensor_free(s->rope_cos);
    h3_gpu_tensor_free(s->rope_sin);
    free(s->cosines);
    free(s->sines);
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

static int kv_eval(struct qwen_session *session, const uint32_t *token_ids,
                   size_t token_count, const qwen_input *mm,
                   qwen_eval_kind kind, qwen_verify_result *verify, char *error,
                   size_t error_size) {
    if (!session || !session->engine || !token_ids || !token_count) {
        set_error(error, error_size, "qwen_kv_eval requires a session and "
                  "tokens");
        return 0;
    }
    if (!session->kv && !context_create(session, error, error_size)) return 0;
    qwen_kv_context *kv = session->kv;
    int mm_prefill = mm && mm->vision_span_count && kv->length == 0;

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
    uint16_t *calib_buf = NULL;

    /* mRoPE for a multimodal prefill; sequential for text decode, continuing
     * from mrope_next when a multimodal prefill advanced the grid. */
    if (mm_prefill) {
        if (!qwen_build_rope_tables(m, 0, mm->position_ids, &s.cosines,
                                    &s.sines, error, error_size))
            goto done;
    } else {
        uint32_t base = kv->mrope_next ? kv->mrope_next : past;
        if (!qwen_build_rope_tables(m, base, NULL, &s.cosines, &s.sines, error,
                                    error_size))
            goto done;
    }

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
    if (kv->l49_path) {
        s.l49_dump = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
        if (!s.l49_dump) {
            set_error(error, error_size, "cannot allocate L49 dump buffer");
            goto done;
        }
    }
    s.logits = h3_gpu_tensor_new_bf16(gpu, (size_t)m * QWEN_LM_VOCAB);
    s.logits_host = malloc((size_t)m * QWEN_LM_VOCAB * sizeof(*s.logits_host));
    if (!s.ids || !s.rope_cos || !s.rope_sin || !s.hidden || !s.norm ||
        !s.query || !s.key_new || !s.value_new || !s.attention_heads ||
        !s.attention_output || !s.gate || !s.up || !s.mlp_output || !s.logits ||
        !s.logits_host) {
        set_error(error, error_size, "cannot allocate KV eval scratch: %s",
                  h3_gpu_error(gpu));
        goto done;
    }

    /* Multimodal prefill: pack the three deepstack additions as full [m,HIDDEN]
     * tensors (zero except the vision-span rows), added to the residual stream
     * after decoder layers 0/1/2 -- exactly as h3_text_encoder.c does. */
    if (mm_prefill) {
        for (int layer = 0; layer < 3; layer++) {
            uint16_t *packed = calloc(hidden_elements, sizeof(*packed));
            if (!packed) {
                set_error(error, error_size, "out of memory for deepstack");
                goto done;
            }
            for (size_t v = 0; v < mm->vision_span_count; v++) {
                const qwen_vision_span *sp = &mm->vision_spans[v];
                memcpy(packed + sp->start * QWEN_LM_HIDDEN, sp->deepstack[layer],
                       sp->tokens * QWEN_LM_HIDDEN * sizeof(*packed));
            }
            s.deepstack[layer] =
                h3_gpu_tensor_from_bf16(gpu, packed, hidden_elements);
            free(packed);
            if (!s.deepstack[layer]) {
                set_error(error, error_size, "cannot allocate deepstack: %s",
                          h3_gpu_error(gpu));
                goto done;
            }
        }
    }

    /* The resident path encodes embedding + all 64 layers + head into a single
     * command buffer and submits once: with INT4/GEMV linears the per-layer
     * `h3_gpu_submit` turnaround was the dominant cost. The streaming path
     * keeps a submit per layer so `local_weights` can be freed between layers
     * (holding all 64 at once would defeat streaming). */
    int fused = kv->resident_layers != NULL && !kv->awq_calib;
    if (kv->awq_calib) {
        calib_buf = malloc((size_t)m * QWEN_LM_INTERMEDIATE *
                           sizeof(*calib_buf));
        if (!calib_buf) {
            set_error(error, error_size, "out of memory for AWQ readback");
            goto done;
        }
    }
#define STAGE_BEGIN(label) (fused ? 1 : \
    gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size, label))
#define STAGE_SUBMIT(label) (fused ? 1 : \
    gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size, label))

    if (fused && !gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size,
                         "fused forward begin"))
        goto done;

    h3_gpu_tensor *mm_emb[8] = {0};
    if (!STAGE_BEGIN("embedding begin") ||
        !gpu_ok(gpu, h3_gpu_embedding_bf16(gpu, s.hidden, kv->embed_weight,
                                           s.ids, m, QWEN_LM_VOCAB,
                                           QWEN_LM_HIDDEN),
                error, error_size, "embedding"))
        goto done;
    if (mm_prefill) {
        int splice_ok = 1;
        for (size_t v = 0; v < mm->vision_span_count && v < 8; v++) {
            const qwen_vision_span *sp = &mm->vision_spans[v];
            mm_emb[v] = h3_gpu_tensor_from_bf16(
                gpu, sp->embeddings, sp->tokens * QWEN_LM_HIDDEN);
            if (!mm_emb[v] ||
                !gpu_ok(gpu, h3_gpu_copy_bf16(gpu, s.hidden,
                                              sp->start * QWEN_LM_HIDDEN,
                                              mm_emb[v], 0,
                                              sp->tokens * QWEN_LM_HIDDEN),
                        error, error_size, "vision embedding splice")) {
                splice_ok = 0;
                break;
            }
        }
        if (!splice_ok) {
            for (int v = 0; v < 8; v++) h3_gpu_tensor_free(mm_emb[v]);
            goto done;
        }
    }
    if (!STAGE_SUBMIT("embedding submit")) {
        for (int v = 0; v < 8; v++) h3_gpu_tensor_free(mm_emb[v]);
        goto done;
    }
    for (int v = 0; v < 8; v++) h3_gpu_tensor_free(mm_emb[v]);

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

        /* prep -> append the new RoPE'd K/V rows to the GPU cache with a blit
         * (no host round-trip) -> cached GQA over the whole cache -> finish.
         * Metal hazard-tracks the blit against the following read. */
        size_t new_elements = (size_t)m * QWEN_LM_KV_DIM;
        size_t offset = (size_t)past * QWEN_LM_KV_DIM;
        int body_ok;
        if (kv->awq_calib) {
            /* Split the layer so the q/k/v-input activations (s.norm after
             * prep) can be read before qwen_layer_finish overwrites s.norm. */
            body_ok =
                gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size,
                       "calib prep begin") &&
                qwen_layer_prep(gpu, w, m, kind, s.hidden, s.norm, s.query, s.key_new,
                                s.value_new, s.rope_cos, s.rope_sin, layer,
                                error, error_size) &&
                gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size,
                       "calib prep submit");
            if (body_ok && h3_gpu_tensor_read_bf16(
                    s.norm, calib_buf, (size_t)m * QWEN_LM_HIDDEN))
                qwen_awq_calib_add(kv->awq_calib, layer, QWEN_AWQ_QKV_IN,
                                   calib_buf, m, QWEN_LM_HIDDEN);
            body_ok = body_ok &&
                gpu_ok(gpu, h3_gpu_begin(gpu), error, error_size,
                       "calib body begin") &&
                gpu_ok(gpu, h3_gpu_copy_bf16(gpu, kv->k_cache[layer], offset,
                                             s.key_new, 0, new_elements),
                       error, error_size, "K cache append") &&
                gpu_ok(gpu, h3_gpu_copy_bf16(gpu, kv->v_cache[layer], offset,
                                             s.value_new, 0, new_elements),
                       error, error_size, "V cache append") &&
                gpu_ok(gpu, h3_gpu_gqa_causal_kv_bf16(
                                gpu, s.attention_heads, s.query,
                                kv->k_cache[layer], kv->v_cache[layer], m, total,
                                QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                                QWEN_LM_HEAD_DIM, qwen_lm_attention_scale()),
                       error, error_size, "cached GQA") &&
                qwen_layer_finish(gpu, w, m, kind, s.hidden, s.attention_heads, s.norm,
                                  s.attention_output, s.gate, s.up,
                                  s.mlp_output, layer, error, error_size) &&
                gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size,
                       "calib body submit");
            if (body_ok) {
                if (h3_gpu_tensor_read_bf16(s.attention_heads, calib_buf,
                                            (size_t)m * QWEN_LM_QUERY_DIM))
                    qwen_awq_calib_add(kv->awq_calib, layer, QWEN_AWQ_O_IN,
                                       calib_buf, m, QWEN_LM_QUERY_DIM);
                if (h3_gpu_tensor_read_bf16(s.norm, calib_buf,
                                            (size_t)m * QWEN_LM_HIDDEN))
                    qwen_awq_calib_add(kv->awq_calib, layer, QWEN_AWQ_MLP_IN,
                                       calib_buf, m, QWEN_LM_HIDDEN);
                if (h3_gpu_tensor_read_bf16(s.gate, calib_buf,
                                            (size_t)m * QWEN_LM_INTERMEDIATE))
                    qwen_awq_calib_add(kv->awq_calib, layer, QWEN_AWQ_DOWN_IN,
                                       calib_buf, m, QWEN_LM_INTERMEDIATE);
            }
        } else {
            body_ok =
                STAGE_BEGIN("layer begin") &&
                qwen_layer_prep(gpu, w, m, kind, s.hidden, s.norm, s.query, s.key_new,
                                s.value_new, s.rope_cos, s.rope_sin, layer,
                                error, error_size) &&
                gpu_ok(gpu, h3_gpu_copy_bf16(gpu, kv->k_cache[layer], offset,
                                             s.key_new, 0, new_elements),
                       error, error_size, "K cache append") &&
                gpu_ok(gpu, h3_gpu_copy_bf16(gpu, kv->v_cache[layer], offset,
                                             s.value_new, 0, new_elements),
                       error, error_size, "V cache append") &&
                gpu_ok(gpu, h3_gpu_gqa_causal_kv_bf16(
                                gpu, s.attention_heads, s.query,
                                kv->k_cache[layer], kv->v_cache[layer], m, total,
                                QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                                QWEN_LM_HEAD_DIM, qwen_lm_attention_scale()),
                       error, error_size, "cached GQA") &&
                qwen_layer_finish(gpu, w, m, kind, s.hidden, s.attention_heads, s.norm,
                                  s.attention_output, s.gate, s.up,
                                  s.mlp_output, layer, error, error_size) &&
                (layer >= 3 || !s.deepstack[layer] ||
                 gpu_ok(gpu, h3_gpu_add_bf16(gpu, s.hidden, s.hidden,
                                             s.deepstack[layer],
                                             (uint32_t)hidden_elements),
                        error, error_size, "deepstack residual")) &&
                (layer != QWEN_LM_RELEASED_LAYERS - 1 || !s.l49_dump ||
                 gpu_ok(gpu, h3_gpu_copy_bf16(gpu, s.l49_dump, 0, s.hidden, 0,
                                              hidden_elements),
                        error, error_size, "L49 snapshot")) &&
                STAGE_SUBMIT("layer submit");
        }
        if (have_local) {
            qwen_layer_weights_free(&local_weights);
            have_local = 0;
        }
        if (!body_ok) goto done;
    }

    /* DECODE with a resident INT4 lm_head goes through the INT4 GEMV; PREFILL,
     * VERIFY (needs the full [m,vocab] readback) and the streaming session
     * stay on the BF16 tiled path. */
    int use_q4_head = kind == QWEN_EVAL_DECODE && m == 1 &&
                      kv->lm_head_q4.packed != NULL;
    if (!STAGE_BEGIN("head begin") ||
        !gpu_ok(gpu, h3_gpu_rms_norm_bf16(gpu, s.norm, s.hidden,
                                          kv->final_norm_weight, m,
                                          QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
                error, error_size, "final RMSNorm"))
        goto done;
    if (use_q4_head &&
        h3_gpu_linear_q4_gemv(gpu, s.logits, s.norm, kv->lm_head_q4.packed,
                              kv->lm_head_q4.scales, kv->lm_head_q4.awq_inv_scale,
                              NULL, QWEN_LM_HIDDEN, QWEN_LM_VOCAB,
                              QWEN_Q4_GROUP)) {
        /* done */
    } else if (!gpu_ok(gpu, h3_gpu_linear_bf16(gpu, s.logits, s.norm,
                                               kv->lm_head_weight, NULL, m,
                                               QWEN_LM_HIDDEN, QWEN_LM_VOCAB),
                       error, error_size, "lm_head")) {
        goto done;
    }
    if (!STAGE_SUBMIT("head submit"))
        goto done;
    if (fused && !gpu_ok(gpu, h3_gpu_submit(gpu), error, error_size,
                         "fused forward submit"))
        goto done;
#undef STAGE_BEGIN
#undef STAGE_SUBMIT

    if (!h3_gpu_tensor_read_bf16(s.logits, s.logits_host,
                                 (size_t)m * QWEN_LM_VOCAB)) {
        set_error(error, error_size, "cannot read KV logits: %s",
                  h3_gpu_error(gpu));
        goto done;
    }

    if (s.l49_dump && kv->l49_path) {
        uint16_t *snap = malloc(hidden_elements * sizeof(*snap));
        FILE *lf = snap ? fopen(kv->l49_path, "ab") : NULL;
        int wrote = snap && lf &&
                    h3_gpu_tensor_read_bf16(s.l49_dump, snap, hidden_elements) &&
                    fwrite(snap, sizeof(*snap), hidden_elements, lf) ==
                        hidden_elements;
        if (lf) fclose(lf);
        free(snap);
        if (!wrote) {
            set_error(error, error_size, "cannot append L49 dump to %s",
                      kv->l49_path);
            goto done;
        }
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

    /* VERIFY: per-row top1/top2 over the whole [m, vocab] readback. Row r's
     * logits predict the token after block[0..r]. */
    if (kind == QWEN_EVAL_VERIFY && verify) {
        memset(verify, 0, sizeof(*verify));
        verify->rows = m;
        for (uint32_t r = 0; r < m && r < QWEN_VERIFY_MAX; r++) {
            const uint16_t *row = s.logits_host + (size_t)r * QWEN_LM_VOCAB;
            float b1 = -INFINITY, b2 = -INFINITY;
            uint32_t i1 = 0, i2 = 0;
            for (size_t index = 0; index < QWEN_LM_VOCAB; index++) {
                float v = bf16_to_f32(row[index]);
                if (v > b1) {
                    b2 = b1; i2 = i1;
                    b1 = v; i1 = (uint32_t)index;
                } else if (v > b2) {
                    b2 = v; i2 = (uint32_t)index;
                }
            }
            verify->top1[r] = i1;
            verify->top2[r] = i2;
            verify->top1_logit[r] = b1;
            verify->top2_logit[r] = b2;
            verify->margin[r] = b1 - b2;
        }
    }

    memcpy(kv->history + past, token_ids, m * sizeof(*kv->history));
    kv->length = total;
    if (mm_prefill) {
        kv->mrope_base_len = total;
        kv->mrope_next = mm->position_ids[m - 1] + 1;
        kv->mrope_base_pos = kv->mrope_next;
    } else if (kv->mrope_next) {
        kv->mrope_next += m;
    }
    ok = 1;

done:
    if (have_local) qwen_layer_weights_free(&local_weights);
    free(calib_buf);
    eval_scratch_free(&s);
    return ok;
}

int qwen_kv_eval(struct qwen_session *session, const uint32_t *token_ids,
                 size_t token_count, char *error, size_t error_size) {
    /* PREFILL for the first eval or any multi-token chunk; DECODE for a single
     * incremental token. VERIFY is never inferred from the token count -- it
     * only comes from qwen_kv_eval_verify_block(). */
    qwen_eval_kind kind =
        (session && session->kv && session->kv->length > 0 && token_count == 1)
            ? QWEN_EVAL_DECODE
            : QWEN_EVAL_PREFILL;
    return kv_eval(session, token_ids, token_count, NULL, kind, NULL, error,
                   error_size);
}

int qwen_kv_eval_verify_block(struct qwen_session *session,
                              const uint32_t *block, size_t block_count,
                              qwen_verify_result *result, char *error,
                              size_t error_size) {
    if (!session || !block || !result) {
        set_error(error, error_size,
                  "qwen_kv_eval_verify_block requires session, block, result");
        return 0;
    }
    /* The W4 decode-batch kernel currently handles rows 2..5 (H3_GEMVB_MAXM);
     * a wider block would silently fall back to BF16 projections and break the
     * "same weights as scalar decode" contract, so cap it here. */
    if (block_count < 2 || block_count > 5) {
        set_error(error, error_size,
                  "verify block must be 2..5 tokens (got %zu)", block_count);
        return 0;
    }
    if (!session->kv || session->kv->length == 0) {
        set_error(error, error_size,
                  "verify block needs an established context (prefill first)");
        return 0;
    }
    return kv_eval(session, block, block_count, NULL, QWEN_EVAL_VERIFY, result,
                   error, error_size);
}

int qwen_kv_eval_multimodal(struct qwen_session *session,
                            const qwen_input *input, char *error,
                            size_t error_size) {
    if (!session || !input || !input->token_ids || !input->token_count) {
        set_error(error, error_size,
                  "qwen_kv_eval_multimodal requires a session and input");
        return 0;
    }
    if (!input->vision_span_count || !input->vision_spans ||
        !input->position_ids) {
        set_error(error, error_size,
                  "multimodal eval needs vision_spans and position_ids");
        return 0;
    }
    if (session->kv && session->kv->length != 0) {
        set_error(error, error_size,
                  "multimodal eval must be the first eval on a session");
        return 0;
    }
    return kv_eval(session, input->token_ids, input->token_count, input,
                   QWEN_EVAL_PREFILL, NULL, error, error_size);
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
    /* mRoPE: after a multimodal prefill the decode position is
     * mrope_base_pos + (kept decode tokens). Rewinding into the prompt itself
     * (keep < mrope_base_len) is unsupported -- reject it *before* mutating any
     * state so a refused rewind is a no-op. keep == 0 clears everything. */
    if (keep != 0 && kv->mrope_base_len && keep < kv->mrope_base_len) {
        set_error(error, error_size,
                  "cannot rewind into a multimodal prompt (keep %zu < %u)",
                  keep, kv->mrope_base_len);
        return 0;
    }
    kv->length = (uint32_t)keep;
    if (keep == 0) {
        kv->mrope_next = kv->mrope_base_len = kv->mrope_base_pos = 0;
    } else if (kv->mrope_base_len) {
        kv->mrope_next =
            kv->mrope_base_pos + ((uint32_t)keep - kv->mrope_base_len);
    }
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

const uint32_t *qwen_kv_history(const struct qwen_session *session,
                               size_t *length_out) {
    if (!session || !session->kv || session->kv->length == 0) {
        if (length_out) *length_out = 0;
        return NULL;
    }
    if (length_out) *length_out = session->kv->length;
    return session->kv->history;
}
