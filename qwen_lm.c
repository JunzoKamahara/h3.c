#include "qwen_engine_internal.h"

#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 1 -- Qwen3-VL Chat LLM tail.
 *
 * Continues from the canonical layer-49 intermediate state (qwen_engine.h)
 * through decoder layers 50..63, the final language-model RMSNorm and lm_head,
 * and returns next-token logits for the last prompt position with a CPU argmax.
 *
 * The per-layer recipe and the mRoPE table construction mirror
 * h3_text_encoder.c (encode_layer / text_encode_bf16_impl) exactly; layers
 * 50..63 use the identical operation sequence, epsilon and theta as layers
 * 0..49. `make phase0-parity` guards the 0..49 half against drift; the Phase 1
 * test guards this tail's boundary consistency (forward_full ==
 * continue_from_intermediate(get_h3_conditioning)) and determinism.
 *
 * No KV cache: this is a full-prompt forward (spec section 11). Weights for the
 * 14 tail layers are streamed one layer at a time -- slow but simple, which is
 * the Phase 1 remit. */

enum {
    LM_TOTAL_LAYERS = 64,
    LM_TAIL_START = 50,
    LM_TAIL_LAYERS = LM_TOTAL_LAYERS - LM_TAIL_START,
    LM_VOCAB = 151936,
    LM_HIDDEN = 5120,
    LM_INTERMEDIATE = 25600,
    LM_QUERY_HEADS = 64,
    LM_KV_HEADS = 8,
    LM_HEAD_DIM = 128,
    LM_QUERY_DIM = LM_QUERY_HEADS * LM_HEAD_DIM,
    LM_KV_DIM = LM_KV_HEADS * LM_HEAD_DIM,
    LM_ROPE_HALF = LM_HEAD_DIM / 2
};

static const float LM_RMS_EPSILON = 1e-6f;
static const float LM_ROPE_THETA = 5000000.0f;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Round to BF16, matching h3_text_encoder.c's mRoPE table pinning. */
static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *post_norm;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *down;
} lm_layer_weights;

static void lm_layer_weights_free(lm_layer_weights *weights) {
    h3_gpu_tensor_free(weights->input_norm);
    h3_gpu_tensor_free(weights->query);
    h3_gpu_tensor_free(weights->key);
    h3_gpu_tensor_free(weights->value);
    h3_gpu_tensor_free(weights->query_norm);
    h3_gpu_tensor_free(weights->key_norm);
    h3_gpu_tensor_free(weights->attention_output);
    h3_gpu_tensor_free(weights->post_norm);
    h3_gpu_tensor_free(weights->gate);
    h3_gpu_tensor_free(weights->up);
    h3_gpu_tensor_free(weights->down);
    memset(weights, 0, sizeof(*weights));
}

static int lm_layer_weights_load(const h3_weight_store *store, h3_gpu *gpu,
                                 int layer, lm_layer_weights *weights,
                                 char *error, size_t error_size) {
    memset(weights, 0, sizeof(*weights));
    char prefix[96];
    if (snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d.",
                 layer) < 0) {
        fail(error, error_size, "cannot format Qwen layer name");
        return 0;
    }
#define LOAD(field, suffix, ndim, ...) do {                                     \
    char name[192];                                                            \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    uint64_t shape[] = {__VA_ARGS__};                                          \
    weights->field = h3_weight_load_bf16(store, gpu, name, ndim, shape,        \
                                         error, error_size);                   \
    if (!weights->field) { lm_layer_weights_free(weights); return 0; }         \
} while (0)
    LOAD(input_norm, "input_layernorm.weight", 1, LM_HIDDEN);
    LOAD(query, "self_attn.q_proj.weight", 2, LM_QUERY_DIM, LM_HIDDEN);
    LOAD(key, "self_attn.k_proj.weight", 2, LM_KV_DIM, LM_HIDDEN);
    LOAD(value, "self_attn.v_proj.weight", 2, LM_KV_DIM, LM_HIDDEN);
    LOAD(query_norm, "self_attn.q_norm.weight", 1, LM_HEAD_DIM);
    LOAD(key_norm, "self_attn.k_norm.weight", 1, LM_HEAD_DIM);
    LOAD(attention_output, "self_attn.o_proj.weight", 2, LM_HIDDEN,
         LM_QUERY_DIM);
    LOAD(post_norm, "post_attention_layernorm.weight", 1, LM_HIDDEN);
    LOAD(gate, "mlp.gate_proj.weight", 2, LM_INTERMEDIATE, LM_HIDDEN);
    LOAD(up, "mlp.up_proj.weight", 2, LM_INTERMEDIATE, LM_HIDDEN);
    LOAD(down, "mlp.down_proj.weight", 2, LM_HIDDEN, LM_INTERMEDIATE);
#undef LOAD
    return 1;
}

static int op(h3_gpu *gpu, int ok, char *error, size_t error_size,
              const char *what, int layer) {
    if (ok) return 1;
    if (layer >= 0)
        fail(error, error_size, "Qwen layer %d %s failed: %s", layer, what,
             h3_gpu_error(gpu));
    else
        fail(error, error_size, "Qwen LM %s failed: %s", what,
             h3_gpu_error(gpu));
    return 0;
}

/* One decoder layer: identical to h3_text_encoder.c encode_layer(). */
static int lm_encode_layer(h3_gpu *gpu, const lm_layer_weights *w,
                           uint32_t tokens, h3_gpu_tensor *hidden,
                           h3_gpu_tensor *norm, h3_gpu_tensor *query,
                           h3_gpu_tensor *key, h3_gpu_tensor *value,
                           h3_gpu_tensor *attention_heads,
                           h3_gpu_tensor *attention_output,
                           h3_gpu_tensor *gate, h3_gpu_tensor *up,
                           h3_gpu_tensor *mlp_output, h3_gpu_tensor *rope_cos,
                           h3_gpu_tensor *rope_sin, int layer, char *error,
                           size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!op(gpu, (call), error, error_size, label, layer)) return 0;           \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->input_norm, tokens, LM_HIDDEN,
                            LM_RMS_EPSILON), "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, query, norm, w->query, NULL, tokens, LM_HIDDEN,
                          LM_QUERY_DIM), "query projection");
    OP(h3_gpu_linear_bf16(gpu, key, norm, w->key, NULL, tokens, LM_HIDDEN,
                          LM_KV_DIM), "key projection");
    OP(h3_gpu_linear_bf16(gpu, value, norm, w->value, NULL, tokens, LM_HIDDEN,
                          LM_KV_DIM), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, query, w->query_norm, tokens,
                                 LM_QUERY_HEADS, LM_HEAD_DIM, LM_RMS_EPSILON),
       "query RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, key, w->key_norm, tokens, LM_KV_HEADS,
                                 LM_HEAD_DIM, LM_RMS_EPSILON), "key RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, tokens,
                             LM_QUERY_HEADS, LM_KV_HEADS, LM_HEAD_DIM), "RoPE");
    OP(h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key, value, tokens,
                              LM_QUERY_HEADS, LM_KV_HEADS, LM_HEAD_DIM,
                              1.0f / sqrtf((float)LM_HEAD_DIM)), "causal GQA");
    OP(h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                          w->attention_output, NULL, tokens, LM_QUERY_DIM,
                          LM_HIDDEN), "attention output projection");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                       tokens * LM_HIDDEN), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->post_norm, tokens, LM_HIDDEN,
                            LM_RMS_EPSILON), "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, gate, norm, w->gate, NULL, tokens, LM_HIDDEN,
                          LM_INTERMEDIATE), "MLP gate");
    OP(h3_gpu_linear_bf16(gpu, up, norm, w->up, NULL, tokens, LM_HIDDEN,
                          LM_INTERMEDIATE), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up, tokens * LM_INTERMEDIATE),
       "fused SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, mlp_output, gate, w->down, NULL, tokens,
                          LM_INTERMEDIATE, LM_HIDDEN), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output, tokens * LM_HIDDEN),
       "MLP residual");
#undef OP
    return 1;
}

/* mRoPE cos/sin tables, identical construction to text_encode_bf16_impl(). */
static int lm_build_rope(h3_gpu *gpu, size_t tokens,
                         const uint32_t *position_ids, h3_gpu_tensor **rope_cos,
                         h3_gpu_tensor **rope_sin, char *error,
                         size_t error_size) {
    float *cosines = malloc(tokens * LM_ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(tokens * LM_ROPE_HALF * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines);
        free(sines);
        fail(error, error_size, "out of memory allocating Qwen RoPE tables");
        return 0;
    }
    float inverse_frequency[LM_ROPE_HALF];
    for (size_t index = 0; index < LM_ROPE_HALF; index++)
        inverse_frequency[index] =
            1.0f / powf(LM_ROPE_THETA,
                        (float)(index * 2) / (float)LM_HEAD_DIM);
    for (size_t position = 0; position < tokens; position++) {
        for (size_t index = 0; index < LM_ROPE_HALF; index++) {
            size_t axis = 0;
            if (position_ids && index < 60 && index % 3 == 1) axis = 1;
            else if (position_ids && index < 60 && index % 3 == 2) axis = 2;
            float coordinate = position_ids
                ? (float)position_ids[axis * tokens + position]
                : (float)position;
            float angle = coordinate * inverse_frequency[index];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            cosines[position * LM_ROPE_HALF + index] =
                position_ids ? round_bf16(cosine) : cosine;
            sines[position * LM_ROPE_HALF + index] =
                position_ids ? round_bf16(sine) : sine;
        }
    }
    *rope_cos = h3_gpu_tensor_from_f32(gpu, cosines, tokens * LM_ROPE_HALF);
    *rope_sin = h3_gpu_tensor_from_f32(gpu, sines, tokens * LM_ROPE_HALF);
    free(cosines);
    free(sines);
    if (!*rope_cos || !*rope_sin) {
        fail(error, error_size, "cannot upload Qwen RoPE tables: %s",
             h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

int qwen_lm_decode_tail(const struct qwen_engine *engine,
                        const uint16_t *hidden_layer49, size_t tokens,
                        const uint32_t *position_ids, qwen_logits *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!engine || !hidden_layer49 || !tokens || !output) {
        fail(error, error_size, "qwen_lm_decode_tail requires engine, hidden "
             "and output");
        return 0;
    }
    if (tokens > UINT32_MAX / LM_INTERMEDIATE) {
        fail(error, error_size, "token count %zu is too large for Phase 1",
             tokens);
        return 0;
    }
    uint32_t rows = (uint32_t)tokens;
    size_t hidden_count = tokens * LM_HIDDEN;

    h3_weight_store *store =
        h3_weight_store_open(engine->weight_directory, error, error_size);
    if (!store) return 0;
    h3_gpu *gpu = h3_gpu_create(engine->shader_source_path, error, error_size);
    if (!gpu) {
        h3_weight_store_free(store);
        return 0;
    }
    h3_gpu_profile_set_label(gpu, "Qwen LM tail");

    int ok = 0;
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    h3_gpu_tensor *hidden = NULL, *norm = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *attention_heads = NULL;
    h3_gpu_tensor *attention_output = NULL, *gate = NULL, *up = NULL;
    h3_gpu_tensor *mlp_output = NULL;
    h3_gpu_tensor *final_norm_weight = NULL, *lm_head_weight = NULL;
    h3_gpu_tensor *logits = NULL;
    uint16_t *logits_host = NULL;
    lm_layer_weights weights;
    memset(&weights, 0, sizeof(weights));

    if (!lm_build_rope(gpu, tokens, position_ids, &rope_cos, &rope_sin, error,
                       error_size))
        goto done;

    hidden = h3_gpu_tensor_from_bf16(gpu, hidden_layer49, hidden_count);
    norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    query = h3_gpu_tensor_new_bf16(gpu, tokens * LM_QUERY_DIM);
    key = h3_gpu_tensor_new_bf16(gpu, tokens * LM_KV_DIM);
    value = h3_gpu_tensor_new_bf16(gpu, tokens * LM_KV_DIM);
    attention_heads = h3_gpu_tensor_new_bf16(gpu, tokens * LM_QUERY_DIM);
    attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    gate = h3_gpu_tensor_new_bf16(gpu, tokens * LM_INTERMEDIATE);
    up = h3_gpu_tensor_new_bf16(gpu, tokens * LM_INTERMEDIATE);
    mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    if (!hidden || !norm || !query || !key || !value || !attention_heads ||
        !attention_output || !gate || !up || !mlp_output) {
        fail(error, error_size, "cannot allocate Qwen LM activations: %s",
             h3_gpu_error(gpu));
        goto done;
    }

    for (int layer = LM_TAIL_START; layer < LM_TOTAL_LAYERS; layer++) {
        if (!lm_layer_weights_load(store, gpu, layer, &weights, error,
                                   error_size))
            goto done;
        int layer_ok =
            op(gpu, h3_gpu_begin(gpu), error, error_size, "layer stream begin",
               layer) &&
            lm_encode_layer(gpu, &weights, rows, hidden, norm, query, key,
                            value, attention_heads, attention_output, gate, up,
                            mlp_output, rope_cos, rope_sin, layer, error,
                            error_size) &&
            op(gpu, h3_gpu_submit(gpu), error, error_size, "layer stream submit",
               layer);
        lm_layer_weights_free(&weights);
        if (!layer_ok) goto done;
    }

    {
        uint64_t norm_shape[] = {LM_HIDDEN};
        uint64_t head_shape[] = {LM_VOCAB, LM_HIDDEN};
        final_norm_weight =
            h3_weight_load_bf16(store, gpu, "model.language_model.norm.weight",
                                1, norm_shape, error, error_size);
        lm_head_weight = h3_weight_load_bf16(store, gpu, "lm_head.weight", 2,
                                             head_shape, error, error_size);
    }
    if (!final_norm_weight || !lm_head_weight) goto done;

    logits = h3_gpu_tensor_new_bf16(gpu, tokens * LM_VOCAB);
    if (!logits) {
        fail(error, error_size, "cannot allocate Qwen logits (%zu x %d): %s",
             tokens, LM_VOCAB, h3_gpu_error(gpu));
        goto done;
    }
    if (!op(gpu, h3_gpu_begin(gpu), error, error_size, "head stream begin",
            -1) ||
        !op(gpu, h3_gpu_rms_norm_bf16(gpu, norm, hidden, final_norm_weight, rows,
                                      LM_HIDDEN, LM_RMS_EPSILON), error,
            error_size, "final RMSNorm", -1) ||
        !op(gpu, h3_gpu_linear_bf16(gpu, logits, norm, lm_head_weight, NULL,
                                    rows, LM_HIDDEN, LM_VOCAB), error,
            error_size, "lm_head", -1) ||
        !op(gpu, h3_gpu_submit(gpu), error, error_size, "head stream submit",
            -1))
        goto done;

    logits_host = malloc(tokens * LM_VOCAB * sizeof(*logits_host));
    if (!logits_host) {
        fail(error, error_size, "out of memory reading Qwen logits");
        goto done;
    }
    if (!h3_gpu_tensor_read_bf16(logits, logits_host, tokens * LM_VOCAB)) {
        fail(error, error_size, "cannot read Qwen logits: %s",
             h3_gpu_error(gpu));
        goto done;
    }

    output->values = malloc(LM_VOCAB * sizeof(*output->values));
    if (!output->values) {
        fail(error, error_size, "out of memory materializing Qwen logits");
        goto done;
    }
    output->vocab = LM_VOCAB;
    {
        const uint16_t *last = logits_host + (tokens - 1) * LM_VOCAB;
        float best = -INFINITY;
        uint32_t best_index = 0;
        for (size_t index = 0; index < LM_VOCAB; index++) {
            float value_f = bf16_to_f32(last[index]);
            output->values[index] = value_f;
            if (value_f > best) {
                best = value_f;
                best_index = (uint32_t)index;
            }
        }
        output->argmax_token = best_index;
    }
    ok = 1;

done:
    free(logits_host);
    lm_layer_weights_free(&weights);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(attention_heads);
    h3_gpu_tensor_free(attention_output);
    h3_gpu_tensor_free(gate);
    h3_gpu_tensor_free(up);
    h3_gpu_tensor_free(mlp_output);
    h3_gpu_tensor_free(final_norm_weight);
    h3_gpu_tensor_free(lm_head_weight);
    h3_gpu_tensor_free(logits);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    if (!ok) qwen_logits_free(output);
    return ok;
}
