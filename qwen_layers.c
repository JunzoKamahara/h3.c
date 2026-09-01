#include "qwen_layers.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float QWEN_LM_RMS_EPSILON = 1e-6f;
const float QWEN_LM_ROPE_THETA = 5000000.0f;

float qwen_lm_attention_scale(void) {
    return 1.0f / sqrtf((float)QWEN_LM_HEAD_DIM);
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void qwen_layer_weights_free(qwen_layer_weights *weights) {
    if (!weights) return;
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

int qwen_layer_weights_load(const h3_weight_store *store, h3_gpu *gpu, int layer,
                            qwen_layer_weights *out, char *error,
                            size_t error_size) {
    memset(out, 0, sizeof(*out));
    char prefix[96];
    if (snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d.",
                 layer) < 0) {
        fail(error, error_size, "cannot format Qwen layer name");
        return 0;
    }
#define LOAD(field, suffix, ndim, ...) do {                                    \
    char name[192];                                                            \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    uint64_t shape[] = {__VA_ARGS__};                                          \
    out->field = h3_weight_load_bf16(store, gpu, name, ndim, shape, error,     \
                                     error_size);                              \
    if (!out->field) { qwen_layer_weights_free(out); return 0; }               \
} while (0)
    LOAD(input_norm, "input_layernorm.weight", 1, QWEN_LM_HIDDEN);
    LOAD(query, "self_attn.q_proj.weight", 2, QWEN_LM_QUERY_DIM, QWEN_LM_HIDDEN);
    LOAD(key, "self_attn.k_proj.weight", 2, QWEN_LM_KV_DIM, QWEN_LM_HIDDEN);
    LOAD(value, "self_attn.v_proj.weight", 2, QWEN_LM_KV_DIM, QWEN_LM_HIDDEN);
    LOAD(query_norm, "self_attn.q_norm.weight", 1, QWEN_LM_HEAD_DIM);
    LOAD(key_norm, "self_attn.k_norm.weight", 1, QWEN_LM_HEAD_DIM);
    LOAD(attention_output, "self_attn.o_proj.weight", 2, QWEN_LM_HIDDEN,
         QWEN_LM_QUERY_DIM);
    LOAD(post_norm, "post_attention_layernorm.weight", 1, QWEN_LM_HIDDEN);
    LOAD(gate, "mlp.gate_proj.weight", 2, QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN);
    LOAD(up, "mlp.up_proj.weight", 2, QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN);
    LOAD(down, "mlp.down_proj.weight", 2, QWEN_LM_HIDDEN, QWEN_LM_INTERMEDIATE);
#undef LOAD
    return 1;
}

int qwen_build_rope_tables(size_t tokens, size_t position_offset,
                           const uint32_t *positions, float **cosines_out,
                           float **sines_out, char *error, size_t error_size) {
    *cosines_out = NULL;
    *sines_out = NULL;
    float *cosines = malloc(tokens * QWEN_LM_ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(tokens * QWEN_LM_ROPE_HALF * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines);
        free(sines);
        fail(error, error_size, "out of memory allocating Qwen RoPE tables");
        return 0;
    }
    float inverse_frequency[QWEN_LM_ROPE_HALF];
    for (size_t index = 0; index < QWEN_LM_ROPE_HALF; index++)
        inverse_frequency[index] =
            1.0f / powf(QWEN_LM_ROPE_THETA,
                        (float)(index * 2) / (float)QWEN_LM_HEAD_DIM);
    for (size_t position = 0; position < tokens; position++) {
        for (size_t index = 0; index < QWEN_LM_ROPE_HALF; index++) {
            size_t axis = 0;
            if (positions && index < 60 && index % 3 == 1) axis = 1;
            else if (positions && index < 60 && index % 3 == 2) axis = 2;
            float coordinate = positions
                ? (float)positions[axis * tokens + position]
                : (float)(position_offset + position);
            float angle = coordinate * inverse_frequency[index];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            cosines[position * QWEN_LM_ROPE_HALF + index] =
                positions ? round_bf16(cosine) : cosine;
            sines[position * QWEN_LM_ROPE_HALF + index] =
                positions ? round_bf16(sine) : sine;
        }
    }
    *cosines_out = cosines;
    *sines_out = sines;
    return 1;
}

static int op(h3_gpu *gpu, int ok, char *error, size_t error_size,
              const char *what, int layer) {
    if (ok) return 1;
    if (layer >= 0)
        fail(error, error_size, "Qwen layer %d %s failed: %s", layer, what,
             h3_gpu_error(gpu));
    else
        fail(error, error_size, "Qwen %s failed: %s", what, h3_gpu_error(gpu));
    return 0;
}

int qwen_layer_prep(h3_gpu *gpu, const qwen_layer_weights *w, uint32_t rows,
                    h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                    h3_gpu_tensor *query, h3_gpu_tensor *key,
                    h3_gpu_tensor *value, h3_gpu_tensor *rope_cos,
                    h3_gpu_tensor *rope_sin, int layer, char *error,
                    size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!op(gpu, (call), error, error_size, label, layer)) return 0;           \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->input_norm, rows,
                            QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
       "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, query, norm, w->query, NULL, rows,
                          QWEN_LM_HIDDEN, QWEN_LM_QUERY_DIM),
       "query projection");
    OP(h3_gpu_linear_bf16(gpu, key, norm, w->key, NULL, rows, QWEN_LM_HIDDEN,
                          QWEN_LM_KV_DIM), "key projection");
    OP(h3_gpu_linear_bf16(gpu, value, norm, w->value, NULL, rows,
                          QWEN_LM_HIDDEN, QWEN_LM_KV_DIM), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, query, w->query_norm, rows,
                                 QWEN_LM_QUERY_HEADS, QWEN_LM_HEAD_DIM,
                                 QWEN_LM_RMS_EPSILON), "query RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, key, w->key_norm, rows, QWEN_LM_KV_HEADS,
                                 QWEN_LM_HEAD_DIM, QWEN_LM_RMS_EPSILON),
       "key RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, rows,
                             QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                             QWEN_LM_HEAD_DIM), "RoPE");
#undef OP
    return 1;
}

int qwen_layer_finish(h3_gpu *gpu, const qwen_layer_weights *w, uint32_t rows,
                      h3_gpu_tensor *hidden, h3_gpu_tensor *attention_heads,
                      h3_gpu_tensor *norm, h3_gpu_tensor *attention_output,
                      h3_gpu_tensor *gate, h3_gpu_tensor *up,
                      h3_gpu_tensor *mlp_output, int layer, char *error,
                      size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!op(gpu, (call), error, error_size, label, layer)) return 0;           \
} while (0)
    OP(h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                          w->attention_output, NULL, rows, QWEN_LM_QUERY_DIM,
                          QWEN_LM_HIDDEN), "attention output projection");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                       rows * QWEN_LM_HIDDEN), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->post_norm, rows,
                            QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
       "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, gate, norm, w->gate, NULL, rows, QWEN_LM_HIDDEN,
                          QWEN_LM_INTERMEDIATE), "MLP gate");
    OP(h3_gpu_linear_bf16(gpu, up, norm, w->up, NULL, rows, QWEN_LM_HIDDEN,
                          QWEN_LM_INTERMEDIATE), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up, rows * QWEN_LM_INTERMEDIATE),
       "fused SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, mlp_output, gate, w->down, NULL, rows,
                          QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output, rows * QWEN_LM_HIDDEN),
       "MLP residual");
#undef OP
    return 1;
}
