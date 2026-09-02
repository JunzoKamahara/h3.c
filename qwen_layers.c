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
    qwen_q4_weight_free(&weights->q4_query);
    qwen_q4_weight_free(&weights->q4_key);
    qwen_q4_weight_free(&weights->q4_value);
    qwen_q4_weight_free(&weights->q4_attention_output);
    qwen_q4_weight_free(&weights->q4_gate);
    qwen_q4_weight_free(&weights->q4_up);
    qwen_q4_weight_free(&weights->q4_down);
    memset(weights, 0, sizeof(*weights));
}

int qwen_layer_weights_quantize(qwen_layer_weights *weights, h3_gpu *gpu,
                                int layer, const char *awq_calib_path,
                                uint32_t proj_mask, char *error,
                                size_t error_size) {
    if (!proj_mask) { weights->has_q4 = 0; return 1; }
    /* Per-projection act-scale slot: q/k/v share the input-RMSNorm activation,
     * gate/up share the post-attn-RMSNorm activation. */
    struct {
        const h3_gpu_tensor *src;
        qwen_q4_weight *dst;
        uint32_t rows, cols;
        int slot;
    } jobs[] = {
        {weights->query, &weights->q4_query, QWEN_LM_QUERY_DIM, QWEN_LM_HIDDEN,
         QWEN_AWQ_QKV_IN},
        {weights->key, &weights->q4_key, QWEN_LM_KV_DIM, QWEN_LM_HIDDEN,
         QWEN_AWQ_QKV_IN},
        {weights->value, &weights->q4_value, QWEN_LM_KV_DIM, QWEN_LM_HIDDEN,
         QWEN_AWQ_QKV_IN},
        {weights->attention_output, &weights->q4_attention_output,
         QWEN_LM_HIDDEN, QWEN_LM_QUERY_DIM, QWEN_AWQ_O_IN},
        {weights->gate, &weights->q4_gate, QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN,
         QWEN_AWQ_MLP_IN},
        {weights->up, &weights->q4_up, QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN,
         QWEN_AWQ_MLP_IN},
        {weights->down, &weights->q4_down, QWEN_LM_HIDDEN, QWEN_LM_INTERMEDIATE,
         QWEN_AWQ_DOWN_IN},
    };

    float *act[QWEN_AWQ_SLOTS] = {0};
    uint32_t act_cols[QWEN_AWQ_SLOTS] = {0};
    if (awq_calib_path &&
        !qwen_awq_calib_load_layer(awq_calib_path, layer, act, act_cols, error,
                                   error_size))
        return 0;

    int ok = 1, any = 0;
    for (size_t i = 0; i < sizeof(jobs) / sizeof(jobs[0]) && ok; i++) {
        if (!(proj_mask & (1u << i))) continue;   /* ablation: keep BF16 */
        const float *a = NULL;
        if (awq_calib_path) {
            a = act[jobs[i].slot];
            if (!a || act_cols[jobs[i].slot] != jobs[i].cols) {
                snprintf(error, error_size,
                         "AWQ calib layer %d slot %d: missing/size mismatch",
                         layer, jobs[i].slot);
                ok = 0;
                break;
            }
        }
        ok = qwen_q4_quantize_awq(gpu, jobs[i].src, jobs[i].rows, jobs[i].cols,
                                  a, jobs[i].dst, error, error_size);
        any = any || ok;
    }
    for (int s = 0; s < QWEN_AWQ_SLOTS; s++) free(act[s]);
    if (ok) weights->has_q4 = any;
    return ok;
}

/* Precision follows the eval kind (QINT-015d): DECODE -> INT4 GEMV (rows==1),
 * VERIFY -> INT4 decode-batch (rows 2..QWEN_SPEC_MAX), PREFILL -> BF16. Falls
 * back to BF16 when no INT4 copy is present (streaming / Q4 disabled) or the
 * batch kernel is not yet available. */
static int qwen_linear(h3_gpu *gpu, qwen_eval_kind kind, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight_bf16,
                       const qwen_q4_weight *q4, int has_q4, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    if (has_q4 && q4->packed && kind == QWEN_EVAL_DECODE && rows == 1 &&
        h3_gpu_linear_q4_gemv(gpu, output, input, q4->packed, q4->scales,
                              q4->awq_inv_scale, NULL, input_dim, output_dim,
                              QWEN_Q4_GROUP))
        return 1;
    if (has_q4 && q4->packed && kind == QWEN_EVAL_VERIFY && rows >= 2 &&
        h3_gpu_linear_q4_decode_batch(gpu, output, input, q4->packed,
                                      q4->scales, q4->awq_inv_scale, NULL, rows,
                                      input_dim, output_dim, QWEN_Q4_GROUP))
        return 1;
    return h3_gpu_linear_bf16(gpu, output, input, weight_bf16, NULL, rows,
                              input_dim, output_dim);
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
                    qwen_eval_kind kind, h3_gpu_tensor *hidden,
                    h3_gpu_tensor *norm, h3_gpu_tensor *query, h3_gpu_tensor *key,
                    h3_gpu_tensor *value, h3_gpu_tensor *rope_cos,
                    h3_gpu_tensor *rope_sin, int layer, char *error,
                    size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!op(gpu, (call), error, error_size, label, layer)) return 0;           \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->input_norm, rows,
                            QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
       "input RMSNorm");
    OP(qwen_linear(gpu, kind, query, norm, w->query, &w->q4_query, w->has_q4,
                   rows, QWEN_LM_HIDDEN, QWEN_LM_QUERY_DIM), "query projection");
    OP(qwen_linear(gpu, kind, key, norm, w->key, &w->q4_key, w->has_q4, rows,
                   QWEN_LM_HIDDEN, QWEN_LM_KV_DIM), "key projection");
    OP(qwen_linear(gpu, kind, value, norm, w->value, &w->q4_value, w->has_q4,
                   rows, QWEN_LM_HIDDEN, QWEN_LM_KV_DIM), "value projection");
    if (kind != QWEN_EVAL_PREFILL) {
        /* DECODE / VERIFY: one dispatch for Q/K head RMSNorm + RoPE (handles
         * rows 1..N). Not bit-exact vs the trio below (normed value stays F32
         * into the rotation); PREFILL keeps the separate kernels so its parity
         * memcmp holds. */
        OP(h3_gpu_qk_headnorm_rope_bf16(gpu, query, key, w->query_norm,
                                        w->key_norm, rope_cos, rope_sin, rows,
                                        QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                                        QWEN_LM_HEAD_DIM, QWEN_LM_RMS_EPSILON),
           "fused Q/K RMSNorm + RoPE");
    } else {
        OP(h3_gpu_head_rms_norm_bf16(gpu, query, w->query_norm, rows,
                                     QWEN_LM_QUERY_HEADS, QWEN_LM_HEAD_DIM,
                                     QWEN_LM_RMS_EPSILON), "query RMSNorm");
        OP(h3_gpu_head_rms_norm_bf16(gpu, key, w->key_norm, rows,
                                     QWEN_LM_KV_HEADS, QWEN_LM_HEAD_DIM,
                                     QWEN_LM_RMS_EPSILON), "key RMSNorm");
        OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, rows,
                                 QWEN_LM_QUERY_HEADS, QWEN_LM_KV_HEADS,
                                 QWEN_LM_HEAD_DIM), "RoPE");
    }
#undef OP
    return 1;
}

int qwen_layer_finish(h3_gpu *gpu, const qwen_layer_weights *w, uint32_t rows,
                      qwen_eval_kind kind, h3_gpu_tensor *hidden,
                      h3_gpu_tensor *attention_heads, h3_gpu_tensor *norm,
                      h3_gpu_tensor *attention_output, h3_gpu_tensor *gate,
                      h3_gpu_tensor *up, h3_gpu_tensor *mlp_output, int layer,
                      char *error, size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!op(gpu, (call), error, error_size, label, layer)) return 0;           \
} while (0)
    OP(qwen_linear(gpu, kind, attention_output, attention_heads,
                   w->attention_output, &w->q4_attention_output, w->has_q4, rows,
                   QWEN_LM_QUERY_DIM, QWEN_LM_HIDDEN),
       "attention output projection");
    /* attention residual + post-attention RMSNorm in one dispatch
     * (bit-exact with the add + rms_norm pair). */
    OP(h3_gpu_add_rms_norm_bf16(gpu, hidden, norm, hidden, attention_output,
                                w->post_norm, rows, QWEN_LM_HIDDEN,
                                QWEN_LM_RMS_EPSILON),
       "attention residual + post-attention RMSNorm");
    OP(qwen_linear(gpu, kind, gate, norm, w->gate, &w->q4_gate, w->has_q4, rows,
                   QWEN_LM_HIDDEN, QWEN_LM_INTERMEDIATE), "MLP gate");
    OP(qwen_linear(gpu, kind, up, norm, w->up, &w->q4_up, w->has_q4, rows,
                   QWEN_LM_HIDDEN, QWEN_LM_INTERMEDIATE), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up, rows * QWEN_LM_INTERMEDIATE),
       "fused SwiGLU");
    OP(qwen_linear(gpu, kind, mlp_output, gate, w->down, &w->q4_down, w->has_q4,
                   rows, QWEN_LM_INTERMEDIATE, QWEN_LM_HIDDEN), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output, rows * QWEN_LM_HIDDEN),
       "MLP residual");
#undef OP
    return 1;
}
