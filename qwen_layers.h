#ifndef QWEN_LAYERS_H
#define QWEN_LAYERS_H

/* Shared Qwen3-VL decoder-layer primitives for the Phase 1 LM tail (qwen_lm.c)
 * and the Phase 2 KV-cache decoder (qwen_kv.c). Not a public interface.
 *
 * The per-layer recipe, epsilon, theta and mRoPE table construction mirror
 * h3_text_encoder.c (encode_layer / text_encode_bf16_impl) exactly and must
 * stay in sync with it; `make phase0-parity` guards layers 0..49. */

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

enum {
    QWEN_LM_TOTAL_LAYERS = 64,
    QWEN_LM_RELEASED_LAYERS = 50,
    QWEN_LM_VOCAB = 151936,
    QWEN_LM_HIDDEN = 5120,
    QWEN_LM_INTERMEDIATE = 25600,
    QWEN_LM_QUERY_HEADS = 64,
    QWEN_LM_KV_HEADS = 8,
    QWEN_LM_HEAD_DIM = 128,
    QWEN_LM_QUERY_DIM = QWEN_LM_QUERY_HEADS * QWEN_LM_HEAD_DIM,
    QWEN_LM_KV_DIM = QWEN_LM_KV_HEADS * QWEN_LM_HEAD_DIM,
    QWEN_LM_ROPE_HALF = QWEN_LM_HEAD_DIM / 2
};

extern const float QWEN_LM_RMS_EPSILON;
extern const float QWEN_LM_ROPE_THETA;

/* Attention scale, computed the same way as h3_text_encoder.c: 1/sqrtf(128). */
float qwen_lm_attention_scale(void);

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
} qwen_layer_weights;

int qwen_layer_weights_load(const h3_weight_store *store, h3_gpu *gpu, int layer,
                            qwen_layer_weights *out, char *error,
                            size_t error_size);
void qwen_layer_weights_free(qwen_layer_weights *weights);

/* F32 [tokens, QWEN_LM_ROPE_HALF] cos/sin tables. With `positions` non-NULL
 * (axis-major [3, tokens] mRoPE ids) the values are pinned to BF16 exactly as
 * h3_text_encoder.c does, and `position_offset` must be 0. With `positions`
 * NULL the coordinates are sequential `position_offset + i` in full F32 (the
 * legacy text path; offset lets the KV decoder continue past a cached prefix).
 * Caller frees the two returned host buffers. */
int qwen_build_rope_tables(size_t tokens, size_t position_offset,
                           const uint32_t *positions, float **cosines,
                           float **sines, char *error, size_t error_size);

/* Encode (no submit) the pre-attention half of one decoder layer: input
 * RMSNorm, Q/K/V projections, per-head Q/K RMSNorm, and text RoPE over `rows`
 * rows using the supplied cos/sin tables. `query`/`key`/`value` receive the
 * post-RoPE projections; `hidden` is left as the residual base. */
int qwen_layer_prep(h3_gpu *gpu, const qwen_layer_weights *w, uint32_t rows,
                    h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                    h3_gpu_tensor *query, h3_gpu_tensor *key,
                    h3_gpu_tensor *value, h3_gpu_tensor *rope_cos,
                    h3_gpu_tensor *rope_sin, int layer, char *error,
                    size_t error_size);

/* Encode (no submit) the post-attention half: output projection, attention
 * residual, post-attention RMSNorm, SwiGLU MLP and MLP residual.
 * `attention_heads` holds the attention result for `rows` rows. */
int qwen_layer_finish(h3_gpu *gpu, const qwen_layer_weights *w, uint32_t rows,
                      h3_gpu_tensor *hidden, h3_gpu_tensor *attention_heads,
                      h3_gpu_tensor *norm, h3_gpu_tensor *attention_output,
                      h3_gpu_tensor *gate, h3_gpu_tensor *up,
                      h3_gpu_tensor *mlp_output, int layer, char *error,
                      size_t error_size);

#endif
