#include "qwen_engine_internal.h"
#include "qwen_layers.h"

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
 * The per-layer recipe lives in qwen_layers.c and is shared with the Phase 2
 * KV-cache decoder. This path is a plain full-prompt forward: no KV cache, tail
 * weights streamed one layer at a time. `make phase1-parity` checks that
 * forward_full() decomposes exactly into
 * continue_from_intermediate(get_h3_conditioning()) and is deterministic. */

static void fail(char *error, size_t error_size, const char *format, ...) {
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

static int op(h3_gpu *gpu, int ok, char *error, size_t error_size,
              const char *what) {
    if (ok) return 1;
    fail(error, error_size, "Qwen LM %s failed: %s", what, h3_gpu_error(gpu));
    return 0;
}

int qwen_lm_decode_tail(const struct qwen_engine *engine,
                        const uint16_t *hidden_layer49, size_t tokens,
                        const uint32_t *position_ids, qwen_logits *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!engine || !hidden_layer49 || !tokens || !output) {
        fail(error, error_size,
             "qwen_lm_decode_tail requires engine, hidden and output");
        return 0;
    }
    if (tokens > UINT32_MAX / QWEN_LM_INTERMEDIATE) {
        fail(error, error_size, "token count %zu is too large for Phase 1",
             tokens);
        return 0;
    }
    uint32_t rows = (uint32_t)tokens;
    size_t hidden_count = tokens * QWEN_LM_HIDDEN;

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
    float *cosines = NULL, *sines = NULL;
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    h3_gpu_tensor *hidden = NULL, *norm = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *attention_heads = NULL;
    h3_gpu_tensor *attention_output = NULL, *gate = NULL, *up = NULL;
    h3_gpu_tensor *mlp_output = NULL;
    h3_gpu_tensor *final_norm_weight = NULL, *lm_head_weight = NULL;
    h3_gpu_tensor *logits = NULL;
    uint16_t *logits_host = NULL;
    qwen_layer_weights weights;
    memset(&weights, 0, sizeof(weights));

    if (!qwen_build_rope_tables(tokens, 0, position_ids, &cosines, &sines,
                                error, error_size))
        goto done;
    rope_cos = h3_gpu_tensor_from_f32(gpu, cosines, tokens * QWEN_LM_ROPE_HALF);
    rope_sin = h3_gpu_tensor_from_f32(gpu, sines, tokens * QWEN_LM_ROPE_HALF);
    hidden = h3_gpu_tensor_from_bf16(gpu, hidden_layer49, hidden_count);
    norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    query = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_QUERY_DIM);
    key = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_KV_DIM);
    value = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_KV_DIM);
    attention_heads = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_QUERY_DIM);
    attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    gate = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_INTERMEDIATE);
    up = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_INTERMEDIATE);
    mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    if (!rope_cos || !rope_sin || !hidden || !norm || !query || !key ||
        !value || !attention_heads || !attention_output || !gate || !up ||
        !mlp_output) {
        fail(error, error_size, "cannot allocate Qwen LM activations: %s",
             h3_gpu_error(gpu));
        goto done;
    }

    for (int layer = QWEN_LM_RELEASED_LAYERS; layer < QWEN_LM_TOTAL_LAYERS;
         layer++) {
        if (!qwen_layer_weights_load(store, gpu, layer, &weights, error,
                                     error_size))
            goto done;
        int layer_ok =
            op(gpu, h3_gpu_begin(gpu), error, error_size, "layer begin") &&
            qwen_layer_prep(gpu, &weights, rows, QWEN_EVAL_PREFILL, hidden, norm,
                            query, key, value, rope_cos, rope_sin, layer, error,
                            error_size) &&
            op(gpu, h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key,
                                           value, rows, QWEN_LM_QUERY_HEADS,
                                           QWEN_LM_KV_HEADS, QWEN_LM_HEAD_DIM,
                                           qwen_lm_attention_scale()),
               error, error_size, "causal GQA") &&
            qwen_layer_finish(gpu, &weights, rows, QWEN_EVAL_PREFILL, hidden,
                              attention_heads, norm, attention_output, gate, up,
                              mlp_output, layer, error, error_size) &&
            op(gpu, h3_gpu_submit(gpu), error, error_size, "layer submit");
        qwen_layer_weights_free(&weights);
        if (!layer_ok) goto done;
    }

    {
        uint64_t norm_shape[] = {QWEN_LM_HIDDEN};
        uint64_t head_shape[] = {QWEN_LM_VOCAB, QWEN_LM_HIDDEN};
        final_norm_weight =
            h3_weight_load_bf16(store, gpu, "model.language_model.norm.weight",
                                1, norm_shape, error, error_size);
        lm_head_weight = h3_weight_load_bf16(store, gpu, "lm_head.weight", 2,
                                             head_shape, error, error_size);
    }
    if (!final_norm_weight || !lm_head_weight) goto done;

    logits = h3_gpu_tensor_new_bf16(gpu, tokens * QWEN_LM_VOCAB);
    if (!logits) {
        fail(error, error_size, "cannot allocate Qwen logits (%zu x %d): %s",
             tokens, QWEN_LM_VOCAB, h3_gpu_error(gpu));
        goto done;
    }
    if (!op(gpu, h3_gpu_begin(gpu), error, error_size, "head begin") ||
        !op(gpu, h3_gpu_rms_norm_bf16(gpu, norm, hidden, final_norm_weight,
                                      rows, QWEN_LM_HIDDEN, QWEN_LM_RMS_EPSILON),
            error, error_size, "final RMSNorm") ||
        !op(gpu, h3_gpu_linear_bf16(gpu, logits, norm, lm_head_weight, NULL,
                                    rows, QWEN_LM_HIDDEN, QWEN_LM_VOCAB),
            error, error_size, "lm_head") ||
        !op(gpu, h3_gpu_submit(gpu), error, error_size, "head submit"))
        goto done;

    logits_host = malloc(tokens * QWEN_LM_VOCAB * sizeof(*logits_host));
    if (!logits_host) {
        fail(error, error_size, "out of memory reading Qwen logits");
        goto done;
    }
    if (!h3_gpu_tensor_read_bf16(logits, logits_host, tokens * QWEN_LM_VOCAB)) {
        fail(error, error_size, "cannot read Qwen logits: %s",
             h3_gpu_error(gpu));
        goto done;
    }

    output->values = malloc(QWEN_LM_VOCAB * sizeof(*output->values));
    if (!output->values) {
        fail(error, error_size, "out of memory materializing Qwen logits");
        goto done;
    }
    output->vocab = QWEN_LM_VOCAB;
    {
        const uint16_t *last = logits_host + (tokens - 1) * QWEN_LM_VOCAB;
        float best = -INFINITY;
        uint32_t best_index = 0;
        for (size_t index = 0; index < QWEN_LM_VOCAB; index++) {
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
    free(cosines);
    free(sines);
    free(logits_host);
    qwen_layer_weights_free(&weights);
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
