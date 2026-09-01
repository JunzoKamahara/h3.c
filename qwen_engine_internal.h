#ifndef QWEN_ENGINE_INTERNAL_H
#define QWEN_ENGINE_INTERNAL_H

/* Shared internals for the qwen_engine translation unit and the Phase 1
 * language-model tail (qwen_lm.c). Not a public interface. */

#include "qwen_engine.h"

struct qwen_engine {
    char *weight_directory;
    char *shader_source_path;
};

struct qwen_session {
    qwen_engine *engine;
};

/* Run Qwen3-VL decoder layers 50..63 on a layer-49 intermediate state, then the
 * final RMSNorm and lm_head, and fill `output` with last-position logits and
 * their CPU argmax. `hidden_layer49` is BF16 [tokens, QWEN_HIDDEN_SIZE].
 * `position_ids` is axis-major [3, tokens] mRoPE ids, or NULL for sequential
 * text positions -- it must match whatever produced `hidden_layer49`. */
int qwen_lm_decode_tail(const struct qwen_engine *engine,
                        const uint16_t *hidden_layer49, size_t tokens,
                        const uint32_t *position_ids,
                        qwen_logits *output, char *error, size_t error_size);

#endif
