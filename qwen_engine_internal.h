#ifndef QWEN_ENGINE_INTERNAL_H
#define QWEN_ENGINE_INTERNAL_H

/* Shared internals for the qwen_engine translation unit, the Phase 1
 * language-model tail (qwen_lm.c) and the Phase 2 KV-cache decoder
 * (qwen_kv.c). Not a public interface. */

#include "qwen_engine.h"

struct qwen_engine {
    char *weight_directory;
    char *shader_source_path;
};

typedef struct qwen_kv_context qwen_kv_context;

struct qwen_session {
    qwen_engine *engine;
    qwen_kv_context *kv; /* NULL until the first qwen_session_eval() */
    int resident_mode;   /* 0 = default (resident, fall back to streaming),
                          * 1 = force resident, -1 = force streaming */
};

/* Phase 1: run Qwen3-VL decoder layers 50..63 on a layer-49 intermediate
 * state, then the final RMSNorm and lm_head, and fill `output` with
 * last-position logits and their CPU argmax. `hidden_layer49` is BF16
 * [tokens, QWEN_HIDDEN_SIZE]. `position_ids` is axis-major [3, tokens] mRoPE
 * ids, or NULL for sequential text positions. */
int qwen_lm_decode_tail(const struct qwen_engine *engine,
                        const uint16_t *hidden_layer49, size_t tokens,
                        const uint32_t *position_ids,
                        qwen_logits *output, char *error, size_t error_size);

/* Phase 2: append `token_count` text tokens to the session's KV cache, running
 * the full 64-layer forward on just the new rows, and update the session's
 * latest logits. Allocates the KV context on first use. */
int qwen_kv_eval(struct qwen_session *session, const uint32_t *token_ids,
                 size_t token_count, char *error, size_t error_size);

/* P7-005: multimodal prefill -- must be the first eval on the session. `input`
 * carries token_ids, vision_spans (embeddings + deepstack), axis-major
 * position_ids and tags; the vision rows are spliced into the residual stream
 * and the deepstack additions applied after layers 0/1/2. Subsequent
 * qwen_kv_eval() calls decode text with mRoPE positions continuing past the
 * vision grid. */
int qwen_kv_eval_multimodal(struct qwen_session *session,
                            const qwen_input *input, char *error,
                            size_t error_size);

/* QINT-015d: append `block_count` (2..QWEN_VERIFY_MAX) tokens in one VERIFY
 * forward and fill per-row top1/top2/margin. All rows are left appended; the
 * caller rewinds. */
int qwen_kv_eval_verify_block(struct qwen_session *session,
                              const uint32_t *block, size_t block_count,
                              qwen_verify_result *result, char *error,
                              size_t error_size);

/* Truncate the KV cache, history and position back to `keep` tokens. */
int qwen_kv_rewind(struct qwen_session *session, size_t keep, char *error,
                   size_t error_size);

/* QINT-015h: configure / read the EAGLE-3 auxiliary-hidden snapshot. See the
 * qwen_session_set_aux_layers / qwen_session_aux_hidden contract in
 * qwen_engine.h. set_aux_layers allocates the KV context if needed. */
int qwen_kv_set_aux_layers(struct qwen_session *session, const int *layer_ids,
                           size_t count, char *error, size_t error_size);
const uint16_t *qwen_kv_aux_hidden(const struct qwen_session *session,
                                   size_t *rows, size_t *n_aux, size_t *hidden,
                                   const int **layer_ids);

const qwen_logits *qwen_kv_latest_logits(const struct qwen_session *session);
size_t qwen_kv_length(const struct qwen_session *session);

/* The session's token history (prompt + everything evalled), [*length_out].
 * Valid until the next eval / rewind. NULL with *length_out = 0 before the
 * first eval. */
const uint32_t *qwen_kv_history(const struct qwen_session *session,
                               size_t *length_out);

void qwen_kv_context_free(qwen_kv_context *kv);

#endif
