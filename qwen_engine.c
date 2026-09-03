#include "qwen_engine.h"
#include "qwen_engine_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 0: the engine/session pair is a thin wrapper that funnels every caller
 * through the one shared 50-layer code path in h3_text_encoder.c. Chat/VLM and
 * H3 conditioning therefore execute identical GPU work, which is what keeps the
 * layer-49 intermediate state bit-for-bit stable across Chat-side changes.
 *
 * Phase 1 adds the decoder-layers-50..63 + lm_head tail in qwen_lm.c, reached
 * through qwen_session_continue_from_intermediate() and
 * qwen_engine_forward_full(). The layer-49 boundary above is unchanged. */

static void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static char *dup_string(const char *value) {
    if (!value) return NULL;
    size_t length = strlen(value) + 1;
    char *copy = malloc(length);
    if (copy) memcpy(copy, value, length);
    return copy;
}

void qwen_intermediate_state_free(qwen_intermediate_state *state) {
    if (!state) return;
    free(state->values);
    free(state->tags);
    memset(state, 0, sizeof(*state));
}

void qwen_logits_free(qwen_logits *logits) {
    if (!logits) return;
    free(logits->values);
    memset(logits, 0, sizeof(*logits));
}

int qwen_engine_open(qwen_engine **out,
                     const char *weight_directory,
                     const char *shader_source_path,
                     char *error, size_t error_size) {
    if (out) *out = NULL;
    if (!out || !weight_directory || !shader_source_path) {
        set_error(error, error_size,
                  "qwen_engine_open requires out, weight_directory and "
                  "shader_source_path");
        return 0;
    }
    qwen_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) {
        set_error(error, error_size, "out of memory allocating qwen_engine");
        return 0;
    }
    engine->weight_directory = dup_string(weight_directory);
    engine->shader_source_path = dup_string(shader_source_path);
    if (!engine->weight_directory || !engine->shader_source_path) {
        qwen_engine_close(engine);
        set_error(error, error_size, "out of memory copying qwen_engine paths");
        return 0;
    }
    *out = engine;
    return 1;
}

void qwen_engine_close(qwen_engine *engine) {
    if (!engine) return;
    free(engine->weight_directory);
    free(engine->shader_source_path);
    free(engine);
}

int qwen_session_create(qwen_session **out, qwen_engine *engine,
                        char *error, size_t error_size) {
    if (out) *out = NULL;
    if (!out || !engine) {
        set_error(error, error_size,
                  "qwen_session_create requires out and engine");
        return 0;
    }
    qwen_session *session = calloc(1, sizeof(*session));
    if (!session) {
        set_error(error, error_size, "out of memory allocating qwen_session");
        return 0;
    }
    session->engine = engine;
    *out = session;
    return 1;
}

void qwen_session_free(qwen_session *session) {
    if (!session) return;
    qwen_kv_context_free(session->kv);
    free(session);
}

int qwen_session_eval(qwen_session *session,
                      const uint32_t *token_ids, size_t token_count,
                      char *error, size_t error_size) {
    if (!session || !session->engine || !token_ids || !token_count) {
        set_error(error, error_size,
                  "qwen_session_eval requires a session and tokens");
        return 0;
    }
    return qwen_kv_eval(session, token_ids, token_count, error, error_size);
}

int qwen_session_eval_multimodal(qwen_session *session,
                                 const qwen_input *input, char *error,
                                 size_t error_size) {
    if (!session || !session->engine || !input) {
        set_error(error, error_size,
                  "qwen_session_eval_multimodal requires a session and input");
        return 0;
    }
    return qwen_kv_eval_multimodal(session, input, error, error_size);
}

int qwen_session_sample(qwen_session *session, uint32_t *token_out,
                        char *error, size_t error_size) {
    if (!session || !token_out) {
        set_error(error, error_size,
                  "qwen_session_sample requires a session and token_out");
        return 0;
    }
    const qwen_logits *logits = qwen_kv_latest_logits(session);
    if (!logits) {
        set_error(error, error_size,
                  "qwen_session_sample called before qwen_session_eval");
        return 0;
    }
    *token_out = logits->argmax_token;
    return 1;
}

const qwen_logits *qwen_session_logits(const qwen_session *session) {
    return session ? qwen_kv_latest_logits(session) : NULL;
}

int qwen_session_verify_block(qwen_session *session, const uint32_t *block,
                              size_t block_count, qwen_verify_result *result,
                              char *error, size_t error_size) {
    if (!session || !session->engine) {
        set_error(error, error_size,
                  "qwen_session_verify_block requires a session");
        return 0;
    }
    return qwen_kv_eval_verify_block(session, block, block_count, result, error,
                                    error_size);
}

size_t qwen_session_length(const qwen_session *session) {
    return session ? qwen_kv_length(session) : 0;
}

int qwen_session_set_aux_layers(qwen_session *session, const int *layer_ids,
                                size_t count, char *error, size_t error_size) {
    if (!session || !session->engine) {
        set_error(error, error_size,
                  "qwen_session_set_aux_layers requires a session");
        return 0;
    }
    return qwen_kv_set_aux_layers(session, layer_ids, count, error, error_size);
}

const uint16_t *qwen_session_aux_hidden(const qwen_session *session,
                                        size_t *rows, size_t *n_aux,
                                        size_t *hidden,
                                        const int **layer_ids) {
    if (rows) *rows = 0;
    if (n_aux) *n_aux = 0;
    if (hidden) *hidden = 0;
    if (layer_ids) *layer_ids = NULL;
    if (!session) return NULL;
    return qwen_kv_aux_hidden(session, rows, n_aux, hidden, layer_ids);
}

int qwen_session_embedding_row_f32(const qwen_session *session,
                                   uint32_t token_id, float *dst,
                                   size_t dst_count) {
    return qwen_kv_embedding_row_f32(session, token_id, dst, dst_count);
}

int qwen_session_set_aux_prefill_all_rows(qwen_session *session, int on) {
    return qwen_kv_set_aux_prefill_all(session, on);
}

const uint32_t *qwen_session_history(const qwen_session *session,
                                     size_t *length_out) {
    return session ? qwen_kv_history(session, length_out)
                   : (length_out ? (*length_out = 0, NULL) : NULL);
}

int qwen_session_rewind(qwen_session *session, size_t keep,
                        char *error, size_t error_size) {
    if (!session) {
        set_error(error, error_size, "qwen_session_rewind requires a session");
        return 0;
    }
    return qwen_kv_rewind(session, keep, error, error_size);
}

int qwen_session_sync(qwen_session *session, char *error, size_t error_size) {
    if (!session || !session->engine) {
        set_error(error, error_size, "qwen_session_sync requires a session");
        return 0;
    }
    return 1;
}

int qwen_session_set_resident(qwen_session *session, int resident,
                              char *error, size_t error_size) {
    if (!session) {
        set_error(error, error_size,
                  "qwen_session_set_resident requires a session");
        return 0;
    }
    if (session->kv) {
        set_error(error, error_size,
                  "qwen_session_set_resident must be called before the first "
                  "eval");
        return 0;
    }
    session->resident_mode = resident ? 1 : -1;
    return 1;
}

/* Move the legacy encoder result into the canonical intermediate-state type.
 * gpu_stats is intentionally dropped; it is diagnostics, not contract. */
static void embedding_into_state(h3_text_embedding *embedding,
                                 qwen_intermediate_state *output) {
    memset(output, 0, sizeof(*output));
    output->tokens = embedding->tokens;
    output->hidden_size = embedding->width;
    output->values = embedding->values;
    output->tags = embedding->tags;
    /* h3_text_encode_*_layers_bf16 is always the unquantised BF16 path. */
    output->policy = QWEN_EXEC_BF16_CANONICAL;
    embedding->values = NULL;
    embedding->tags = NULL;
    memset(embedding, 0, sizeof(*embedding));
}

static int engine_forward_to_layer(const qwen_engine *engine,
                                   const qwen_input *input,
                                   int stop_layer,
                                   qwen_intermediate_state *output,
                                   h3_text_progress progress,
                                   void *progress_opaque,
                                   char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!engine || !input || !output) {
        set_error(error, error_size,
                  "engine_forward_to_layer requires engine, input and output");
        return 0;
    }
    if (!input->token_ids || !input->token_count) {
        set_error(error, error_size, "qwen_input carries no tokens");
        return 0;
    }
    if (stop_layer < 1 || stop_layer > QWEN_RELEASED_LAYERS) {
        set_error(error, error_size,
                  "qwen forward-to-layer supports stop_layer 1..%d (got %d)",
                  QWEN_RELEASED_LAYERS, stop_layer);
        return 0;
    }

    h3_text_embedding embedding;
    memset(&embedding, 0, sizeof(embedding));
    int ok;

    if (input->vision_span_count) {
        if (!input->vision_spans || !input->position_ids || !input->tags) {
            set_error(error, error_size,
                      "multimodal qwen_input needs vision_spans, position_ids "
                      "and tags");
            return 0;
        }
        h3_text_vision_span *spans =
            calloc(input->vision_span_count, sizeof(*spans));
        if (!spans) {
            set_error(error, error_size,
                      "out of memory translating %zu vision span(s)",
                      input->vision_span_count);
            return 0;
        }
        for (size_t index = 0; index < input->vision_span_count; index++) {
            const qwen_vision_span *source = &input->vision_spans[index];
            spans[index].start = source->start;
            spans[index].tokens = source->tokens;
            spans[index].embeddings = source->embeddings;
            spans[index].deepstack[0] = source->deepstack[0];
            spans[index].deepstack[1] = source->deepstack[1];
            spans[index].deepstack[2] = source->deepstack[2];
        }
        ok = h3_text_encode_multimodal_layers_bf16(
            engine->weight_directory, engine->shader_source_path,
            input->token_ids, input->token_count,
            spans, input->vision_span_count,
            input->position_ids, input->tags, stop_layer,
            progress, progress_opaque, &embedding, error, error_size);
        free(spans);
    } else {
        if (input->position_ids || input->tags) {
            set_error(error, error_size,
                      "text-only qwen_input must leave position_ids and tags "
                      "NULL");
            return 0;
        }
        ok = h3_text_encode_layers_bf16(
            engine->weight_directory, engine->shader_source_path,
            input->token_ids, input->token_count, stop_layer,
            progress, progress_opaque, &embedding, error, error_size);
    }
    if (!ok) return 0;

    embedding_into_state(&embedding, output);
    return 1;
}

int qwen_session_forward_to_layer(qwen_session *session,
                                  const qwen_input *input,
                                  int stop_layer,
                                  qwen_intermediate_state *output,
                                  h3_text_progress progress,
                                  void *progress_opaque,
                                  char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!session || !session->engine) {
        set_error(error, error_size,
                  "qwen_session_forward_to_layer requires a session");
        return 0;
    }
    return engine_forward_to_layer(session->engine, input, stop_layer, output,
                                   progress, progress_opaque, error, error_size);
}

int qwen_session_get_h3_conditioning(qwen_session *session,
                                     const qwen_input *input,
                                     qwen_intermediate_state *output,
                                     h3_text_progress progress,
                                     void *progress_opaque,
                                     char *error, size_t error_size) {
    return qwen_session_forward_to_layer(session, input, QWEN_RELEASED_LAYERS,
                                         output, progress, progress_opaque,
                                         error, error_size);
}

int qwen_session_continue_from_intermediate(qwen_session *session,
                                            const qwen_intermediate_state *state,
                                            const uint32_t *position_ids,
                                            qwen_logits *output,
                                            char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!session || !session->engine || !state || !output) {
        set_error(error, error_size,
                  "qwen_session_continue_from_intermediate requires session, "
                  "state and output");
        return 0;
    }
    if (!state->values || !state->tokens ||
        state->hidden_size != QWEN_HIDDEN_SIZE) {
        set_error(error, error_size,
                  "intermediate state must be BF16 [tokens, %u]",
                  QWEN_HIDDEN_SIZE);
        return 0;
    }
    return qwen_lm_decode_tail(session->engine, state->values, state->tokens,
                               position_ids, output, error, error_size);
}

int qwen_engine_forward_full(qwen_engine *engine,
                             const qwen_input *input,
                             qwen_logits *output,
                             h3_text_progress progress,
                             void *progress_opaque,
                             char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!engine || !input || !output) {
        set_error(error, error_size,
                  "qwen_engine_forward_full requires engine, input and output");
        return 0;
    }
    qwen_intermediate_state state;
    if (!engine_forward_to_layer(engine, input, QWEN_RELEASED_LAYERS, &state,
                                 progress, progress_opaque, error, error_size))
        return 0;
    int ok = qwen_lm_decode_tail(engine, state.values, state.tokens,
                                 input->position_ids, output, error, error_size);
    qwen_intermediate_state_free(&state);
    return ok;
}

int h3_conditioning_accepts(const qwen_intermediate_state *state) {
    return state && state->policy == QWEN_EXEC_BF16_CANONICAL;
}

void qwen_intermediate_state_into_h3_text_embedding(
        qwen_intermediate_state *state, h3_text_embedding *output) {
    if (!output) return;
    memset(output, 0, sizeof(*output));
    if (!state) return;
    /* Only BF16-canonical layer-49 states are admissible H3 conditioning
     * (QINT-012). A quantised chat-decode state must not reach the DiT. */
    if (!h3_conditioning_accepts(state)) {
        qwen_intermediate_state_free(state);
        return;
    }
    output->tokens = state->tokens;
    output->width = state->hidden_size;
    output->values = state->values;
    output->tags = state->tags;
    state->values = NULL;
    state->tags = NULL;
    memset(state, 0, sizeof(*state));
}
