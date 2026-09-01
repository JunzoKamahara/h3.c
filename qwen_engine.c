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
    free(session);
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
    /* The bare intermediate state carries no positions; assume sequential text
     * positions. Multimodal callers must use qwen_engine_forward_full(). */
    return qwen_lm_decode_tail(session->engine, state->values, state->tokens,
                               NULL, output, error, error_size);
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

void qwen_intermediate_state_into_h3_text_embedding(
        qwen_intermediate_state *state, h3_text_embedding *output) {
    if (!output) return;
    memset(output, 0, sizeof(*output));
    if (!state) return;
    output->tokens = state->tokens;
    output->width = state->hidden_size;
    output->values = state->values;
    output->tags = state->tags;
    state->values = NULL;
    state->tags = NULL;
    memset(state, 0, sizeof(*state));
}
