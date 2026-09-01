#ifndef QWEN_ENGINE_H
#define QWEN_ENGINE_H

/* Qwen intermediate-state runtime boundary (design spec Phase 0).
 *
 * MiniMax-H3 runs the first 50 of Qwen3-VL's 64 decoder layers and treats the
 * hidden state after layer 49 -- before the final RMSNorm -- as the H3
 * conditioning signal. This header promotes that cut to a first-class internal
 * interface shared by H3 media generation and, in later phases, Qwen Chat/VLM.
 *
 * Phase 0 scope: define qwen_input / qwen_intermediate_state, expose
 * qwen_session_forward_to_layer() and qwen_session_get_h3_conditioning(), and
 * route the existing text encoder through them with no change in numerical
 * output. Layers 50..63, the LM head, KV cache and HTTP are out of scope. */

#include "h3_text_encoder.h" /* h3_text_embedding, h3_text_progress */

#include <stddef.h>
#include <stdint.h>

#define QWEN_HIDDEN_SIZE 5120u

/* stop_layer for the canonical H3 conditioning cut: run layers 0..49. */
#define QWEN_RELEASED_LAYERS 50

/* Vision presentation span. Field-for-field compatible with
 * h3_text_vision_span: the base token embeddings at [start, start + tokens)
 * are replaced by `embeddings`, and the three `deepstack` rows are added after
 * language layers 0, 1 and 2. Every buffer is BF16 [tokens, QWEN_HIDDEN_SIZE],
 * row-major, and owned by the caller. */
typedef struct {
    size_t start;
    size_t tokens;
    const uint16_t *embeddings;
    const uint16_t *deepstack[3];
} qwen_vision_span;

/* Unified Qwen3-VL input (spec section 9). A text-only caller leaves every
 * optional field NULL / zero; that selects the sequential-position path and is
 * bit-for-bit the legacy h3_text_encode_bf16() input. */
typedef struct {
    const uint32_t *token_ids;
    size_t token_count;

    const qwen_vision_span *vision_spans;
    size_t vision_span_count;

    /* Axis-major mRoPE position ids, [3, token_count]. Required when
     * vision_span_count > 0; must be NULL for the text-only path. */
    const uint32_t *position_ids;

    /* Per-row H3 modality tag, [token_count], values 0..2. Required when
     * vision_span_count > 0; must be NULL for the text-only path. */
    const uint8_t *tags;
} qwen_input;

/* Canonical layer-49 intermediate state (spec sections 3.1 / 3.2).
 *
 * `values` is the UNNORMALIZED BF16 hidden state after decoder layer 49 -- the
 * final language-model RMSNorm has NOT been applied. shape = [tokens,
 * hidden_size]; hidden_size is always QWEN_HIDDEN_SIZE for the H3 Qwen
 * backbone. `tags` mirrors the H3 presentation rows and is NULL for text-only
 * input. Phase 0 uses caller ownership; free with
 * qwen_intermediate_state_free(). */
typedef struct {
    size_t tokens;
    size_t hidden_size;
    uint16_t *values;
    uint8_t *tags;
} qwen_intermediate_state;

void qwen_intermediate_state_free(qwen_intermediate_state *state);

/* Phase 1 output: full-vocabulary logits for the final prompt position, plus a
 * CPU argmax convenience. `values` is [vocab] F32 (the next-token distribution
 * for the last input token); one greedy decode step is `argmax_token`. */
typedef struct {
    size_t vocab;           /* Always 151936 for the H3 Qwen backbone. */
    float *values;          /* [vocab] F32, owned. */
    uint32_t argmax_token;  /* CPU argmax over values. */
} qwen_logits;

void qwen_logits_free(qwen_logits *logits);

/* The engine owns the location of the shared Qwen3-VL checkpoint. A single
 * engine backs both Chat/VLM and H3 conditioning; the checkpoint is never
 * loaded twice (spec section 16). Phase 0 keeps the engine a thin handle over
 * the weight directory and shader path -- weight residency and the KV cache
 * arrive in Phase 1 / Phase 2. */
typedef struct qwen_engine qwen_engine;
typedef struct qwen_session qwen_session;

int qwen_engine_open(qwen_engine **out,
                     const char *weight_directory,
                     const char *shader_source_path,
                     char *error, size_t error_size);
void qwen_engine_close(qwen_engine *engine);

int qwen_session_create(qwen_session **out, qwen_engine *engine,
                        char *error, size_t error_size);
void qwen_session_free(qwen_session *session);

/* Generic forward-to-layer (spec section 3.3). stop_layer == 50 runs decoder
 * layers 0..49. Phase 0 accepts stop_layer in [1, 50]. `progress` is optional.
 * On success `output` owns freshly allocated buffers. */
int qwen_session_forward_to_layer(qwen_session *session,
                                  const qwen_input *input,
                                  int stop_layer,
                                  qwen_intermediate_state *output,
                                  h3_text_progress progress,
                                  void *progress_opaque,
                                  char *error, size_t error_size);

/* Canonical H3 conditioning interface (spec section 3.4). Fixes stop_layer to
 * 50 internally; the H3 side never names a layer. The result is semantically
 * and numerically compatible with the legacy h3_text_encode_bf16() and
 * h3_text_encode_multimodal_bf16() paths. */
int qwen_session_get_h3_conditioning(qwen_session *session,
                                     const qwen_input *input,
                                     qwen_intermediate_state *output,
                                     h3_text_progress progress,
                                     void *progress_opaque,
                                     char *error, size_t error_size);

/* Phase 1 -- Full 64-layer Chat LLM.
 *
 * Continue from the canonical layer-49 intermediate state through decoder
 * layers 50..63, the final language-model RMSNorm and lm_head, producing
 * next-token logits for the last position (spec sections 10-12). No KV cache:
 * this is a full-prompt forward. The layer-49 boundary and all H3 conditioning
 * output are untouched.
 *
 * `qwen_session_continue_from_intermediate` assumes sequential text positions
 * (the bare intermediate state carries none); use `qwen_engine_forward_full`
 * when the input has mRoPE position ids. */
int qwen_session_continue_from_intermediate(qwen_session *session,
                                            const qwen_intermediate_state *state,
                                            qwen_logits *output,
                                            char *error, size_t error_size);

int qwen_engine_forward_full(qwen_engine *engine,
                             const qwen_input *input,
                             qwen_logits *output,
                             h3_text_progress progress,
                             void *progress_opaque,
                             char *error, size_t error_size);

/* Bridge to the legacy H3 conditioning type (spec sections 17 / 18). Moves
 * ownership of the BF16 and tag buffers into `output`; `state` is left empty.
 * `gpu_stats` on the legacy struct is zeroed -- it is not part of the
 * conditioning contract. */
void qwen_intermediate_state_into_h3_text_embedding(
        qwen_intermediate_state *state, h3_text_embedding *output);

#endif
