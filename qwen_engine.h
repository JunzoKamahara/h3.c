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
#include "h3_tokenizer.h"    /* h3_tokenizer */

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

/* Execution policy that produced a layer-49 state (QINT-012). The layer-49
 * boundary is a shared *semantic* interface, but the layers-0..49 numbers
 * depend on the weight precision: Mixed-W4/BF16 chat decode drifts the
 * layer-49 hidden by ~14 % relative (cos ~0.991, 185/5120 channels with a
 * >10 % RMS change). Only BF16-canonical states may be fed to H3 conditioning;
 * H3 keeps its own BF16 layers-0..49 path regardless. */
typedef enum {
    QWEN_EXEC_BF16_CANONICAL = 0, /* layers 0..49 all BF16 -- H3-admissible   */
    QWEN_EXEC_MIXED_W4_BF16,      /* `mixed` chat decode (q/o/MLP W4, k/v BF16)*/
    QWEN_EXEC_W4_FAST            /* `--fast` pure W4 chat decode              */
} qwen_execution_policy;

/* Canonical layer-49 intermediate state (spec sections 3.1 / 3.2).
 *
 * `values` is the UNNORMALIZED BF16 hidden state after decoder layer 49 -- the
 * final language-model RMSNorm has NOT been applied. shape = [tokens,
 * hidden_size]; hidden_size is always QWEN_HIDDEN_SIZE for the H3 Qwen
 * backbone. `tags` mirrors the H3 presentation rows and is NULL for text-only
 * input. `policy` records how layers 0..49 were executed -- states handed to
 * H3 (`qwen_intermediate_state_into_h3_text_embedding`) must be
 * `QWEN_EXEC_BF16_CANONICAL`. Phase 0 uses caller ownership; free with
 * qwen_intermediate_state_free(). */
typedef struct {
    size_t tokens;
    size_t hidden_size;
    uint16_t *values;
    uint8_t *tags;
    qwen_execution_policy policy;
} qwen_intermediate_state;

/* 1 iff `state` may be used as canonical H3 conditioning (BF16-canonical). */
int h3_conditioning_accepts(const qwen_intermediate_state *state);

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
 * `position_ids` is the axis-major [3, tokens] mRoPE ids that produced the
 * intermediate state (the multimodal branch point -- H3 media generation and
 * this Chat tail consume the very same layer-49 state); pass NULL for the
 * sequential text case. */
int qwen_session_continue_from_intermediate(qwen_session *session,
                                            const qwen_intermediate_state *state,
                                            const uint32_t *position_ids,
                                            qwen_logits *output,
                                            char *error, size_t error_size);

int qwen_engine_forward_full(qwen_engine *engine,
                             const qwen_input *input,
                             qwen_logits *output,
                             h3_text_progress progress,
                             void *progress_opaque,
                             char *error, size_t error_size);

/* Phase 2 -- stateful KV-cache chat session (spec sections 7.2, 13, 14).
 *
 * On the first qwen_session_eval() a session gains a KV-cache context: a
 * persistent GPU context, per-layer K/V caches, token history, current
 * position and the latest logits. Later evals decode incrementally -- only the
 * new tokens flow through the projections and MLP; attention reads the cache.
 * Decoder-layer weights are still streamed per eval (residency is a later
 * phase), so the win is correctness and O(new tokens) compute, not yet speed.
 *
 * The stateless Phase 0/1 entry points above are unaffected by an active KV
 * context; media conditioning still uses the layer-49 hidden, never the cache.
 *
 * Session lifetime helpers: qwen_session_create() / qwen_session_free() are
 * declared above.
 */

/* Append `token_count` text tokens to the context and refresh the latest
 * logits. First call = prefill; later calls = incremental decode. Optional
 * capacity override: env H3_QWEN_KV_CAPACITY (default 4096 tokens). */
int qwen_session_eval(qwen_session *session,
                      const uint32_t *token_ids, size_t token_count,
                      char *error, size_t error_size);

/* Multimodal prefill (P7-005). Must be the first eval on the session: `input`
 * carries token_ids + vision_spans + axis-major position_ids + tags (build it
 * with h3_multimodal_build_chat_input()). The vision embeddings are spliced in
 * and the deepstack residuals applied after layers 0/1/2; later
 * qwen_session_eval() calls decode text with mRoPE positions continuing past
 * the vision grid. Uses the same resident / quantised decode path as
 * qwen_session_eval(). */
int qwen_session_eval_multimodal(qwen_session *session,
                                 const qwen_input *input, char *error,
                                 size_t error_size);

/* Greedy argmax of the latest logits into *token_out. */
int qwen_session_sample(qwen_session *session, uint32_t *token_out,
                        char *error, size_t error_size);

/* QINT-015d -- speculative batch verifier.
 *
 * Append `block` (2..QWEN_VERIFY_MAX tokens) to the context in ONE forward and
 * report, per row, the target's next-token prediction *after* that row:
 *   result.top1[r] = argmax of the logits at the position that follows
 *                    block[0..r]  (so row 0 predicts what comes after block[0],
 *                    row m-1 what comes after the whole block).
 * All `block` rows are left appended to the KV -- verify_block does NOT
 * accept/reject; the caller inspects `result` and rewinds to the accepted
 * frontier with qwen_session_rewind(). Under the Mixed-W4/BF16 policy the
 * projections run through the INT4 decode-batch kernel, i.e. the same weights
 * as scalar decode. Greedy only.
 *
 * The upper bound is 5, matching the INT4 decode-batch Metal kernel
 * (h3_linear_q4_decode_batch handles 2..5 rows). A wider block would fall back
 * to BF16 projections and silently break the "same weights as scalar decode"
 * guarantee, so the kernel limit and this constant are deliberately equal. */
#define QWEN_VERIFY_MAX 5u
typedef struct {
    uint32_t rows;
    uint32_t top1[QWEN_VERIFY_MAX];
    uint32_t top2[QWEN_VERIFY_MAX];
    float top1_logit[QWEN_VERIFY_MAX];
    float top2_logit[QWEN_VERIFY_MAX];
    float margin[QWEN_VERIFY_MAX]; /* top1_logit - top2_logit */
} qwen_verify_result;

int qwen_session_verify_block(qwen_session *session, const uint32_t *block,
                              size_t block_count, qwen_verify_result *result,
                              char *error, size_t error_size);

/* QINT-015h -- EAGLE-3 auxiliary-hidden capture.
 *
 * A learned draft head reuses the target's own residual stream. Call
 * qwen_session_set_aux_layers() with up to QWEN_MAX_AUX_LAYERS decoder-layer
 * ids (each 0..63); from then on every eval snapshots the residual after each
 * of those layers. `count == 0` disables capture -- the default, so nothing
 * changes for callers that never opt in. Must match the layer ids the
 * checkpoint's config names. Cheap: DECODE / PREFILL keep only the frontier
 * (last) row, VERIFY keeps all rows. */
#define QWEN_MAX_AUX_LAYERS 4u
int qwen_session_set_aux_layers(qwen_session *session, const int *layer_ids,
                                size_t count, char *error, size_t error_size);

/* The most recent eval's auxiliary hidden snapshot, laid out aux-major:
 * slot `a` row `r` is at `base + (a * *rows + r) * *hidden` (bf16). `*rows` is
 * 1 for DECODE / PREFILL (the frontier position) and the block length for
 * VERIFY. `*n_aux` and `*layer_ids` echo the set_aux_layers() configuration.
 * Returns NULL with *rows = *n_aux = 0 when capture is off or nothing has been
 * evalled since the last rewind. Valid until the next eval / rewind. */
const uint16_t *qwen_session_aux_hidden(const qwen_session *session,
                                        size_t *rows, size_t *n_aux,
                                        size_t *hidden, const int **layer_ids);

/* The most recent eval's next-token logits (last position), or NULL before the
 * first eval. Valid until the next eval / rewind. */
const qwen_logits *qwen_session_logits(const qwen_session *session);

/* Tokens currently in the context. */
size_t qwen_session_length(const qwen_session *session);

/* The session's token history (prompt + every token evalled so far), length in
 * *length_out. Points into session-owned storage; valid until the next eval or
 * rewind. NULL with *length_out = 0 before the first eval. Used by the
 * speculative coordinator's draft backends (QINT-015). */
const uint32_t *qwen_session_history(const qwen_session *session,
                                     size_t *length_out);

/* Drop the context back to its first `keep` tokens (KV cache + history +
 * position); `keep` must not exceed the current length. */
int qwen_session_rewind(qwen_session *session, size_t keep,
                        char *error, size_t error_size);

/* Validate the session. Phase 2 executes synchronously, so this is a light
 * consistency check kept for lifecycle symmetry with the spec. */
int qwen_session_sync(qwen_session *session, char *error, size_t error_size);

/* Weight residency. By default all 64 decoder layers are pinned in Unified
 * Memory (~62 GB BF16, loaded on the first eval), which is what makes decode
 * fast; if that allocation fails the session falls back to streaming weights
 * from disk per eval. `resident` != 0 forces resident (hard error if it will
 * not fit); `resident` == 0 forces streaming. Must be called before the first
 * qwen_session_eval(). The environment variable H3_QWEN_RESIDENT=0 also forces
 * streaming, H3_QWEN_RESIDENT=1 forces resident. */
int qwen_session_set_resident(qwen_session *session, int resident,
                              char *error, size_t error_size);

/* Bridge to the legacy H3 conditioning type (spec sections 17 / 18). Moves
 * ownership of the BF16 and tag buffers into `output`; `state` is left empty.
 * `gpu_stats` on the legacy struct is zeroed -- it is not part of the
 * conditioning contract. */
void qwen_intermediate_state_into_h3_text_embedding(
        qwen_intermediate_state *state, h3_text_embedding *output);

/* Phase 3 -- chat template (spec section 19); Phase 5 adds the `tools` system
 * block and assistant tool_calls rendering.
 *
 * Renders a message list into the MiniMax-H3 / Qwen3-VL ChatML form and, with a
 * tokenizer, into token ids ready for qwen_session_eval(). */

#define QWEN_TOKEN_IM_START   151644u
#define QWEN_TOKEN_IM_END     151645u  /* end-of-turn / EOS */
#define QWEN_TOKEN_ENDOFTEXT  151643u

typedef enum {
    QWEN_ROLE_SYSTEM,
    QWEN_ROLE_USER,
    QWEN_ROLE_ASSISTANT,
    QWEN_ROLE_TOOL
} qwen_role;

typedef struct {
    qwen_role role;
    const char *content;
    /* Assistant turns only (Phase 5): a JSON array string of prior tool calls,
     * e.g. [{"name":"f","arguments":{...}}], or NULL. Rendered as
     * <tool_call>...</tool_call> markup after `content`. */
    const char *tool_calls_json;
} qwen_chat_message;

/* Render `messages` to a ChatML string. A leading system message becomes the
 * system turn; consecutive tool messages are folded into one user turn of
 * <tool_response> blocks. With add_generation_prompt != 0 the string ends with
 * an open "<|im_start|>assistant\n". Caller frees *text_out. */
int qwen_chat_render(const qwen_chat_message *messages, size_t count,
                     int add_generation_prompt, char **text_out,
                     char *error, size_t error_size);

/* qwen_chat_render() followed by h3_tokenizer_encode(). Caller frees *ids with
 * h3_tokenizer_ids_free(). */
int qwen_chat_tokenize(const h3_tokenizer *tokenizer,
                       const qwen_chat_message *messages, size_t count,
                       int add_generation_prompt, uint32_t **ids,
                       size_t *id_count, char *error, size_t error_size);

/* Phase 5: render / tokenize with a `tools` system block. `tool_jsons` is an
 * array of `tool_count` compact JSON strings, each one tool definition (an
 * object such as {"type":"function","function":{...}}); NULL / 0 is identical
 * to qwen_chat_render(). Mirrors the `{% if tools %}` branch of
 * chat_template.json: the tool signatures and the <tool_call> instructions are
 * folded into the system turn. */
int qwen_chat_render_tools(const qwen_chat_message *messages, size_t count,
                           const char *const *tool_jsons, size_t tool_count,
                           int add_generation_prompt, char **text_out,
                           char *error, size_t error_size);

int qwen_chat_tokenize_tools(const h3_tokenizer *tokenizer,
                             const qwen_chat_message *messages, size_t count,
                             const char *const *tool_jsons, size_t tool_count,
                             int add_generation_prompt, uint32_t **ids,
                             size_t *id_count, char *error, size_t error_size);

#endif
