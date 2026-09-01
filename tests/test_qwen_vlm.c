/* Phase 7 -- VLM: the multimodal layer-49 intermediate state is shared by H3
 * conditioning and the Chat tail, all the way to a chat token.
 *
 *   1. qwen_session_get_h3_conditioning(multimodal input) is bit-for-bit the
 *      legacy H3 path h3_text_encode_multimodal_bf16() -- the exact same
 *      layer-49 state H3 media generation consumes;
 *   2. qwen_session_continue_from_intermediate(that state, mRoPE positions)
 *      equals qwen_engine_forward_full(multimodal input) bit-for-bit -- the
 *      Chat tail on the shared state matches a one-shot 64-layer forward;
 *   3. the decoded token is a valid vocab id and forward_full is deterministic.
 *
 * Vision rows are synthesised (deterministic BF16), so no image codec or
 * vision-encoder fixture is needed; the real 50+14 layer GPU path runs.
 *
 *   ./h3_qwen_vlm_test MiniMax-H3
 */

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_vlm.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (!result) fail("alloc");
    snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static uint16_t round_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void fill_pattern(uint16_t *values, size_t count, unsigned seed) {
    for (size_t index = 0; index < count; index++) {
        unsigned mixed = seed * 2654435761u + (unsigned)index * 40503u;
        float centred = (float)(mixed % 4096u) / 4096.0f - 0.5f;
        values[index] = round_bf16_bits(centred * 0.1f);
    }
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];
    char *weights = path_join(model_root, "FL2VA/text_encoder");

    enum { TOKENS = 10, SPAN_START = 3, SPAN_TOKENS = 4, HIDDEN = 5120 };
    uint32_t ids[TOKENS] = {9707, 25, 151652, 100, 200, 300, 400, 151653,
                            1879, 13};
    uint32_t position_ids[3 * TOKENS];
    for (size_t axis = 0; axis < 3; axis++)
        for (size_t i = 0; i < TOKENS; i++)
            position_ids[axis * TOKENS + i] = (uint32_t)i;
    uint8_t tags[TOKENS];
    for (size_t i = 0; i < TOKENS; i++)
        tags[i] = (i >= SPAN_START && i < SPAN_START + SPAN_TOKENS) ? 0u : 1u;

    size_t span_elems = (size_t)SPAN_TOKENS * HIDDEN;
    uint16_t *embeddings = malloc(span_elems * sizeof(uint16_t));
    uint16_t *deepstack[3] = {malloc(span_elems * sizeof(uint16_t)),
                              malloc(span_elems * sizeof(uint16_t)),
                              malloc(span_elems * sizeof(uint16_t))};
    if (!embeddings || !deepstack[0] || !deepstack[1] || !deepstack[2])
        fail("out of memory building synthetic vision span");
    fill_pattern(embeddings, span_elems, 7u);
    fill_pattern(deepstack[0], span_elems, 8u);
    fill_pattern(deepstack[1], span_elems, 9u);
    fill_pattern(deepstack[2], span_elems, 10u);

    /* ---- legacy H3 conditioning path ---- */
    h3_text_vision_span legacy_span = {0};
    legacy_span.start = SPAN_START;
    legacy_span.tokens = SPAN_TOKENS;
    legacy_span.embeddings = embeddings;
    legacy_span.deepstack[0] = deepstack[0];
    legacy_span.deepstack[1] = deepstack[1];
    legacy_span.deepstack[2] = deepstack[2];
    h3_text_embedding h3_state;
    if (!h3_text_encode_multimodal_bf16(weights, "h3_shaders.metal", ids,
                                        TOKENS, &legacy_span, 1, position_ids,
                                        tags, NULL, NULL, &h3_state, error,
                                        sizeof(error)))
        fail(error);
    require(h3_state.tokens == TOKENS && h3_state.width == HIDDEN,
            "H3 multimodal state shape");

    /* ---- new runtime path ---- */
    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 0, error, sizeof(error)))
        fail(error);

    qwen_vision_span span = {0};
    span.start = SPAN_START;
    span.tokens = SPAN_TOKENS;
    span.embeddings = embeddings;
    span.deepstack[0] = deepstack[0];
    span.deepstack[1] = deepstack[1];
    span.deepstack[2] = deepstack[2];
    qwen_input input = {0};
    input.token_ids = ids;
    input.token_count = TOKENS;
    input.vision_spans = &span;
    input.vision_span_count = 1;
    input.position_ids = position_ids;
    input.tags = tags;

    qwen_intermediate_state state;
    if (!qwen_session_get_h3_conditioning(session, &input, &state, NULL, NULL,
                                          error, sizeof(error)))
        fail(error);

    /* (1) the layer-49 state is identical to what H3 media generation uses */
    require(state.tokens == TOKENS && state.hidden_size == HIDDEN,
            "runtime multimodal state shape");
    require(memcmp(state.values, h3_state.values,
                   (size_t)TOKENS * HIDDEN * sizeof(uint16_t)) == 0,
            "multimodal layer-49 state differs between H3 and the runtime");
    require(state.tags && h3_state.tags &&
                memcmp(state.tags, h3_state.tags, TOKENS) == 0,
            "multimodal tags differ");
    printf("(1) multimodal layer-49 state shared with H3: %dx%d BF16 "
           "bit-for-bit\n", TOKENS, HIDDEN);

    /* (2) Chat tail on that shared state == one-shot 64-layer forward */
    qwen_logits from_state;
    if (!qwen_session_continue_from_intermediate(session, &state, position_ids,
                                                 &from_state, error,
                                                 sizeof(error)))
        fail(error);
    qwen_logits full;
    if (!qwen_engine_forward_full(engine, &input, &full, NULL, NULL, error,
                                  sizeof(error)))
        fail(error);
    require(from_state.vocab == full.vocab && full.vocab == 151936,
            "logits vocab");
    require(from_state.argmax_token == full.argmax_token,
            "continue-from-state and forward_full pick different tokens");
    require(memcmp(from_state.values, full.values,
                   full.vocab * sizeof(float)) == 0,
            "continue-from-state and forward_full logits differ");
    require(full.argmax_token < 151936, "argmax token out of vocab");
    printf("(2) chat tail on the shared state == forward_full, next token = "
           "%u\n", full.argmax_token);

    /* (3) determinism */
    qwen_logits again;
    if (!qwen_engine_forward_full(engine, &input, &again, NULL, NULL, error,
                                  sizeof(error)))
        fail(error);
    require(again.argmax_token == full.argmax_token &&
                memcmp(again.values, full.values,
                       full.vocab * sizeof(float)) == 0,
            "multimodal forward_full is not deterministic");
    printf("(3) multimodal forward_full is deterministic\n");

    qwen_logits_free(&from_state);
    qwen_logits_free(&full);
    qwen_logits_free(&again);
    qwen_intermediate_state_free(&state);
    h3_text_embedding_free(&h3_state);
    qwen_session_free(session);
    qwen_engine_close(engine);
    free(embeddings);
    free(deepstack[0]);
    free(deepstack[1]);
    free(deepstack[2]);
    free(weights);
    puts("ok: qwen Phase 7 VLM (multimodal layer-49 state shared by H3 and "
         "Chat)");
    return 0;
}
