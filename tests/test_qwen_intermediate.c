/* Phase 0 parity test for the Qwen intermediate-state runtime boundary.
 *
 * Compares the legacy H3 conditioning path
 *   h3_text_encode_bf16() / h3_text_encode_multimodal_bf16()
 * against the new canonical interface
 *   qwen_session_get_h3_conditioning() / qwen_session_forward_to_layer()
 * and requires bit-for-bit identical layer-49 hidden state, shape, dtype and
 * tags. Also checks that stop_layer is honoured.
 *
 * Needs the released Qwen text-encoder weights; run from the repo root:
 *   ./h3_qwen_intermediate_test MiniMax-H3
 * The multimodal case splices deterministic synthetic vision rows, so it needs
 * no vision-encoder weights or fixtures -- it exercises the real 50-layer GPU
 * path and the qwen_input -> presentation translation. */

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_intermediate.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (!result) fail("path allocation failed");
    snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static void progress(int completed, int total, void *opaque) {
    const char *label = opaque;
    if (completed == 1 || completed == total || completed % 10 == 0)
        fprintf(stderr, "  %s: %d/%d layers\n", label, completed, total);
}

static uint16_t round_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Deterministic, reproducible BF16 filler for synthetic vision rows. */
static void fill_pattern(uint16_t *values, size_t count, unsigned seed) {
    for (size_t index = 0; index < count; index++) {
        unsigned mixed = seed * 2654435761u + (unsigned)index * 40503u;
        float centred = (float)(mixed % 4096u) / 4096.0f - 0.5f;
        values[index] = round_bf16_bits(centred * 0.1f);
    }
}

static void compare_bitwise(const qwen_intermediate_state *state,
                            const h3_text_embedding *legacy,
                            size_t token_count, const char *what) {
    require(state->tokens == token_count, "new state token count mismatch");
    require(legacy->tokens == token_count, "legacy token count mismatch");
    require(state->hidden_size == H3_TEXT_HIDDEN_SIZE,
            "new state hidden_size is not 5120");
    require(legacy->width == H3_TEXT_HIDDEN_SIZE, "legacy width is not 5120");
    size_t elements = token_count * H3_TEXT_HIDDEN_SIZE;
    require(state->values != NULL && legacy->values != NULL,
            "missing hidden values");
    if (memcmp(state->values, legacy->values,
               elements * sizeof(uint16_t)) != 0) {
        size_t first = 0;
        for (; first < elements; first++)
            if (state->values[first] != legacy->values[first]) break;
        fprintf(stderr,
                "  %s: first BF16 divergence at element %zu: new=0x%04x "
                "legacy=0x%04x\n",
                what, first, state->values[first], legacy->values[first]);
        fail("layer-49 hidden state is not bit-for-bit identical");
    }
    printf("  %s: %zux%u BF16, bit-for-bit identical\n", what, token_count,
           H3_TEXT_HIDDEN_SIZE);
}

static void text_only_parity(qwen_session *session, const char *model_root) {
    char error[512];
    char *tokenizer_path =
        path_join(model_root, "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, "A red fox walking through snow", 1,
                             &ids, &token_count, error, sizeof(error)))
        fail(error);
    printf("[text-only] %zu tokens\n", token_count);

    h3_text_embedding legacy;
    if (!h3_text_encode_bf16(weights_path, "h3_shaders.metal", ids, token_count,
                             progress, (void *)"legacy", &legacy, error,
                             sizeof(error)))
        fail(error);

    qwen_input input = {0};
    input.token_ids = ids;
    input.token_count = token_count;
    qwen_intermediate_state state;
    if (!qwen_session_get_h3_conditioning(session, &input, &state, progress,
                                          (void *)"qwen", error, sizeof(error)))
        fail(error);

    compare_bitwise(&state, &legacy, token_count, "text-only layer-49");
    require(state.tags == NULL && legacy.tags == NULL,
            "text-only tags must be NULL on both paths");

    /* stop_layer must be honoured: a 3-layer prefix differs from the 50-layer
     * cut and still matches the legacy 3-layer prefix bit-for-bit. */
    qwen_intermediate_state prefix;
    if (!qwen_session_forward_to_layer(session, &input, 3, &prefix, NULL, NULL,
                                       error, sizeof(error)))
        fail(error);
    require(prefix.tokens == token_count && prefix.hidden_size == 5120,
            "3-layer prefix has wrong shape");
    require(memcmp(prefix.values, state.values,
                   token_count * 5120 * sizeof(uint16_t)) != 0,
            "stop_layer=3 output unexpectedly equals stop_layer=50");

    h3_text_embedding legacy_prefix;
    if (!h3_text_encode_layers_bf16(weights_path, "h3_shaders.metal", ids,
                                    token_count, 3, NULL, NULL, &legacy_prefix,
                                    error, sizeof(error)))
        fail(error);
    compare_bitwise(&prefix, &legacy_prefix, token_count, "text-only layer-3");

    h3_text_embedding_free(&legacy);
    h3_text_embedding_free(&legacy_prefix);
    qwen_intermediate_state_free(&state);
    qwen_intermediate_state_free(&prefix);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
}

static void multimodal_parity(qwen_session *session, const char *model_root) {
    char error[512];
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");

    enum { TOKENS = 8, SPAN_START = 2, SPAN_TOKENS = 3, HIDDEN = 5120 };
    uint32_t ids[TOKENS] = {9707, 11, 100, 200, 300, 1879, 30, 13};
    uint32_t position_ids[3 * TOKENS];
    for (size_t axis = 0; axis < 3; axis++)
        for (size_t index = 0; index < TOKENS; index++)
            position_ids[axis * TOKENS + index] = (uint32_t)index;
    uint8_t tags[TOKENS];
    for (size_t index = 0; index < TOKENS; index++)
        tags[index] = (index >= SPAN_START && index < SPAN_START + SPAN_TOKENS)
                          ? 0u
                          : 1u;

    size_t span_elements = (size_t)SPAN_TOKENS * HIDDEN;
    uint16_t *embeddings = malloc(span_elements * sizeof(uint16_t));
    uint16_t *deepstack[3] = {malloc(span_elements * sizeof(uint16_t)),
                              malloc(span_elements * sizeof(uint16_t)),
                              malloc(span_elements * sizeof(uint16_t))};
    if (!embeddings || !deepstack[0] || !deepstack[1] || !deepstack[2])
        fail("out of memory building synthetic vision span");
    fill_pattern(embeddings, span_elements, 1u);
    fill_pattern(deepstack[0], span_elements, 2u);
    fill_pattern(deepstack[1], span_elements, 3u);
    fill_pattern(deepstack[2], span_elements, 4u);

    h3_text_vision_span legacy_span = {0};
    legacy_span.start = SPAN_START;
    legacy_span.tokens = SPAN_TOKENS;
    legacy_span.embeddings = embeddings;
    legacy_span.deepstack[0] = deepstack[0];
    legacy_span.deepstack[1] = deepstack[1];
    legacy_span.deepstack[2] = deepstack[2];

    h3_text_embedding legacy;
    if (!h3_text_encode_multimodal_bf16(
            weights_path, "h3_shaders.metal", ids, TOKENS, &legacy_span, 1,
            position_ids, tags, progress, (void *)"legacy-mm", &legacy, error,
            sizeof(error)))
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
    if (!qwen_session_get_h3_conditioning(session, &input, &state, progress,
                                          (void *)"qwen-mm", error,
                                          sizeof(error)))
        fail(error);

    compare_bitwise(&state, &legacy, TOKENS, "multimodal layer-49");
    require(state.tags != NULL && legacy.tags != NULL,
            "multimodal tags must be present on both paths");
    require(memcmp(state.tags, legacy.tags, TOKENS) == 0,
            "multimodal tags differ between paths");
    require(memcmp(state.tags, tags, TOKENS) == 0,
            "multimodal tags do not round-trip the input");
    printf("  multimodal tags: %d rows, identical\n", TOKENS);

    h3_text_embedding_free(&legacy);
    qwen_intermediate_state_free(&state);
    free(embeddings);
    free(deepstack[0]);
    free(deepstack[1]);
    free(deepstack[2]);
    free(weights_path);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];

    qwen_engine *engine = NULL;
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");
    if (!qwen_engine_open(&engine, weights_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    free(weights_path);

    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 0, error, sizeof(error)))
        fail(error);

    text_only_parity(session, model_root);
    multimodal_parity(session, model_root);

    qwen_session_free(session);
    qwen_engine_close(engine);
    puts("ok: qwen intermediate-state parity (text-only + multimodal + "
         "stop_layer)");
    return 0;
}
