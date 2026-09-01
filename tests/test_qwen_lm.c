/* Phase 1 test for the Qwen 64-layer Chat LLM tail.
 *
 * Without an MLX logits fixture this checks the properties Phase 1 owns:
 *   1. boundary decomposition -- qwen_engine_forward_full() is bit-for-bit
 *      equal to continue_from_intermediate(get_h3_conditioning());
 *   2. determinism -- two forward_full() runs agree bit-for-bit;
 *   3. the layer-49 intermediate state is still shape [N, 5120] BF16.
 * With an optional golden (argv[2] = safetensors holding F32 "x.logits"
 * [151936] for the last position) it also does a bounded numeric compare.
 *
 *   ./h3_qwen_lm_test MiniMax-H3 [GOLDEN.safetensors] [PROMPT] [--decode]
 */

#include "h3_safetensors.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_lm.c: %s\n", message);
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
    (void)opaque;
    if (completed == 1 || completed == total || completed % 10 == 0)
        fprintf(stderr, "  layer-49 encode: %d/%d\n", completed, total);
}

static void compare_golden(const char *path, const qwen_logits *logits) {
    h3_st_header header;
    char error[512];
    if (!h3_st_read_header(path, &header, &error[0], sizeof(error))) fail(error);
    const h3_st_tensor *tensor = h3_st_find(&header, "x.logits");
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != (uint64_t)logits->vocab) {
        h3_st_free_header(&header);
        fail("golden logits fixture has the wrong schema");
    }
    float *want = malloc(logits->vocab * sizeof(*want));
    if (!want) fail("golden allocation failed");
    if (!h3_st_read_data(&header, tensor, want, logits->vocab * sizeof(*want),
                         &error[0], sizeof(error)))
        fail(error);
    double max_abs = 0.0, max_ref = 0.0, sq_err = 0.0, sq_ref = 0.0;
    uint32_t want_argmax = 0;
    for (size_t index = 0; index < logits->vocab; index++) {
        double delta = (double)logits->values[index] - (double)want[index];
        if (fabs(delta) > max_abs) max_abs = fabs(delta);
        if (fabs(want[index]) > max_ref) max_ref = fabs(want[index]);
        sq_err += delta * delta;
        sq_ref += (double)want[index] * (double)want[index];
        if (want[index] > want[want_argmax]) want_argmax = (uint32_t)index;
    }
    double relative_l2 = sqrt(sq_err / (sq_ref > 1e-24 ? sq_ref : 1e-24));
    printf("golden logits: argmax native=%u reference=%u, relative-L2 %.6g, "
           "absolute-max %.6g\n",
           logits->argmax_token, want_argmax, relative_l2, max_abs);
    require(logits->argmax_token == want_argmax,
            "native and reference next-token argmax differ");
    require(relative_l2 < 0.05, "logits exceed the reference error bound");
    free(want);
    h3_st_free_header(&header);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *golden = (argc > 2 && strcmp(argv[2], "--decode") != 0)
                             ? argv[2]
                             : NULL;
    const char *prompt = (argc > 3 && strcmp(argv[3], "--decode") != 0)
                             ? argv[3]
                             : "The capital of France is";
    int decode = 0;
    for (int index = 2; index < argc; index++)
        if (strcmp(argv[index], "--decode") == 0) decode = 1;

    char error[512];
    char *tokenizer_path =
        path_join(model_root, "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");

    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count, error,
                             sizeof(error)))
        fail(error);
    printf("prompt: \"%s\" -> %zu tokens\n", prompt, token_count);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 0, error, sizeof(error)))
        fail(error);

    qwen_input input = {0};
    input.token_ids = ids;
    input.token_count = token_count;

    /* (1) forward_full */
    qwen_logits full;
    if (!qwen_engine_forward_full(engine, &input, &full, progress, NULL, error,
                                 sizeof(error)))
        fail(error);
    require(full.vocab == 151936, "logits vocab is not 151936");
    require(full.argmax_token < 151936, "argmax token is outside the vocabulary");

    /* (2) get_h3_conditioning + continue_from_intermediate */
    qwen_intermediate_state state;
    if (!qwen_session_get_h3_conditioning(session, &input, &state, progress,
                                          NULL, error, sizeof(error)))
        fail(error);
    require(state.tokens == token_count && state.hidden_size == 5120,
            "layer-49 intermediate state changed shape");
    qwen_logits split;
    if (!qwen_session_continue_from_intermediate(session, &state, NULL, &split,
                                                 error, sizeof(error)))
        fail(error);

    require(full.argmax_token == split.argmax_token,
            "forward_full and continue_from_intermediate disagree on the token");
    require(memcmp(full.values, split.values,
                   full.vocab * sizeof(float)) == 0,
            "forward_full and continue_from_intermediate logits differ");
    printf("boundary: forward_full == continue_from_intermediate(layer-49), "
           "next token = %u\n",
           full.argmax_token);

    /* (3) determinism */
    qwen_logits again;
    if (!qwen_engine_forward_full(engine, &input, &again, NULL, NULL, error,
                                 sizeof(error)))
        fail(error);
    require(again.argmax_token == full.argmax_token &&
                memcmp(again.values, full.values,
                       full.vocab * sizeof(float)) == 0,
            "forward_full is not run-to-run deterministic");
    printf("determinism: two forward_full runs are bit-for-bit identical\n");

    uint32_t next = full.argmax_token;
    char *piece = h3_tokenizer_decode(tokenizer, &next, 1, error, sizeof(error));
    if (piece) {
        printf("greedy next token: %u -> \"%s\"\n", next, piece);
        free(piece);
    }

    if (golden) compare_golden(golden, &full);

    if (decode) {
        size_t grown = token_count;
        uint32_t *chain = malloc((token_count + 16) * sizeof(*chain));
        if (!chain) fail("decode allocation failed");
        memcpy(chain, ids, token_count * sizeof(*chain));
        for (int step = 0; step < 16; step++) {
            qwen_input step_input = {0};
            step_input.token_ids = chain;
            step_input.token_count = grown;
            qwen_logits step_logits;
            if (!qwen_engine_forward_full(engine, &step_input, &step_logits,
                                          NULL, NULL, error, sizeof(error)))
                fail(error);
            chain[grown++] = step_logits.argmax_token;
            qwen_logits_free(&step_logits);
        }
        char *text = h3_tokenizer_decode(tokenizer, chain, grown, error,
                                         sizeof(error));
        if (text) {
            printf("greedy 16-step: \"%s\"\n", text);
            free(text);
        }
        free(chain);
    }

    qwen_logits_free(&full);
    qwen_logits_free(&split);
    qwen_logits_free(&again);
    qwen_intermediate_state_free(&state);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    puts("ok: qwen Phase 1 LM tail (boundary + determinism)");
    return 0;
}
