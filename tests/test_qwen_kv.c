/* Phase 2 test for the Qwen KV-cache chat session.
 *
 * Anchors the cache against the Phase 1 full-prompt forward:
 *   1. prefill parity   -- session eval(prompt) logits == forward_full(prompt),
 *      and each subsequent greedy step matches forward_full over the grown
 *      sequence (bit-for-bit; prefill reduces the cached-attention kernel to
 *      plain causal);
 *   2. chunked prefill   -- eval(a) then eval(b) == eval(a+b);
 *   3. rewind + multi-turn -- rewind() then re-eval reproduces the earlier
 *      step's logits;
 *   4. determinism       -- two sessions replaying the same tokens agree.
 *
 *   ./h3_qwen_kv_test MiniMax-H3 [PROMPT]
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUNDS 2

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_kv.c: %s\n", message);
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

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double worst = 0.0;
    for (size_t index = 0; index < n; index++) {
        double delta = fabs((double)a[index] - (double)b[index]);
        if (delta > worst) worst = delta;
    }
    return worst;
}

static float *dup_logits(const qwen_logits *logits) {
    float *copy = malloc(logits->vocab * sizeof(*copy));
    if (!copy) fail("logits copy allocation failed");
    memcpy(copy, logits->values, logits->vocab * sizeof(*copy));
    return copy;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";

    char error[512];
    char *tokenizer_path =
        path_join(model_root, "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");

    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t prompt_len = 0;
    if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &prompt_len, error,
                             sizeof(error)))
        fail(error);
    require(prompt_len >= 4, "need a prompt of at least 4 tokens");
    printf("prompt: \"%s\" -> %zu tokens\n", prompt, prompt_len);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);

    /* Build the reference sequence with the Phase 1 full forward and record
     * the logits at every step. */
    size_t max_len = prompt_len + ROUNDS;
    uint32_t *chain = malloc(max_len * sizeof(*chain));
    float **ref_logits = calloc(ROUNDS + 1, sizeof(*ref_logits));
    uint32_t ref_next[ROUNDS + 1];
    if (!chain || !ref_logits) fail("allocation failed");
    memcpy(chain, ids, prompt_len * sizeof(*chain));
    size_t len = prompt_len;
    for (int round = 0; round <= ROUNDS; round++) {
        qwen_input input = {0};
        input.token_ids = chain;
        input.token_count = len;
        qwen_logits logits;
        if (!qwen_engine_forward_full(engine, &input, &logits, NULL, NULL,
                                      error, sizeof(error)))
            fail(error);
        ref_logits[round] = dup_logits(&logits);
        ref_next[round] = logits.argmax_token;
        qwen_logits_free(&logits);
        if (round < ROUNDS) chain[len++] = ref_next[round];
    }
    printf("reference (forward_full) next tokens:");
    for (int round = 0; round <= ROUNDS; round++)
        printf(" %u", ref_next[round]);
    putchar('\n');

    /* (1) KV session: prefill + greedy decode must match the reference. */
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval(session, ids, prompt_len, error, sizeof(error)))
        fail(error);
    require(qwen_session_length(session) == prompt_len, "prefill length wrong");
    for (int round = 0; round <= ROUNDS; round++) {
        const qwen_logits *got = qwen_session_logits(session);
        require(got != NULL, "session has no logits after eval");
        double worst = max_abs_diff(got->values, ref_logits[round],
                                    got->vocab);
        printf("  step %d: argmax kv=%u ref=%u, max|dlogit|=%.3g\n", round,
               got->argmax_token, ref_next[round], worst);
        require(got->argmax_token == ref_next[round],
                "KV next-token disagrees with forward_full");
        require(memcmp(got->values, ref_logits[round],
                       got->vocab * sizeof(float)) == 0,
                "KV logits are not bit-for-bit equal to forward_full");
        if (round < ROUNDS) {
            uint32_t next = 0;
            if (!qwen_session_sample(session, &next, error, sizeof(error)))
                fail(error);
            require(next == ref_next[round], "sample() disagrees with argmax");
            if (!qwen_session_eval(session, &next, 1, error, sizeof(error)))
                fail(error);
        }
    }
    printf("(1) prefill + %d-step greedy decode match forward_full "
           "bit-for-bit\n", ROUNDS);

    /* (2) chunked prefill: eval(a) + eval(b) == eval(a+b). */
    size_t split = prompt_len / 2;
    qwen_session *chunked = NULL;
    if (!qwen_session_create(&chunked, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval(chunked, ids, split, error, sizeof(error)) ||
        !qwen_session_eval(chunked, ids + split, prompt_len - split, error,
                           sizeof(error)))
        fail(error);
    require(qwen_session_length(chunked) == prompt_len,
            "chunked prefill length wrong");
    {
        const qwen_logits *got = qwen_session_logits(chunked);
        require(got != NULL, "chunked session has no logits");
        require(memcmp(got->values, ref_logits[0],
                       got->vocab * sizeof(float)) == 0,
                "chunked prefill logits differ from single-shot prefill");
    }
    printf("(2) split prefill %zu+%zu matches single prefill bit-for-bit\n",
           split, prompt_len - split);

    /* (3) rewind + multi-turn: drop back to the prompt, replay, reproduce. */
    if (!qwen_session_rewind(session, prompt_len, error, sizeof(error)))
        fail(error);
    require(qwen_session_length(session) == prompt_len, "rewind length wrong");
    if (!qwen_session_eval(session, &ref_next[0], 1, error, sizeof(error)))
        fail(error);
    require(memcmp(qwen_session_logits(session)->values, ref_logits[1],
                   qwen_session_logits(session)->vocab * sizeof(float)) == 0,
            "post-rewind eval does not reproduce the earlier logits");
    printf("(3) rewind to %zu then re-eval reproduces step 1 bit-for-bit\n",
           prompt_len);

    /* (4) determinism across sessions. */
    qwen_session *twin = NULL;
    if (!qwen_session_create(&twin, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval(twin, ids, prompt_len, error, sizeof(error)) ||
        !qwen_session_eval(twin, &ref_next[0], 1, error, sizeof(error)))
        fail(error);
    require(memcmp(qwen_session_logits(twin)->values, ref_logits[1],
                   qwen_session_logits(twin)->vocab * sizeof(float)) == 0,
            "independent session is not deterministic");
    printf("(4) independent session replays identical logits\n");

    qwen_session_free(twin);
    qwen_session_free(chunked);
    qwen_session_free(session);
    for (int round = 0; round <= ROUNDS; round++) free(ref_logits[round]);
    free(ref_logits);
    free(chain);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    puts("ok: qwen Phase 2 KV cache (prefill parity + chunk + rewind + "
         "determinism)");
    return 0;
}
