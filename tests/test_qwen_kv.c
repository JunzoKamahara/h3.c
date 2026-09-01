/* Phase 2 test for the Qwen KV-cache chat session.
 *
 * Anchors the cache against the Phase 1 full-prompt forward:
 *   1. prefill parity   -- session eval(prompt) logits == forward_full(prompt)
 *      bit-for-bit; each subsequent greedy step matches forward_full over the
 *      grown sequence (argmax + tight relative error: incremental decode runs
 *      rows=1 through the batch-1 GEMV kernel, a different reduction order from
 *      the rows>1 tiled kernel forward_full uses);
 *   2. chunked prefill   -- eval(a) then eval(b) == eval(a+b) bit-for-bit;
 *   3. rewind + multi-turn -- rewind() then re-eval reproduces the earlier
 *      step's logits (argmax + tight relative error);
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

/* Relative L2 distance ||a - b|| / ||b||. Incremental decode (rows=1) runs the
 * batch-1 GEMV kernel, whose reduction order differs from the rows>1 tiled
 * kernel used by forward_full; the two agree to a tight relative bound and
 * always pick the same argmax, but not bit-for-bit. */
static double rel_l2(const float *a, const float *b, size_t n) {
    double se = 0.0, sr = 0.0;
    for (size_t index = 0; index < n; index++) {
        double delta = (double)a[index] - (double)b[index];
        se += delta * delta;
        sr += (double)b[index] * (double)b[index];
    }
    return sqrt(se / (sr > 1e-30 ? sr : 1e-30));
}

/* Decode-vs-forward_full tolerance. Each cached K/V vector is written by the
 * rows=1 GEMV kernel, so its small delta from the rows>1 tiled path compounds
 * mildly across decode positions (observed rel_l2 ~4e-3 at step 1, ~1.6e-2 at
 * step 2; argmax matches at every step). 3e-2 covers the compounding with
 * headroom while still catching real divergence (a genuine bug moves argmax or
 * pushes rel_l2 toward 1). */
#define KV_DECODE_REL_TOL 3e-2

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
    if (!qwen_session_set_resident(session, 0, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval(session, ids, prompt_len, error, sizeof(error)))
        fail(error);
    require(qwen_session_length(session) == prompt_len, "prefill length wrong");
    for (int round = 0; round <= ROUNDS; round++) {
        const qwen_logits *got = qwen_session_logits(session);
        require(got != NULL, "session has no logits after eval");
        double worst = max_abs_diff(got->values, ref_logits[round],
                                    got->vocab);
        double rel = rel_l2(got->values, ref_logits[round], got->vocab);
        printf("  step %d: argmax kv=%u ref=%u, max|dlogit|=%.3g rel_l2=%.2e\n",
               round, got->argmax_token, ref_next[round], worst, rel);
        require(got->argmax_token == ref_next[round],
                "KV next-token disagrees with forward_full");
        if (round == 0) {
            /* Prefill: rows>1 tiled kernel on both sides -- bit-for-bit. */
            require(memcmp(got->values, ref_logits[round],
                           got->vocab * sizeof(float)) == 0,
                    "KV prefill logits are not bit-for-bit equal to "
                    "forward_full");
        } else {
            /* Decode: rows=1 GEMV kernel -- argmax + tight relative bound. */
            require(rel < KV_DECODE_REL_TOL,
                    "KV decode logits diverge from forward_full beyond "
                    "tolerance");
        }
        if (round < ROUNDS) {
            uint32_t next = 0;
            if (!qwen_session_sample(session, &next, error, sizeof(error)))
                fail(error);
            require(next == ref_next[round], "sample() disagrees with argmax");
            if (!qwen_session_eval(session, &next, 1, error, sizeof(error)))
                fail(error);
        }
    }
    printf("(1) prefill matches forward_full bit-for-bit; %d-step greedy decode "
           "matches on argmax within %.0e relative\n", ROUNDS,
           (double)KV_DECODE_REL_TOL);

    /* (2) chunked prefill: eval(a) + eval(b) == eval(a+b). */
    size_t split = prompt_len / 2;
    qwen_session *chunked = NULL;
    if (!qwen_session_create(&chunked, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(chunked, 0, error, sizeof(error)))
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
    float *rewind_logits = NULL;
    {
        const qwen_logits *got = qwen_session_logits(session);
        require(got->argmax_token == ref_next[1],
                "post-rewind eval argmax does not match the earlier step");
        require(rel_l2(got->values, ref_logits[1], got->vocab) <
                    KV_DECODE_REL_TOL,
                "post-rewind eval does not reproduce the earlier logits");
        rewind_logits = dup_logits(got);
    }
    printf("(3) rewind to %zu then re-eval reproduces step 1 "
           "(argmax + tight relative)\n", prompt_len);

    /* (4) determinism across sessions. */
    qwen_session *twin = NULL;
    if (!qwen_session_create(&twin, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(twin, 0, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval(twin, ids, prompt_len, error, sizeof(error)) ||
        !qwen_session_eval(twin, &ref_next[0], 1, error, sizeof(error)))
        fail(error);
    /* Same kernel, same KV history as the post-rewind session -> bit-for-bit. */
    require(memcmp(qwen_session_logits(twin)->values, rewind_logits,
                   qwen_session_logits(twin)->vocab * sizeof(float)) == 0,
            "independent session is not deterministic");
    printf("(4) independent session replays identical logits bit-for-bit\n");

    free(rewind_logits);
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
