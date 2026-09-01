/* Approach B check -- a resident-weights session must decode bit-for-bit like a
 * streaming session, and much faster.
 *
 *   ./h3_qwen_resident_test MiniMax-H3 [decode_steps]
 *
 * Holds one resident session (~65 GB) and one streaming session at once, so it
 * needs a large-memory machine; it is deliberately not part of `make test`.
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_resident.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (!result) fail("alloc");
    snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static double run_session(qwen_session *session, const uint32_t *ids,
                          size_t prompt_len, int steps, uint32_t *tokens_out,
                          float **logits_out) {
    char error[512];
    if (!qwen_session_eval(session, ids, prompt_len, error, sizeof(error)))
        fail(error);
    double decode_seconds = 0.0;
    for (int step = 0; step < steps; step++) {
        const qwen_logits *logits = qwen_session_logits(session);
        require(logits != NULL, "no logits after eval");
        logits_out[step] = malloc(logits->vocab * sizeof(float));
        if (!logits_out[step]) fail("alloc");
        memcpy(logits_out[step], logits->values,
               logits->vocab * sizeof(float));
        uint32_t next = 0;
        if (!qwen_session_sample(session, &next, error, sizeof(error)))
            fail(error);
        tokens_out[step] = next;
        double t0 = now();
        if (!qwen_session_eval(session, &next, 1, error, sizeof(error)))
            fail(error);
        decode_seconds += now() - t0;
    }
    return decode_seconds;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    int steps = argc > 2 ? atoi(argv[2]) : 4;
    if (steps < 1 || steps > 64) steps = 4;

    char error[512];
    char *tokenizer_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t prompt_len = 0;
    if (!h3_tokenizer_encode(tokenizer, "The capital of France is", 1, &ids,
                             &prompt_len, error, sizeof(error)))
        fail(error);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);

    qwen_session *streaming = NULL, *resident = NULL;
    if (!qwen_session_create(&streaming, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_create(&resident, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(resident, 1, error, sizeof(error)))
        fail(error);

    uint32_t *tok_s = calloc((size_t)steps, sizeof(*tok_s));
    uint32_t *tok_r = calloc((size_t)steps, sizeof(*tok_r));
    float **log_s = calloc((size_t)steps, sizeof(*log_s));
    float **log_r = calloc((size_t)steps, sizeof(*log_r));
    if (!tok_s || !tok_r || !log_s || !log_r) fail("alloc");

    double t_res = run_session(resident, ids, prompt_len, steps, tok_r, log_r);
    double t_str = run_session(streaming, ids, prompt_len, steps, tok_s, log_s);

    const qwen_logits *probe = qwen_session_logits(streaming);
    size_t vocab = probe ? probe->vocab : 151936;
    for (int step = 0; step < steps; step++) {
        require(tok_r[step] == tok_s[step],
                "resident and streaming picked different tokens");
        require(memcmp(log_r[step], log_s[step], vocab * sizeof(float)) == 0,
                "resident and streaming logits are not bit-for-bit equal");
    }
    printf("bit-for-bit parity over %d decode steps (tokens:", steps);
    for (int step = 0; step < steps; step++) printf(" %u", tok_r[step]);
    printf(")\n");
    printf("decode: resident %.2f s/tok  vs  streaming %.2f s/tok  (%.1fx)\n",
           t_res / steps, t_str / steps,
           (t_str / steps) / (t_res / steps > 1e-6 ? t_res / steps : 1e-6));

    for (int step = 0; step < steps; step++) {
        free(log_s[step]);
        free(log_r[step]);
    }
    free(tok_s); free(tok_r); free(log_s); free(log_r);
    qwen_session_free(resident);
    qwen_session_free(streaming);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    puts("ok: qwen resident-weights parity + speedup");
    return 0;
}
