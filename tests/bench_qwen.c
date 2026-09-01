/* Throughput probe for the Phase 2 KV chat session.
 *
 *   ./h3_qwen_bench MiniMax-H3 [decode_steps] [prompt]
 *
 * Reports prefill and incremental-decode tokens/second. Decoder-layer weights
 * are streamed per eval (no residency yet), so this is dominated by weight I/O,
 * not GPU compute -- it is the number to watch as residency lands. */
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    int decode_steps = argc > 2 ? atoi(argv[2]) : 8;
    const char *prompt = argc > 3 ? argv[3] :
        "Write one sentence about the sea.";
    char err[512], path[1024];
    snprintf(path, sizeof(path), "%s/FL2VA/tokenizer/tokenizer.json", root);
    h3_tokenizer *tok = h3_tokenizer_load(path, err, sizeof(err));
    if (!tok) { fprintf(stderr, "%s\n", err); return 1; }
    snprintf(path, sizeof(path), "%s/FL2VA/text_encoder", root);

    uint32_t *ids = NULL; size_t n = 0;
    if (!h3_tokenizer_encode(tok, prompt, 1, &ids, &n, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); return 1; }

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, path, "h3_shaders.metal", err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); return 1; }
    qwen_session *s = NULL;
    if (!qwen_session_create(&s, engine, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); return 1; }

    printf("prompt tokens: %zu\n", n);

    double t0 = now();
    if (!qwen_session_eval(s, ids, n, err, sizeof(err))) {
        fprintf(stderr, "prefill: %s\n", err); return 1; }
    double t_prefill = now() - t0;
    printf("prefill: %.2f s  (%.3f tok/s over %zu tokens, %.2f s/tok)\n",
           t_prefill, (double)n / t_prefill, n, t_prefill / (double)n);

    double dec_total = 0.0, dec_min = 1e9, dec_max = 0.0;
    for (int i = 0; i < decode_steps; i++) {
        uint32_t next = 0;
        qwen_session_sample(s, &next, err, sizeof(err));
        double a = now();
        if (!qwen_session_eval(s, &next, 1, err, sizeof(err))) {
            fprintf(stderr, "decode: %s\n", err); return 1; }
        double dt = now() - a;
        dec_total += dt;
        if (dt < dec_min) dec_min = dt;
        if (dt > dec_max) dec_max = dt;
        printf("  decode step %d: %.2f s  (ctx=%zu)\n", i + 1, dt,
               qwen_session_length(s));
    }
    printf("decode: %.3f tok/s  (avg %.2f s/tok, min %.2f, max %.2f, "
           "%d steps)\n",
           (double)decode_steps / dec_total, dec_total / decode_steps,
           dec_min, dec_max, decode_steps);

    qwen_session_free(s);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tok);
    return 0;
}
