/* QINT-015d-3 -- speculative verifier benchmark.
 *
 * Draft quality is deliberately removed from the picture: this measures only
 * "verify M candidate token positions in ONE target weight sweep" against
 * "decode M tokens one at a time". Everything runs against the production
 * Mixed-W4/BF16 target (H3_QWEN_Q4=mixed). Wall-clock, warm-up first, and the
 * per-trial rewind back to the base frontier is NOT timed (a real all-accept
 * cycle does no rewind).
 *
 * Per context length and per M in 2..5, from the same frontier:
 *   scalar-1  : one qwen_session_eval(token, 1)
 *   scalar-M  : M sequential qwen_session_eval(token, 1)
 *   verify-M  : one qwen_session_verify_block([t0..t_{M-1}], M)
 * plus an end-to-end oracle-driven qwen_spec_step() for width 2..5 (draft
 * propose + coordinator + verify + top-2 scan + state), full acceptance.
 *
 *   ./h3_qwen_spec_bench [MiniMax-H3]
 */

#include "qwen_draft.h"
#include "qwen_engine.h"
#include "qwen_spec.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NREP 10
#define WARM 3
#define PROMPT "Explain in detail why the daytime sky is blue and sunsets red."

static void fail(const char *m) {
    fprintf(stderr, "FAIL bench_qwen_spec: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a - *(const double *)b;
    return x < 0 ? -1 : x > 0 ? 1 : 0;
}
typedef struct { double med, mn, p90; } msstat;
static msstat summarise(double *s, int n) {
    qsort(s, (size_t)n, sizeof(double), cmp_d);
    msstat r;
    r.mn = s[0];
    r.med = s[n / 2];
    r.p90 = s[(int)((double)n * 0.9)];
    return r;
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

static const uint32_t STOP_IDS[] = {QWEN_TOKEN_IM_END, QWEN_TOKEN_ENDOFTEXT};
#define STOP_COUNT ((size_t)(sizeof(STOP_IDS) / sizeof(STOP_IDS[0])))

/* Prefill `sess` with `prompt`, then chunk-prefill a repeated filler until the
 * context is at least `target_len` tokens. Returns the base length. */
static size_t build_context(qwen_session *sess, h3_tokenizer *tok,
                            const char *prompt, size_t target_len) {
    char err[512];
    qwen_chat_message msg = {QWEN_ROLE_USER, prompt, NULL};
    uint32_t *ids = NULL;
    size_t n = 0;
    require(qwen_chat_tokenize(tok, &msg, 1, 1, &ids, &n, err, sizeof(err)),
            err);
    require(qwen_session_eval(sess, ids, n, err, sizeof(err)), err);
    h3_tokenizer_ids_free(ids);

    static const char *FILLER =
        " The sky is blue because sunlight is scattered by air molecules, and "
        "shorter blue wavelengths scatter far more than longer red ones, so the "
        "light reaching us from every direction is dominated by blue. ";
    while (qwen_session_length(sess) < target_len) {
        uint32_t *fids = NULL;
        size_t fn = 0;
        require(h3_tokenizer_encode(tok, FILLER, 0, &fids, &fn, err,
                                    sizeof(err)),
                err);
        require(qwen_session_eval(sess, fids, fn, err, sizeof(err)), err);
        h3_tokenizer_ids_free(fids);
    }
    return qwen_session_length(sess);
}

static void bench_context(qwen_engine *eng, h3_tokenizer *tok,
                          const char *label, size_t target_len) {
    char err[512];
    qwen_session *sess = NULL;
    require(qwen_session_create(&sess, eng, err, sizeof(err)), err);
    require(qwen_session_set_resident(sess, 1, err, sizeof(err)), err);
    size_t base = build_context(sess, tok, PROMPT, target_len);

    /* 5 real greedy continuation tokens (an all-accept block). */
    uint32_t blk[5];
    for (int i = 0; i < 5; i++) {
        require(qwen_session_sample(sess, &blk[i], err, sizeof(err)), err);
        require(qwen_session_eval(sess, &blk[i], 1, err, sizeof(err)), err);
    }
    require(qwen_session_rewind(sess, base, err, sizeof(err)), err);

    printf("\n=== context \"%s\": %zu tokens ===\n", label, base);

    /* scalar-1 */
    double s1[NREP];
    for (int w = 0; w < WARM; w++) {
        require(qwen_session_eval(sess, &blk[0], 1, err, sizeof(err)), err);
        require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
    }
    for (int r = 0; r < NREP; r++) {
        double t0 = now_ms();
        require(qwen_session_eval(sess, &blk[0], 1, err, sizeof(err)), err);
        s1[r] = now_ms() - t0;
        require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
    }
    msstat S1 = summarise(s1, NREP);

    double sm[4], vm[4], em[4];
    double smmin[4], vmmin[4];
    for (int mi = 0; mi < 4; mi++) {
        size_t M = (size_t)mi + 2;

        double sM[NREP], vM[NREP], eM[NREP];
        for (int w = 0; w < WARM; w++) {
            for (size_t i = 0; i < M; i++)
                require(qwen_session_eval(sess, &blk[i], 1, err, sizeof(err)),
                        err);
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
            qwen_verify_result vr;
            require(qwen_session_verify_block(sess, blk, M, &vr, err,
                                              sizeof(err)),
                    err);
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
        }
        for (int r = 0; r < NREP; r++) {
            double t0 = now_ms();
            for (size_t i = 0; i < M; i++)
                require(qwen_session_eval(sess, &blk[i], 1, err, sizeof(err)),
                        err);
            sM[r] = now_ms() - t0;
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);

            qwen_verify_result vr;
            t0 = now_ms();
            require(qwen_session_verify_block(sess, blk, M, &vr, err,
                                              sizeof(err)),
                    err);
            vM[r] = now_ms() - t0;
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
        }

        /* end-to-end oracle coordinator at width = M (block = M rows). */
        size_t hlen = 0;
        const uint32_t *hist = qwen_session_history(sess, &hlen);
        uint32_t *full = malloc((hlen + 5) * sizeof(*full));
        require(full != NULL, "alloc");
        memcpy(full, hist, hlen * sizeof(*full));
        memcpy(full + hlen, blk, 5 * sizeof(*full));
        for (int w = 0; w < WARM; w++) {
            qwen_draft_backend *o =
                qwen_draft_oracle_new(full, hlen + 5, (size_t)-1, 0);
            qwen_spec sp;
            require(qwen_spec_init(&sp, sess, o, (unsigned)M, err, sizeof(err)),
                    err);
            sp.have_pending = 1;
            sp.pending_anchor = blk[0];
            qwen_spec_cycle cyc;
            require(qwen_spec_step(&sp, M, STOP_IDS, STOP_COUNT, &cyc, err,
                                   sizeof(err)),
                    err);
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
            qwen_draft_destroy(o);
        }
        for (int r = 0; r < NREP; r++) {
            qwen_draft_backend *o =
                qwen_draft_oracle_new(full, hlen + 5, (size_t)-1, 0);
            qwen_spec sp;
            require(qwen_spec_init(&sp, sess, o, (unsigned)M, err, sizeof(err)),
                    err);
            sp.have_pending = 1;
            sp.pending_anchor = blk[0];
            qwen_spec_cycle cyc;
            double t0 = now_ms();
            require(qwen_spec_step(&sp, M, STOP_IDS, STOP_COUNT, &cyc, err,
                                   sizeof(err)),
                    err);
            eM[r] = now_ms() - t0;
            require(cyc.committed_count == M, "oracle cycle did not fully accept");
            require(qwen_session_rewind(sess, base, err, sizeof(err)), err);
            qwen_draft_destroy(o);
        }
        free(full);

        msstat SM = summarise(sM, NREP), VM = summarise(vM, NREP),
               EM = summarise(eM, NREP);
        sm[mi] = SM.med; smmin[mi] = SM.mn;
        vm[mi] = VM.med; vmmin[mi] = VM.mn;
        em[mi] = EM.med;
        printf("  M=%zu  scalar-M med=%.1f min=%.1f  verify-M med=%.1f min=%.1f "
               " oracle-step med=%.1f  (ms)\n",
               M, SM.med, SM.mn, VM.med, VM.mn, EM.med);
    }

    /* summary table */
    printf("\n  %-4s %10s %10s %9s %11s %11s %12s\n", "rows", "scalar-M",
           "verify-M", "v/s", "upper tok/s", "ideal x", "batch compr");
    printf("  %-4d %10.1f %10s %9.2f %11.2f %11s %12s\n", 1, S1.med, "-", 1.0,
           1000.0 / S1.med, "-", "-");
    for (int mi = 0; mi < 4; mi++) {
        int M = mi + 2;
        double upper = 1000.0 * M / vm[mi];
        double ideal = (double)M * S1.med / vm[mi];
        double compr = sm[mi] / vm[mi];
        printf("  %-4d %10.1f %10.1f %9.2f %11.2f %11.2f %12.2f\n", M, sm[mi],
               vm[mi], vm[mi] / (S1.med * M), upper, ideal, compr);
    }
    printf("\n  scalar decode          : %.2f tok/s (scalar-1 med %.1f ms)\n",
           1000.0 / S1.med, S1.med);
    for (int mi = 0; mi < 4; mi++) {
        int M = mi + 2;
        printf("  verify W=%d upper bound : %.2f tok/s   end-to-end (oracle, "
               "full-accept) %.2f tok/s\n",
               M, 1000.0 * M / vm[mi], 1000.0 * M / em[mi]);
    }
    (void)vmmin;
    (void)smmin;
    qwen_session_free(sess);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char err[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *wp = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tok = h3_tokenizer_load(tp, err, sizeof(err));
    if (!tok) fail(err);
    qwen_engine *eng = NULL;
    if (!qwen_engine_open(&eng, wp, "h3_shaders.metal", err, sizeof(err)))
        fail(err);
    free(tp);
    free(wp);

    printf("QINT-015d-3 speculative verifier benchmark (Mixed-W4/BF16 target)\n");
    printf("  reps=%d, warm-up=%d, rewind untimed\n", NREP, WARM);

    bench_context(eng, tok, "short", 150);
    bench_context(eng, tok, "long", 1500);

    qwen_engine_close(eng);
    h3_tokenizer_free(tok);
    puts("\nok: spec verifier benchmark");
    return 0;
}
