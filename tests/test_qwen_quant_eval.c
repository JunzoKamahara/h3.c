/* Quantization quality eval (QINT-008..012 baseline).
 *
 * Teacher-forces a fixed prompt set token-by-token through a resident KV
 * session -- so every position runs the rows==1 decode path -- and compares
 * next-token logits against a reference dump. Two passes:
 *
 *   H3_QWEN_Q4=0 ./h3_qwen_quant_eval MiniMax-H3 --emit-ref  ref.f32
 *   H3_QWEN_Q4=1 ./h3_qwen_quant_eval MiniMax-H3 --compare    ref.f32
 *
 * `make quant-eval` runs both. The reference is the BF16 *decode* path
 * (GEMV + per-layer fusion), so the reported deltas isolate the quantization
 * contribution on top of the decode kernels the runtime already ships.
 *
 * Metrics per prompt (mean over scored positions): top-1 agreement, top-5
 * overlap, logit relative L2, logit cosine, and KL(p_ref || p_test) in nats
 * over the full next-token distribution.
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"
#include "qwen_q4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 151936u

/* Held-out eval set. */
static const char *const PROMPTS[] = {
    "The Eiffel Tower is located in the city of",
    "If a train travels 60 miles in 1.5 hours, its average speed is",
    "日本の首都は東京です。二番目に大きい都市は",
    "吾輩は猫である。名前はまだ",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci",
    "<tools>[{\"name\":\"get_weather\"}]</tools>\nUser: weather in Paris?\nAssistant:",
};
#define PROMPT_COUNT ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

/* Calibration set for AWQ (disjoint from PROMPTS). */
static const char *const CALIB_PROMPTS[] = {
    "The history of the Roman Empire spans over a thousand years, beginning",
    "In mathematics, a prime number is a natural number greater than one that",
    "The mitochondria is often described as the powerhouse of the cell because",
    "She walked slowly down the quiet street, thinking about everything that had",
    "量子力学は、原子や電子といった非常に小さな粒子のふるまいを記述する物理学の",
    "機械学習では、大量のデータからパターンを学習し、未知の入力に対して予測を",
    "彼は毎朝六時に起きて、コーヒーを飲みながら新聞を読むのが日課だった。",
    "import numpy as np\n\ndef softmax(x):\n    e = np.exp(x - np.max(x))\n    return e /",
    "func quicksort(arr []int) []int {\n    if len(arr) <= 1 {\n        return arr\n    }",
    "System: You are a helpful assistant.\nUser: Summarize the causes of World War I.\nAssistant:",
    "The recipe calls for two cups of flour, one teaspoon of baking soda, and a pinch of",
    "According to the report, global temperatures have risen by approximately one degree",
};
#define CALIB_COUNT ((int)(sizeof(CALIB_PROMPTS) / sizeof(CALIB_PROMPTS[0])))

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_quant_eval.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

/* Feed prompt tokens one at a time; store the last-position logits after each
 * step into `out` (row i = distribution predicting token i+1). Returns the
 * number of scored steps = token_count - 1 (position 0 has no predecessor to
 * score against, but we still store it). */
static size_t run_prompt(qwen_session *session, const uint32_t *ids,
                         size_t token_count, float *out) {
    char error[512];
    require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
    for (size_t i = 0; i < token_count; i++) {
        if (!qwen_session_eval(session, &ids[i], 1, error, sizeof(error)))
            fail(error);
        const qwen_logits *lg = qwen_session_logits(session);
        require(lg && lg->vocab == VOCAB, "unexpected logits shape");
        memcpy(out + i * (size_t)VOCAB, lg->values, VOCAB * sizeof(float));
    }
    return token_count;
}

/* Indices of the top-k logits (k <= 8), sorted descending by logit. */
static void topk(const float *v, uint32_t *idx, int k) {
    for (int j = 0; j < k; j++) idx[j] = j;
    for (int j = 0; j < k; j++)
        for (int m = j + 1; m < k; m++)
            if (v[idx[m]] > v[idx[j]]) { uint32_t t = idx[j]; idx[j] = idx[m]; idx[m] = t; }
    for (uint32_t i = (uint32_t)k; i < VOCAB; i++) {
        if (v[i] <= v[idx[k - 1]]) continue;
        int p = k - 1;
        while (p > 0 && v[i] > v[idx[p - 1]]) { idx[p] = idx[p - 1]; p--; }
        idx[p] = i;
    }
}

static double kl_nats(const float *ref, const float *test) {
    double mref = ref[0], mtest = test[0];
    for (uint32_t i = 1; i < VOCAB; i++) {
        if (ref[i] > mref) mref = ref[i];
        if (test[i] > mtest) mtest = test[i];
    }
    double zref = 0.0, ztest = 0.0;
    for (uint32_t i = 0; i < VOCAB; i++) {
        zref += exp((double)ref[i] - mref);
        ztest += exp((double)test[i] - mtest);
    }
    double logzref = log(zref), logztest = log(ztest);
    double kl = 0.0;
    for (uint32_t i = 0; i < VOCAB; i++) {
        double lpr = (double)ref[i] - mref - logzref;
        double lpt = (double)test[i] - mtest - logztest;
        double pr = exp(lpr);
        if (pr > 1e-12) kl += pr * (lpr - lpt);
    }
    return kl;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *mode = argc > 2 ? argv[2] : "";
    const char *file = argc > 3 ? argv[3] : "quant_bf16_ref.f32";
    int emit = strcmp(mode, "--emit-ref") == 0;
    int compare = strcmp(mode, "--compare") == 0;
    int calib = strcmp(mode, "--emit-calib") == 0;
    if (!emit && !compare && !calib)
        fail("usage: MiniMax-H3 --emit-ref|--compare|--emit-calib FILE");

    char error[512];
    char *tok_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *w_path = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tok = h3_tokenizer_load(tok_path, error, sizeof(error));
    if (!tok) fail(error);

    const char *const *prompts = calib ? CALIB_PROMPTS : PROMPTS;
    int n_prompts = calib ? CALIB_COUNT : PROMPT_COUNT;
    uint32_t **ids = malloc((size_t)n_prompts * sizeof(*ids));
    size_t *lens = malloc((size_t)n_prompts * sizeof(*lens));
    if (!ids || !lens) fail("alloc");
    size_t max_len = 0, total_steps = 0;
    for (int p = 0; p < n_prompts; p++) {
        if (!h3_tokenizer_encode(tok, prompts[p], 1, &ids[p], &lens[p], error,
                                 sizeof(error)))
            fail(error);
        require(lens[p] >= 3, "prompt too short");
        if (lens[p] > max_len) max_len = lens[p];
        total_steps += lens[p] - 1;
    }

    if (calib) setenv("H3_QWEN_AWQ_CALIB", file, 1);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, w_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 1, error, sizeof(error)))
        fail(error);

    if (calib) {
        float *cbuf = malloc(max_len * (size_t)VOCAB * sizeof(float));
        if (!cbuf) fail("logits alloc");
        for (int p = 0; p < n_prompts; p++)
            run_prompt(session, ids[p], lens[p], cbuf);
        free(cbuf);
        for (int p = 0; p < n_prompts; p++) h3_tokenizer_ids_free(ids[p]);
        free(ids); free(lens);
        qwen_session_free(session);      /* writes the calib file */
        qwen_engine_close(engine);
        h3_tokenizer_free(tok);
        free(tok_path); free(w_path);
        printf("calibration: %d prompts, %zu tokens -> %s\n", n_prompts,
               total_steps + (size_t)n_prompts, file);
        return 0;
    }

    float *cur = malloc(max_len * (size_t)VOCAB * sizeof(float));
    float *ref = emit ? NULL : malloc(max_len * (size_t)VOCAB * sizeof(float));
    if (!cur || (!emit && !ref)) fail("logits alloc");

    FILE *f = fopen(file, emit ? "wb" : "rb");
    if (!f) fail(emit ? "cannot create ref file" : "cannot open ref file");

    if (compare) {
        const char *awq = getenv("H3_QWEN_Q4_AWQ");
        printf("quant eval: test=%s  ref=BF16 decode path  (%d prompts, "
               "%zu scored positions)\n",
               !qwen_q4_enabled() ? "BF16"
                   : (awq && *awq ? "W4A16-AWQ (INT4 fused)"
                                  : "W4A16 RTN (INT4 fused)"),
               n_prompts, total_steps);
        printf("%-4s %6s %7s %10s %8s %9s %6s\n", "p", "top1", "top5", "rel_l2",
               "cos", "KL(nats)", "flips");
    }

    double g_top1 = 0, g_top5 = 0, g_rel = 0, g_cos = 0, g_kl = 0;
    /* argmax flips bucketed by the reference top1-top2 margin. */
    long flip_small = 0, flip_mid = 0, flip_large = 0;
    long flip_by_pos[64] = {0}, seen_by_pos[64] = {0};
    for (int p = 0; p < n_prompts; p++) {
        run_prompt(session, ids[p], lens[p], cur);
        if (emit) {
            if (fwrite(cur, sizeof(float), lens[p] * (size_t)VOCAB, f) !=
                lens[p] * (size_t)VOCAB)
                fail("ref write short");
            continue;
        }
        if (fread(ref, sizeof(float), lens[p] * (size_t)VOCAB, f) !=
            lens[p] * (size_t)VOCAB)
            fail("ref read short (regenerate: make quant-eval)");

        double top1 = 0, top5 = 0, rel = 0, cos = 0, kl = 0;
        long p_flips = 0;
        size_t scored = lens[p] - 1;
        for (size_t i = 1; i < lens[p]; i++) {
            const float *r = ref + i * (size_t)VOCAB;
            const float *t = cur + i * (size_t)VOCAB;
            uint32_t ri[5], ti[5];
            topk(r, ri, 5);
            topk(t, ti, 5);
            int match = ri[0] == ti[0];
            if (match) top1 += 1.0;
            else {
                float margin = r[ri[0]] - r[ri[1]];
                if (margin < 0.1f) flip_small++;
                else if (margin < 1.0f) flip_mid++;
                else flip_large++;
                p_flips++;
                if (i < 64) flip_by_pos[i]++;
            }
            if (i < 64) seen_by_pos[i]++;
            int overlap = 0;
            for (int a = 0; a < 5; a++)
                for (int b = 0; b < 5; b++)
                    if (ri[a] == ti[b]) overlap++;
            top5 += overlap / 5.0;
            double se = 0, sr = 0, dot = 0, nt = 0;
            for (uint32_t k = 0; k < VOCAB; k++) {
                double d = (double)t[k] - (double)r[k];
                se += d * d; sr += (double)r[k] * r[k];
                dot += (double)t[k] * r[k]; nt += (double)t[k] * t[k];
            }
            rel += sqrt(se / (sr > 1e-30 ? sr : 1e-30));
            cos += dot / (sqrt(sr) * sqrt(nt) + 1e-30);
            kl += kl_nats(r, t);
        }
        top1 /= (double)scored; top5 /= (double)scored; rel /= (double)scored;
        cos /= (double)scored; kl /= (double)scored;
        printf("%-4d %6.3f %7.3f %10.2e %8.5f %9.4f %4ld/%zu\n", p, top1, top5,
               rel, cos, kl, p_flips, scored);
        g_top1 += top1 * (double)scored; g_top5 += top5 * (double)scored;
        g_rel += rel * (double)scored;  g_cos += cos * (double)scored;
        g_kl += kl * (double)scored;
    }
    fclose(f);

    if (emit) {
        printf("wrote BF16 reference logits for %d prompts -> %s\n",
               n_prompts, file);
    } else {
        double n = (double)total_steps;
        printf("%-4s %6.3f %7.3f %10.2e %8.5f %9.4f %4ld/%zu\n", "ALL",
               g_top1 / n, g_top5 / n, g_rel / n, g_cos / n, g_kl / n,
               flip_small + flip_mid + flip_large, total_steps);
        printf("flips by ref margin: small(<0.1)=%ld  mid(0.1-1.0)=%ld  "
               "large(>=1.0)=%ld\n", flip_small, flip_mid, flip_large);
        printf("flips by position:");
        for (int i = 1; i < 64; i++)
            if (seen_by_pos[i])
                printf(" %d:%ld/%ld", i, flip_by_pos[i], seen_by_pos[i]);
        putchar('\n');
        puts("ok: quant eval");
    }

    free(cur); free(ref);
    for (int p = 0; p < n_prompts; p++) h3_tokenizer_ids_free(ids[p]);
    free(ids); free(lens);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(tok_path); free(w_path);
    return 0;
}
