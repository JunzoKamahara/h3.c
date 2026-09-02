/* QINT-012: layer-49 hidden-state drift from the decode path.
 *
 * For a fixed prompt set, compares the layer-49 residual stream produced by
 *   (a) qwen_session_forward_to_layer(50)  -- the BF16 canonical (H3 path,
 *       h3_text_encoder.c, streamed BF16), and
 *   (b) the chat KV decode path (token-by-token, rows==1), captured via
 *       H3_QWEN_DUMP_L49.
 *
 *   H3_QWEN_Q4=0     ./h3_qwen_l49_drift MiniMax-H3   -> decode-kernel drift only
 *   H3_QWEN_Q4=mixed ./h3_qwen_l49_drift MiniMax-H3   -> + Mixed-W4/BF16 quant
 *
 * Metrics per prompt (mean over positions): relative L2, cosine, max abs
 * error; plus a global per-channel RMS ratio (mixed / bf16) summary.
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HID 5120u

static const struct { const char *kind, *text; } PROMPTS[] = {
    {"EN",   "The Eiffel Tower is located in the city of"},
    {"EN",   "Photosynthesis converts carbon dioxide and water into glucose and"},
    {"JA",   "日本の首都は東京です。二番目に大きい都市は"},
    {"JA",   "吾輩は猫である。名前はまだ無い。どこで生まれたか"},
    {"code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci"},
    {"tool", "<tools>[{\"name\":\"get_weather\"}]</tools>\nUser: weather in Paris?\nAssistant:"},
};
#define NP ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_l49_drift.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *dump = "l49_dump.bin";
    remove(dump);
    setenv("H3_QWEN_DUMP_L49", dump, 1);

    char error[512];
    char *tok_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *w_path = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tok = h3_tokenizer_load(tok_path, error, sizeof(error));
    if (!tok) fail(error);

    uint32_t *ids[NP];
    size_t lens[NP], total_rows = 0, max_len = 0;
    for (int p = 0; p < NP; p++) {
        if (!h3_tokenizer_encode(tok, PROMPTS[p].text, 1, &ids[p], &lens[p],
                                 error, sizeof(error)))
            fail(error);
        require(lens[p] >= 3, "prompt too short");
        total_rows += lens[p];
        if (lens[p] > max_len) max_len = lens[p];
    }

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, w_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 1, error, sizeof(error)))
        fail(error);

    float *bf16_all = malloc(total_rows * (size_t)HID * sizeof(float));
    uint16_t *mix_raw = malloc(total_rows * (size_t)HID * sizeof(uint16_t));
    if (!bf16_all || !mix_raw) fail("alloc");

    size_t row0[NP + 1];
    row0[0] = 0;
    for (int p = 0; p < NP; p++) {
        row0[p + 1] = row0[p] + lens[p];

        qwen_input in = {0};
        in.token_ids = ids[p];
        in.token_count = lens[p];
        qwen_intermediate_state st = {0};
        require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
        if (!qwen_session_forward_to_layer(session, &in, 50, &st, NULL, NULL,
                                           error, sizeof(error)))
            fail(error);
        require(st.tokens == lens[p] && st.hidden_size == HID,
                "unexpected layer-49 state shape");
        for (size_t i = 0; i < (size_t)lens[p] * HID; i++)
            bf16_all[row0[p] * HID + i] = bf16_to_f32(st.values[i]);
        qwen_intermediate_state_free(&st);

        require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
        for (size_t i = 0; i < lens[p]; i++)
            if (!qwen_session_eval(session, &ids[p][i], 1, error, sizeof(error)))
                fail(error);
    }

    FILE *df = fopen(dump, "rb");
    require(df != NULL, "cannot open L49 dump");
    require(fread(mix_raw, sizeof(uint16_t), total_rows * (size_t)HID, df) ==
                total_rows * (size_t)HID,
            "L49 dump size mismatch");
    fclose(df);

    const char *q4 = getenv("H3_QWEN_Q4");
    printf("L49 drift: path=%s  ref=forward_to_layer(50) BF16  (%d prompts, "
           "%zu positions)\n",
           !q4 || !*q4 || !strcmp(q4, "0") ? "BF16 decode"
               : (!strcmp(q4, "mixed") ? "Mixed-W4/BF16 decode"
                                       : "W4A16 decode"),
           NP, total_rows);
    printf("%-5s %5s %10s %9s %10s\n", "p", "kind", "rel_l2", "cos", "max_abs");

    double *ch_num = calloc(HID, sizeof(double));   /* Σ mix^2 per channel */
    double *ch_den = calloc(HID, sizeof(double));   /* Σ bf16^2 per channel */
    if (!ch_num || !ch_den) fail("alloc");
    double g_rel = 0, g_cos = 0, g_max = 0;
    size_t g_n = 0;
    for (int p = 0; p < NP; p++) {
        double rel = 0, cs = 0, mx = 0;
        for (size_t i = 0; i < lens[p]; i++) {
            const float *b = bf16_all + (row0[p] + i) * HID;
            const uint16_t *mr = mix_raw + (row0[p] + i) * HID;
            double se = 0, sb = 0, dot = 0, sm = 0, lmx = 0;
            for (uint32_t c = 0; c < HID; c++) {
                double mv = bf16_to_f32(mr[c]), bv = b[c];
                double d = mv - bv;
                se += d * d; sb += bv * bv; dot += mv * bv; sm += mv * mv;
                if (fabs(d) > lmx) lmx = fabs(d);
                ch_num[c] += mv * mv;
                ch_den[c] += bv * bv;
            }
            rel += sqrt(se / (sb > 1e-30 ? sb : 1e-30));
            cs += dot / (sqrt(sb) * sqrt(sm) + 1e-30);
            if (lmx > mx) mx = lmx;
        }
        rel /= (double)lens[p]; cs /= (double)lens[p];
        printf("%-5d %5s %10.3e %9.5f %10.3e\n", p, PROMPTS[p].kind, rel, cs,
               mx);
        g_rel += rel * (double)lens[p]; g_cos += cs * (double)lens[p];
        if (mx > g_max) g_max = mx;
        g_n += lens[p];
    }
    double rr_min = 1e30, rr_max = 0, rr_sum = 0;
    long rr_out = 0;
    for (uint32_t c = 0; c < HID; c++) {
        double rr = sqrt(ch_num[c] / (ch_den[c] > 1e-30 ? ch_den[c] : 1e-30));
        rr_sum += rr;
        if (rr < rr_min) rr_min = rr;
        if (rr > rr_max) rr_max = rr;
        if (rr < 0.9 || rr > 1.1) rr_out++;
    }
    printf("%-5s %5s %10.3e %9.5f %10.3e\n", "ALL", "", g_rel / (double)g_n,
           g_cos / (double)g_n, g_max);
    printf("per-channel RMS ratio (mix/bf16): mean=%.4f  min=%.4f  max=%.4f  "
           "outside[0.9,1.1]=%ld/%u\n",
           rr_sum / HID, rr_min, rr_max, rr_out, HID);
    puts("ok: L49 drift");

    free(ch_num); free(ch_den); free(bf16_all); free(mix_raw);
    for (int p = 0; p < NP; p++) h3_tokenizer_ids_free(ids[p]);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(tok_path); free(w_path);
    return 0;
}
