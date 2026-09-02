/* QINT-010: does Mixed-W4/BF16 decode degrade VLM answers vs BF16?
 *
 *   H3_QWEN_Q4=0     ./h3_qwen_vlm_parity --emit bf16.txt
 *   H3_QWEN_Q4=mixed ./h3_qwen_vlm_parity --emit mixed.txt
 *   ./h3_qwen_vlm_parity --compare bf16.txt mixed.txt
 *
 * `make qint-010` runs all three. Each case: an ffmpeg-synthesised image +
 * question -> h3_vision_encode_bf16 -> chat multimodal prefill -> greedy
 * assistant turn. The compare reports exact-match, leading-character
 * agreement, and prints every pair for eyeballing.
 */

#include "h3_ffmpeg.h"
#include "h3_multimodal.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG 224
#define GEN_CAP 48

static const struct {
    const char *lavfi;     /* ffmpeg -f lavfi -i <this> */
    const char *question;
} CASES[] = {
    {"smptebars=size=224x224", "What is in this image? One short sentence."},
    {"smptebars=size=224x224", "この画像には何色が使われていますか。一文で。"},
    {"rgbtestsrc=size=224x224", "Describe the colours and layout briefly."},
    {"color=c=red:size=224x224", "What is the main colour of this image?"},
    {"testsrc2=size=224x224:rate=1", "Describe this image in one sentence."},
};
#define NCASES ((int)(sizeof(CASES) / sizeof(CASES[0])))

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_vlm_parity.c: %s\n", m);
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

static void make_image(int c, char *out, size_t out_size) {
    snprintf(out, out_size, "/tmp/qint010_img%d.png", c);
    FILE *probe = fopen(out, "rb");
    if (probe) { fclose(probe); return; }
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -v error -f lavfi -i %s -frames:v 1 %s",
             CASES[c].lavfi, out);
    if (system(cmd) != 0) fail("ffmpeg image synthesis failed");
}

static char *run_case(qwen_engine *engine, h3_tokenizer *tok,
                      const char *weights, int c) {
    char error[512];
    char img_path[128];
    make_image(c, img_path, sizeof(img_path));

    float *pixels = NULL;
    if (!h3_ffmpeg_read_image_f32(img_path, IMG, IMG, H3_IMAGE_FIT_COVER,
                                  &pixels, error, sizeof(error)))
        fail(error);
    h3_vision_output vout;
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, 1, IMG, IMG,
                               NULL, NULL, &vout, error, sizeof(error)))
        fail(error);
    free(pixels);

    char post[512];
    snprintf(post, sizeof(post),
             "\n%s<|im_end|>\n<|im_start|>assistant\n", CASES[c].question);
    uint32_t *ids = NULL, *positions = NULL;
    uint8_t *tags = NULL;
    h3_text_vision_span *spans = NULL;
    size_t n = 0;
    if (!h3_multimodal_build_chat_input(tok, "<|im_start|>user\n", post, &vout,
                                        1, &ids, &n, &positions, &tags, &spans,
                                        error, sizeof(error)))
        fail(error);

    qwen_vision_span vspan = {0};
    vspan.start = spans[0].start;
    vspan.tokens = spans[0].tokens;
    vspan.embeddings = spans[0].embeddings;
    for (int k = 0; k < 3; k++) vspan.deepstack[k] = spans[0].deepstack[k];
    qwen_input mm = {0};
    mm.token_ids = ids;
    mm.token_count = n;
    mm.vision_spans = &vspan;
    mm.vision_span_count = 1;
    mm.position_ids = positions;
    mm.tags = tags;

    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_eval_multimodal(session, &mm, error, sizeof(error)))
        fail(error);
    uint32_t gen[GEN_CAP];
    int g = 0;
    for (; g < GEN_CAP; g++) {
        uint32_t t = 0;
        if (!qwen_session_sample(session, &t, error, sizeof(error)))
            fail(error);
        if (t == QWEN_TOKEN_IM_END || t == QWEN_TOKEN_ENDOFTEXT) break;
        gen[g] = t;
        if (!qwen_session_eval(session, &t, 1, error, sizeof(error)))
            fail(error);
    }
    char *text = h3_tokenizer_decode(tok, gen, (size_t)g, error,
                                     sizeof(error));
    if (!text) fail(error);

    qwen_session_free(session);
    h3_tokenizer_ids_free(ids);
    free(positions);
    free(tags);
    free(spans);
    h3_vision_output_free(&vout);
    return text;
}

static void emit(const char *root, const char *out_path) {
    char error[512];
    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tok_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    h3_tokenizer *tok = h3_tokenizer_load(tok_path, error, sizeof(error));
    if (!tok) fail(error);
    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);

    FILE *f = fopen(out_path, "wb");
    require(f != NULL, "cannot create emit file");
    for (int c = 0; c < NCASES; c++) {
        char *text = run_case(engine, tok, weights, c);
        uint32_t len = (uint32_t)strlen(text);
        require(fwrite(&len, sizeof(len), 1, f) == 1 &&
                    fwrite(text, 1, len, f) == len,
                "emit write");
        fprintf(stderr, "  case %d: %s\n", c, text);
        free(text);
    }
    fclose(f);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(weights);
    free(tok_path);
    printf("emit: %d VLM turns -> %s\n", NCASES, out_path);
}

static char **read_emit(const char *path) {
    FILE *f = fopen(path, "rb");
    require(f != NULL, "cannot open emit file");
    char **rows = calloc(NCASES, sizeof(*rows));
    if (!rows) fail("alloc");
    for (int c = 0; c < NCASES; c++) {
        uint32_t len = 0;
        require(fread(&len, sizeof(len), 1, f) == 1, "emit truncated");
        rows[c] = malloc(len + 1);
        require(rows[c] && fread(rows[c], 1, len, f) == len, "emit body");
        rows[c][len] = '\0';
    }
    fclose(f);
    return rows;
}

static void compare(const char *bf16_path, const char *mixed_path) {
    char **b = read_emit(bf16_path);
    char **m = read_emit(mixed_path);
    int exact = 0;
    long total_prefix = 0, total_min = 0;
    for (int c = 0; c < NCASES; c++) {
        size_t lb = strlen(b[c]), lm = strlen(m[c]);
        size_t pre = 0, lim = lb < lm ? lb : lm;
        while (pre < lim && b[c][pre] == m[c][pre]) pre++;
        if (lb == lm && pre == lb) exact++;
        total_prefix += (long)pre;
        total_min += (long)lim;
        int run = 1, worst = 1;
        for (size_t i = 1; i < lm; i++) {
            run = m[c][i] == m[c][i - 1] ? run + 1 : 1;
            if (run > worst) worst = run;
        }
        require(lm > 0 && worst < 24,
                "mixed VLM answer is empty or a runaway repeat");
        printf("--- case %d ---\n  bf16 : %s\n  mixed: %s\n  leading match: "
               "%zu/%zu chars%s\n",
               c, b[c], m[c], pre, lim, (lb == lm && pre == lb) ? "  (exact)"
                                                               : "");
    }
    printf("\nQINT-010 VLM parity (%d cases):\n", NCASES);
    printf("  exact-match answers        : %d/%d\n", exact, NCASES);
    printf("  leading-char agreement     : %ld/%ld (%.0f%%)\n", total_prefix,
           total_min, 100.0 * (double)total_prefix / (double)(total_min ? total_min : 1));
    puts("ok: QINT-010 VLM parity");
    for (int c = 0; c < NCASES; c++) { free(b[c]); free(m[c]); }
    free(b);
    free(m);
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    if (argc >= 3 && !strcmp(argv[1], "--emit")) {
        emit(root, argv[2]);
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "--compare")) {
        compare(argv[2], argv[3]);
        return 0;
    }
    fail("usage: --emit FILE | --compare BF16_FILE MIXED_FILE");
}
