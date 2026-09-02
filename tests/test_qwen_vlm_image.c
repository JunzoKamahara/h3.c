/* P7-004: end-to-end VLM chat from a real image.
 *
 * ffmpeg-synthesised image -> h3_vision_encode_bf16 -> h3_multimodal_build_chat_input
 * -> qwen_input (vision span + mRoPE positions + tags) -> autoregressive
 * qwen_engine_forward_full -> decoded assistant answer.
 *
 *   ./h3_qwen_vlm_image_test MiniMax-H3
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
#define GEN_CAP 32

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_vlm_image.c: %s\n", m);
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

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *img_path = "/tmp/h3_vlm_test.png";
    char error[512];

    if (system("ffmpeg -y -v error -f lavfi -i "
               "smptebars=size=224x224:rate=1 -frames:v 1 "
               "/tmp/h3_vlm_test.png") != 0)
        fail("ffmpeg could not synthesise the test image");

    float *pixels = NULL;
    if (!h3_ffmpeg_read_image_f32(img_path, IMG, IMG, H3_IMAGE_FIT_COVER,
                                  &pixels, error, sizeof(error)))
        fail(error);

    char *w_text = path_join(root, "FL2VA/text_encoder");
    char *w_vis = path_join(root, "FL2VA/text_encoder"); /* vision weights live here too */
    char *tok_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");

    h3_vision_output vout;
    if (!h3_vision_encode_bf16(w_vis, "h3_shaders.metal", pixels, 1, IMG, IMG,
                               NULL, NULL, &vout, error, sizeof(error)))
        fail(error);
    free(pixels);
    printf("vision: grid %dx%d, %zu tokens\n", vout.grid_h, vout.grid_w,
           vout.tokens);
    require(vout.grid_h == IMG / 16 && vout.grid_w == IMG / 16 &&
                vout.tokens == (size_t)(IMG / 32) * (IMG / 32),
            "unexpected vision grid");

    h3_tokenizer *tok = h3_tokenizer_load(tok_path, error, sizeof(error));
    if (!tok) fail(error);

    const char *pre = "<|im_start|>user\n";
    const char *post =
        "\nWhat is in this image? Answer in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n";

    uint32_t *ids = NULL, *positions = NULL;
    uint8_t *tags = NULL;
    h3_text_vision_span *spans = NULL;
    size_t n = 0;
    if (!h3_multimodal_build_chat_input(tok, pre, post, &vout, 1, &ids, &n,
                                        &positions, &tags, &spans, error,
                                        sizeof(error)))
        fail(error);
    printf("prompt sequence: %zu tokens (vision span at %zu, %zu pads)\n", n,
           spans[0].start, spans[0].tokens);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, w_text, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);

    qwen_vision_span vspan = {0};
    vspan.start = spans[0].start;
    vspan.tokens = spans[0].tokens;
    vspan.embeddings = spans[0].embeddings;
    vspan.deepstack[0] = spans[0].deepstack[0];
    vspan.deepstack[1] = spans[0].deepstack[1];
    vspan.deepstack[2] = spans[0].deepstack[2];

    /* Autoregressive decode. No KV cache on the multimodal path, so re-run the
     * full 64-layer forward over the growing sequence each step (fine for a
     * short answer / smoke test). */
    size_t cap = n + GEN_CAP;
    uint32_t *seq = malloc(cap * sizeof(*seq));
    uint32_t *pos = malloc(3 * cap * sizeof(*pos));
    uint8_t *tg = malloc(cap);
    require(seq && pos && tg, "alloc");
    memcpy(seq, ids, n * sizeof(*seq));
    for (int a = 0; a < 3; a++)
        memcpy(pos + (size_t)a * cap, positions + (size_t)a * n,
               n * sizeof(*pos));
    memcpy(tg, tags, n);
    uint32_t next_pos = positions[n - 1] + 1; /* text rows: equal on all axes */

    uint32_t gen[GEN_CAP];
    int g = 0;
    for (size_t len = n; g < GEN_CAP; g++, len++) {
        qwen_input in = {0};
        in.token_ids = seq;
        in.token_count = len;
        in.vision_spans = &vspan;
        in.vision_span_count = 1;
        /* qwen_input.position_ids must be contiguous [3, len]; our `pos` is
         * [3, cap] so pack a temporary. */
        uint32_t *pp = malloc(3 * len * sizeof(*pp));
        require(pp != NULL, "pos pack");
        for (int a = 0; a < 3; a++)
            memcpy(pp + (size_t)a * len, pos + (size_t)a * cap,
                   len * sizeof(*pp));
        in.position_ids = pp;
        in.tags = tg;

        qwen_logits logits = {0};
        if (!qwen_engine_forward_full(engine, &in, &logits, NULL, NULL, error,
                                      sizeof(error)))
            fail(error);
        free(pp);
        uint32_t t = logits.argmax_token;
        qwen_logits_free(&logits);
        require(t < 151936u, "generated token out of vocab");
        if (t == QWEN_TOKEN_IM_END || t == QWEN_TOKEN_ENDOFTEXT) break;
        gen[g] = t;
        seq[len] = t;
        for (int a = 0; a < 3; a++) pos[(size_t)a * cap + len] = next_pos;
        tg[len] = 1;
        next_pos++;
    }

    char *answer = h3_tokenizer_decode(tok, gen, (size_t)g, error,
                                       sizeof(error));
    if (!answer) fail(error);
    printf("assistant: \"%s\"\n", answer);
    require(g > 0 && strlen(answer) > 0, "empty VLM answer");
    int has_alpha = 0;
    for (const char *p = answer; *p; p++)
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) has_alpha = 1;
    require(has_alpha, "VLM answer has no letters");

    free(answer);
    free(seq); free(pos); free(tg);
    h3_tokenizer_ids_free(ids);
    free(positions);
    free(tags);
    free(spans);
    h3_vision_output_free(&vout);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(w_text); free(w_vis); free(tok_path);
    puts("ok: P7-004 VLM chat from a real image");
    return 0;
}
