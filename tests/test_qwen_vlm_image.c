/* P7-004 / P7-005: end-to-end VLM chat from a real image, through the
 * KV-cache multimodal prefill + fast decode path.
 *
 * ffmpeg-synthesised image -> h3_vision_encode_bf16
 *   -> h3_multimodal_build_chat_input (ids + mRoPE positions + tags + span)
 *   -> qwen_session_eval_multimodal   (splice vision, deepstack, mRoPE prefill)
 *   -> qwen_session_sample / qwen_session_eval loop
 *
 * Also checks the prefill's first-token logits match a one-shot
 * qwen_engine_forward_full over the same multimodal input.
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
#define GEN_CAP 40

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

    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tok_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");

    h3_vision_output vout;
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, 1, IMG, IMG,
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

    uint32_t *ids = NULL, *positions = NULL;
    uint8_t *tags = NULL;
    h3_text_vision_span *spans = NULL;
    size_t n = 0;
    if (!h3_multimodal_build_chat_input(
            tok, "<|im_start|>user\n",
            "\nWhat is in this image? Answer in one short sentence.<|im_end|>\n"
            "<|im_start|>assistant\n",
            &vout, 1, &ids, &n, &positions, &tags, &spans, error,
            sizeof(error)))
        fail(error);
    printf("prompt sequence: %zu tokens (vision span at %zu, %zu pads)\n", n,
           spans[0].start, spans[0].tokens);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);

    qwen_vision_span vspan = {0};
    vspan.start = spans[0].start;
    vspan.tokens = spans[0].tokens;
    vspan.embeddings = spans[0].embeddings;
    vspan.deepstack[0] = spans[0].deepstack[0];
    vspan.deepstack[1] = spans[0].deepstack[1];
    vspan.deepstack[2] = spans[0].deepstack[2];

    qwen_input mm = {0};
    mm.token_ids = ids;
    mm.token_count = n;
    mm.vision_spans = &vspan;
    mm.vision_span_count = 1;
    mm.position_ids = positions;
    mm.tags = tags;

    /* Reference: one-shot 64-layer forward over the multimodal input. */
    qwen_logits ref = {0};
    if (!qwen_engine_forward_full(engine, &mm, &ref, NULL, NULL, error,
                                  sizeof(error)))
        fail(error);
    uint32_t ref_first = ref.argmax_token;
    qwen_logits_free(&ref);

    /* KV multimodal prefill + fast decode. */
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
    char *answer = h3_tokenizer_decode(tok, gen, (size_t)g, error,
                                       sizeof(error));
    if (!answer) fail(error);

    printf("first token: KV %u vs forward_full %u\n", gen[0], ref_first);
    printf("assistant: \"%s\"\n", answer);
    require(g > 0 && strlen(answer) > 0, "empty VLM answer");
    require(gen[0] == ref_first,
            "KV multimodal first token != forward_full first token");
    int has_alpha = 0;
    for (const char *p = answer; *p; p++)
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) has_alpha = 1;
    require(has_alpha, "VLM answer has no letters");

    free(answer);
    qwen_session_free(session);
    h3_tokenizer_ids_free(ids);
    free(positions);
    free(tags);
    free(spans);
    h3_vision_output_free(&vout);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(weights);
    free(tok_path);
    puts("ok: P7-004/005 VLM chat from a real image (KV multimodal prefill)");
    return 0;
}
