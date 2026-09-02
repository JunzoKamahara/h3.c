/* QEXP-002 (non-blocking): `--quality`-mode shared layer-0..49 prefix for a
 * combined Chat + H3 request.
 *
 * In all-BF16 mode the Chat prefill and the H3 conditioning both need the same
 * layers-0..49 forward. This runs it ONCE and splits:
 *   layer-49 state -> continue_from_intermediate (layers 50..63) -> Chat logits
 *                  -> h3_text_embedding -> DiT + VAE -> media
 * and checks the Chat logits are bit-for-bit `qwen_engine_forward_full`, the
 * shared layer-49 state is bit-for-bit `qwen_session_get_h3_conditioning`, and
 * reports the wall-clock saved (one 0..49 forward instead of two).
 *
 *   make qexp-002          # metrics
 *   make qexp-002-save     # + writes qexp2_shared.mp4
 */

#include "h3_audio_vae.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_vae.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HID 5120u
#define FRAMES 25
#define STEPS 8
#define RES 256
#define SEED 42ULL

static const char *PROMPT = "A red fox walking through snow";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qexp002_shared_prefix.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

static void progress(const char *phase, int c, int t, void *o) {
    (void)o;
    if (t > 0 && (c == t || c % 8 == 0))
        fprintf(stderr, "  %s %d/%d\n", phase, c, t);
}

static uint8_t *rgb_f32_to_u8(const float *rgb, size_t count) {
    uint8_t *out = malloc(count ? count : 1);
    if (!out) fail("rgb u8 alloc");
    for (size_t i = 0; i < count; i++) {
        float s = rgb[i] * 255.0f;
        s = s < 0.0f ? 0.0f : (s > 255.0f ? 255.0f : s);
        out[i] = (uint8_t)lrintf(s);
    }
    return out;
}

/* Run the T2VA DiT + VAE from an already-computed layer-49 conditioning. */
static void generate(const char *root, const uint16_t *cond, size_t n,
                     h3_video_frames *frames, h3_audio_waveform *wave) {
    char error[512];
    char *dit_path = path_join(root, "FL2VA/transformer");
    char *vvae_path = path_join(root, "FL2VA/video_vae/source");
    char *avae_path = path_join(root, "FL2VA/audio_vae");

    h3_temporal_shape temporal = h3_temporal(FRAMES);
    int lw = 0, lh = 0;
    h3_latent_canvas(RES, RES, &lw, &lh);

    h3_text_embedding text = {0};
    text.tokens = n;
    text.width = HID;
    text.values = (uint16_t *)cond;

    h3_layout_spec spec = {(int)n, temporal.video_t, lh, lw, temporal.audio_t,
                           temporal.frame_count, NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) fail(error);
    h3_sigma_schedule sigmas;
    require(h3_serving_schedule_build(STEPS, &sigmas), "serving schedule");

    float rope = (RES == 256) ? 0.5f : 1.0f;
    h3_dit *dit = h3_dit_load_t2va(dit_path, "h3_shaders.metal", &text, &layout,
                                   &sigmas, 50, 1, 0, 1, rope,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, progress,
                                   NULL, error, sizeof(error));
    if (!dit) fail(error);
    size_t nv = h3_dit_video_elements(dit), na = h3_dit_audio_elements(dit);
    float *video = malloc(nv * sizeof(float));
    float *audio = malloc(na * sizeof(float));
    require(video && audio, "latent alloc");
    h3_rng vr, ar;
    h3_rng_seed(&vr, SEED);
    h3_rng_seed(&ar, SEED);
    h3_rng_fill_normal(&vr, video, nv);
    h3_rng_fill_normal(&ar, audio, na);
    if (!h3_dit_denoise_euler(dit, video, audio, 1, progress, NULL, error,
                              sizeof(error)))
        fail(error);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    if (!h3_audio_vae_decode(avae_path, "h3_shaders.metal", audio,
                             temporal.audio_t, NULL, NULL, wave, error,
                             sizeof(error)))
        fail(error);
    if (!h3_video_vae_decode(vvae_path, "h3_shaders.metal", video,
                             temporal.video_t, lh, lw, NULL, NULL, frames,
                             error, sizeof(error)))
        fail(error);
    free(video);
    free(audio);
    free(dit_path);
    free(vvae_path);
    free(avae_path);
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    int save = argc >= 2 && !strcmp(argv[1], "--save");

    char error[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *wp = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tok = h3_tokenizer_load(tp, error, sizeof(error));
    if (!tok) fail(error);
    uint32_t *ids = NULL;
    size_t n = 0;
    if (!h3_tokenizer_encode(tok, PROMPT, 1, &ids, &n, error, sizeof(error)))
        fail(error);

    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, wp, "h3_shaders.metal", error, sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);

    qwen_input in = {0};
    in.token_ids = ids;
    in.token_count = n;

    printf("QEXP-002: prompt \"%s\" -> %zu tokens (all-BF16 / --quality)\n",
           PROMPT, n);

    /* --- naive: two separate layers-0..49 forwards --- */
    qwen_logits naive_logits = {0};
    double t0 = now();
    if (!qwen_engine_forward_full(engine, &in, &naive_logits, NULL, NULL, error,
                                  sizeof(error)))
        fail(error);
    double t_full = now() - t0;

    qwen_intermediate_state naive_state = {0};
    t0 = now();
    if (!qwen_session_get_h3_conditioning(session, &in, &naive_state, NULL, NULL,
                                          error, sizeof(error)))
        fail(error);
    double t_cond_naive = now() - t0;

    /* --- shared prefix: ONE layers-0..49 forward, split two ways --- */
    qwen_intermediate_state shared = {0};
    t0 = now();
    if (!qwen_session_get_h3_conditioning(session, &in, &shared, NULL, NULL,
                                          error, sizeof(error)))
        fail(error);
    double t_prefix = now() - t0;

    require(shared.tokens == naive_state.tokens && shared.hidden_size == HID,
            "shared state shape");
    require(memcmp(shared.values, naive_state.values,
                   n * HID * sizeof(uint16_t)) == 0,
            "shared layer-49 state is not bit-for-bit get_h3_conditioning");

    qwen_logits shared_logits = {0};
    t0 = now();
    if (!qwen_session_continue_from_intermediate(session, &shared, NULL,
                                                 &shared_logits, error,
                                                 sizeof(error)))
        fail(error);
    double t_tail = now() - t0;

    require(shared_logits.vocab == naive_logits.vocab, "vocab");
    require(shared_logits.argmax_token == naive_logits.argmax_token,
            "shared Chat argmax != forward_full argmax");
    require(memcmp(shared_logits.values, naive_logits.values,
                   naive_logits.vocab * sizeof(float)) == 0,
            "shared Chat logits are not bit-for-bit forward_full");

    printf("Chat tail from shared prefix == forward_full: bit-for-bit "
           "(next token %u)\n", shared_logits.argmax_token);

    /* H3 branch from the same shared state. */
    uint16_t *cond = malloc(n * HID * sizeof(*cond));
    require(cond != NULL, "cond dup");
    memcpy(cond, shared.values, n * HID * sizeof(*cond));

    h3_video_frames frames = {0};
    h3_audio_waveform wave = {0};
    printf("### H3 generation from the shared layer-49 state\n");
    double t_h3_0 = now();
    generate(root, cond, n, &frames, &wave);
    double t_h3 = now() - t_h3_0;

    double naive_qwen = t_full + t_cond_naive; /* 0..49 twice + a 50..63 */
    double shared_qwen = t_prefix + t_tail;    /* 0..49 once + a 50..63  */
    printf("\ncombined Chat + H3 request, Qwen backbone wall time:\n");
    printf("  naive  (0..49 twice) : %.2f s  (forward_full %.2f + "
           "get_h3_conditioning %.2f)\n", naive_qwen, t_full, t_cond_naive);
    printf("  shared (0..49 once)  : %.2f s  (prefix %.2f + tail 50..63 "
           "%.2f)\n", shared_qwen, t_prefix, t_tail);
    printf("  saved                : %.2f s  (%.0f%%) -- one prompt-length "
           "layers-0..49 forward\n",
           naive_qwen - shared_qwen,
           100.0 * (naive_qwen - shared_qwen) / naive_qwen);
    printf("  (+ H3 DiT/VAE %.1f s, unchanged either way)\n", t_h3);

    if (save) {
        uint8_t *rgb = rgb_f32_to_u8(
            frames.rgb, (size_t)frames.frames * frames.height * frames.width * 3);
        if (!h3_ffmpeg_write_av_rgb24_f32(
                "qexp2_shared.mp4", rgb, frames.frames, frames.width,
                frames.height, H3_FPS, wave.pcm, wave.samples, wave.channels,
                wave.sample_rate, error, sizeof(error)))
            fail(error);
        fprintf(stderr, "wrote qexp2_shared.mp4\n");
        free(rgb);
    }
    puts("ok: QEXP-002 shared layers-0..49 prefix");

    h3_video_frames_free(&frames);
    h3_audio_waveform_free(&wave);
    free(cond);
    qwen_logits_free(&naive_logits);
    qwen_logits_free(&shared_logits);
    qwen_intermediate_state_free(&naive_state);
    qwen_intermediate_state_free(&shared);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tok);
    free(tp);
    free(wp);
    return 0;
}
