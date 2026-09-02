/* QEXP-001b (non-blocking): perceptual sensitivity of an H3 generation to the
 * Mixed-W4/BF16 layer-49 conditioning drift.
 *
 * Reuses the two conditioning files written by `make qexp-001`
 * (qexp_cond_bf16.bin / qexp_cond_mixed.bin), runs a real (small-resolution)
 * text-to-video-audio generation for each -- DiT denoise + video VAE + audio
 * VAE -- with the SAME seed, and compares the decoded pixels (SSIM / PSNR /
 * mean-abs) and waveform (correlation / SNR).
 *
 *   make qexp-001         # once, to produce the conditioning files
 *   make qexp-001b
 *
 * Loads the FL2VA transformer + video VAE + audio VAE twice. Very long
 * (~20-40 min). Not in `make test`.
 */

#include "h3_audio_vae.h"
#include "h3_dit.h"
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

#define HID 5120u
#define FRAMES 25
#define STEPS 12
#define RES 256
#define SEED 42ULL

static const char *PROMPT = "A red fox walking through snow";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_l49_h3_perceptual.c: %s\n", m);
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

static void progress(const char *phase, int c, int t, void *o) {
    (void)o;
    if (t > 0 && (c == t || c % 4 == 0))
        fprintf(stderr, "  %s %d/%d\n", phase, c, t);
}

static size_t tokenize_prompt(const char *root) {
    char error[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    h3_tokenizer *tok = h3_tokenizer_load(tp, error, sizeof(error));
    if (!tok) fail(error);
    uint32_t *ids = NULL;
    size_t n = 0;
    if (!h3_tokenizer_encode(tok, PROMPT, 1, &ids, &n, error, sizeof(error)))
        fail(error);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tok);
    free(tp);
    return n;
}

static uint16_t *read_cond(const char *file, size_t n) {
    FILE *f = fopen(file, "rb");
    require(f != NULL,
            "conditioning file missing -- run `make qexp-001` first");
    uint16_t *v = malloc(n * HID * sizeof(*v));
    require(v && fread(v, sizeof(*v), n * HID, f) == n * HID,
            "conditioning size mismatch");
    fclose(f);
    return v;
}

/* One text-to-VA generation with a fixed conditioning. */
static void generate(const char *root, uint16_t *cond, size_t n,
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
    text.values = cond;

    h3_layout_spec spec = {(int)n, temporal.video_t, lh, lw, temporal.audio_t,
                           temporal.frame_count, NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) fail(error);
    h3_sigma_schedule sigmas;
    require(h3_serving_schedule_build(STEPS, &sigmas),
            "cannot build serving schedule");

    float spatial_rope_scale = (RES == 256) ? 0.5f : 1.0f;
    h3_dit *dit = h3_dit_load_t2va(
        dit_path, "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 1 /* ssd_streaming */, spatial_rope_scale,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        progress, NULL, error, sizeof(error));
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

static float luma(const float *p) {
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}

/* Mean SSIM over 8x8 luma windows, all frames. */
static double video_ssim(const h3_video_frames *a, const h3_video_frames *b) {
    const double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
    int fr = a->frames, h = a->height, w = a->width;
    double acc = 0;
    long win = 0;
    for (int f = 0; f < fr; f++) {
        for (int y = 0; y + 8 <= h; y += 8) {
            for (int x = 0; x + 8 <= w; x += 8) {
                double ma = 0, mb = 0;
                for (int j = 0; j < 8; j++)
                    for (int i = 0; i < 8; i++) {
                        size_t o = (((size_t)f * h + (y + j)) * w + (x + i)) * 3;
                        ma += luma(a->rgb + o);
                        mb += luma(b->rgb + o);
                    }
                ma /= 64.0;
                mb /= 64.0;
                double va = 0, vb = 0, cov = 0;
                for (int j = 0; j < 8; j++)
                    for (int i = 0; i < 8; i++) {
                        size_t o = (((size_t)f * h + (y + j)) * w + (x + i)) * 3;
                        double da = luma(a->rgb + o) - ma;
                        double db = luma(b->rgb + o) - mb;
                        va += da * da;
                        vb += db * db;
                        cov += da * db;
                    }
                va /= 63.0;
                vb /= 63.0;
                cov /= 63.0;
                acc += ((2 * ma * mb + c1) * (2 * cov + c2)) /
                       ((ma * ma + mb * mb + c1) * (va + vb + c2));
                win++;
            }
        }
    }
    return win ? acc / (double)win : 0.0;
}

static void video_pixel_stats(const h3_video_frames *a,
                              const h3_video_frames *b, double *psnr,
                              double *mae) {
    size_t n = (size_t)a->frames * a->height * a->width * 3;
    double se = 0, ae = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a->rgb[i] - b->rgb[i];
        se += d * d;
        ae += fabs(d);
    }
    double mse = se / (double)n;
    *psnr = mse > 1e-12 ? 10.0 * log10(1.0 / mse) : 99.0;
    *mae = ae / (double)n;
}

static void audio_stats(const h3_audio_waveform *a, const h3_audio_waveform *b,
                        double *corr, double *snr) {
    size_t n = (size_t)a->channels * a->samples;
    double sa = 0, sb = 0, dot = 0, se = 0, pw = 0;
    for (size_t i = 0; i < n; i++) {
        double av = a->pcm[i], bv = b->pcm[i];
        sa += av * av;
        sb += bv * bv;
        dot += av * bv;
        double d = bv - av;
        se += d * d;
        pw += bv * bv;
    }
    *corr = dot / (sqrt(sa) * sqrt(sb) + 1e-30);
    *snr = se > 1e-12 ? 10.0 * log10(pw / se) : 99.0;
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    int control = argc >= 2 && !strcmp(argv[1], "--control");
    require((argc >= 4 && !strcmp(argv[1], "--run")) || (control && argc >= 3),
            "usage: --run BF16_COND MIXED_COND | --control BF16_COND");
    size_t n = tokenize_prompt(root);
    uint16_t *cb = read_cond(argv[2], n);
    /* --control feeds the SAME conditioning twice, to size DiT/VAE run-to-run
     * nondeterminism against the conditioning-sensitivity signal. */
    uint16_t *cm = control ? cb : read_cond(argv[3], n);

    printf("QEXP-001b%s: %dx%d, %d frames, %d serving steps, seed %llu\n",
           control ? " [CONTROL: identical conditioning]" : "", RES, RES,
           FRAMES, STEPS, (unsigned long long)SEED);

    h3_video_frames fb = {0}, fm = {0};
    h3_audio_waveform wb = {0}, wm = {0};
    printf("### generation A (%s conditioning)\n", "BF16");
    generate(root, cb, n, &fb, &wb);
    printf("### generation B (%s conditioning)\n",
           control ? "BF16 again" : "Mixed-W4");
    generate(root, cm, n, &fm, &wm);

    require(fb.frames == fm.frames && fb.height == fm.height &&
                fb.width == fm.width,
            "video geometry differs between runs");
    require(wb.channels == wm.channels && wb.samples == wm.samples,
            "audio geometry differs between runs");

    double ssim = video_ssim(&fb, &fm), psnr, mae, acorr, asnr;
    video_pixel_stats(&fb, &fm, &psnr, &mae);
    audio_stats(&wb, &wm, &acorr, &asnr);

    printf("\nMixed-W4 vs BF16 conditioning -- decoded output:\n");
    printf("  video : SSIM=%.4f  PSNR=%.2f dB  mean|dpix|=%.4f  (%d frames, "
           "%dx%d)\n",
           ssim, psnr, mae, fb.frames, fb.width, fb.height);
    printf("  audio : corr=%.4f  SNR=%.2f dB  (%d ch, %d samples)\n", acorr,
           asnr, wb.channels, wb.samples);
    puts("ok: QEXP-001b perceptual sensitivity");

    h3_video_frames_free(&fb);
    h3_video_frames_free(&fm);
    h3_audio_waveform_free(&wb);
    h3_audio_waveform_free(&wm);
    free(cb);
    if (!control) free(cm);
    return 0;
}
