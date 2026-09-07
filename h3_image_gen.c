#include "h3_image_gen.h"

#include "h3_audio_vae.h"
#include "h3_ffmpeg.h"
#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Qwen backbone hidden width -- the conditioning row stride. */
#define H3_IMAGE_HIDDEN 5120u

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* The joint transformer has no single-frame mode; 5 is the shortest releasable
 * clip (h3_align_frame_count floor). An image denoises it and keeps frame 0. */
#define H3_IMAGE_FRAMES 5

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static char *join_path(const char *root, const char *suffix) {
    size_t n = strlen(root) + strlen(suffix) + 2;
    char *r = malloc(n);
    if (r) snprintf(r, n, "%s/%s", root, suffix);
    return r;
}

/* Denoised latents plus the geometry needed to decode them. */
typedef struct {
    float *video;
    float *audio;
    int video_t;
    int audio_t;
    int latent_h;
    int latent_w;
} h3_denoised;

static void denoised_free(h3_denoised *d) {
    free(d->video);
    free(d->audio);
    memset(d, 0, sizeof(*d));
}

/* Shared front half: conditioning -> FL2VA transformer -> Euler denoise. The
 * audio latent is always produced (joint model) even when only the image is
 * wanted. */
static int run_denoise(const char *fl2va_directory,
                       const char *shader_source_path,
                       const uint16_t *conditioning, size_t conditioning_tokens,
                       int width, int height, int frames, int steps,
                       uint64_t seed, h3_denoised *out, h3_video_timing *timing,
                       h3_dit_progress progress, void *progress_opaque,
                       char *error, size_t error_size) {
    memset(out, 0, sizeof(*out));
    if (!fl2va_directory || !shader_source_path || !conditioning ||
        conditioning_tokens == 0) {
        set_error(error, error_size, "invalid generation request");
        return 0;
    }
    if (width < 32 || height < 32 || width % H3_CANVAS_MULTIPLE ||
        height % H3_CANVAS_MULTIPLE) {
        set_error(error, error_size,
                  "size must be a multiple of 32, at least 32");
        return 0;
    }
    if (steps < 1 || steps > H3_MAX_STEPS) {
        set_error(error, error_size, "step count out of range");
        return 0;
    }

    int ok = 0;
    char *dit_path = join_path(fl2va_directory, "transformer");
    float *video = NULL, *audio = NULL;
    h3_dit *dit = NULL;
    h3_layout layout = {0};
    int layout_built = 0;
    if (!dit_path) {
        set_error(error, error_size, "out of memory");
        goto done;
    }

    h3_temporal_shape temporal = h3_temporal(frames);
    int lw = 0, lh = 0;
    h3_latent_canvas(width, height, &lw, &lh);

    h3_text_embedding text = {0};
    text.tokens = conditioning_tokens;
    text.width = H3_IMAGE_HIDDEN;
    text.values = (uint16_t *)conditioning;

    h3_layout_spec spec = {(int)conditioning_tokens, temporal.video_t, lh, lw,
                           temporal.audio_t, temporal.frame_count,
                           NULL, 0, NULL, 0};
    if (!h3_layout_build(&spec, &layout, error, error_size)) goto done;
    layout_built = 1;

    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas)) {
        set_error(error, error_size, "cannot build serving schedule");
        goto done;
    }

    float spatial_rope_scale =
        (width == 256 && height == 256) ? 0.5f : 1.0f;

    double load_start = now_seconds();
    dit = h3_dit_load_t2va(dit_path, shader_source_path, &text, &layout, &sigmas,
                           50, 1, 0, 1 /* ssd_streaming */, spatial_rope_scale,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, progress,
                           progress_opaque, error, error_size);
    if (timing) timing->transformer_load_s = now_seconds() - load_start;
    if (!dit) goto done;

    size_t nv = h3_dit_video_elements(dit), na = h3_dit_audio_elements(dit);
    video = malloc(nv * sizeof(*video));
    audio = malloc(na * sizeof(*audio));
    if (!video || !audio) {
        set_error(error, error_size, "latent allocation failed");
        goto done;
    }
    h3_rng vr, ar;
    h3_rng_seed(&vr, seed);
    h3_rng_seed(&ar, seed);
    h3_rng_fill_normal(&vr, video, nv);
    h3_rng_fill_normal(&ar, audio, na);

    double denoise_start = now_seconds();
    if (!h3_dit_denoise_euler(dit, video, audio, 1, progress, progress_opaque,
                              error, error_size))
        goto done;
    if (timing) timing->denoise_s = now_seconds() - denoise_start;

    out->video = video;
    out->audio = audio;
    video = audio = NULL;
    out->video_t = temporal.video_t;
    out->audio_t = temporal.audio_t;
    out->latent_h = lh;
    out->latent_w = lw;
    ok = 1;

done:
    if (dit) h3_dit_free(dit);
    if (layout_built) h3_layout_free(&layout);
    free(video);
    free(audio);
    free(dit_path);
    return ok;
}

static uint8_t *rgb_f32_to_u8(const float *rgb, size_t count) {
    uint8_t *out = malloc(count ? count : 1);
    if (!out) return NULL;
    for (size_t i = 0; i < count; i++) {
        float s = rgb[i] * 255.0f;
        s = s < 0.0f ? 0.0f : (s > 255.0f ? 255.0f : s);
        out[i] = (uint8_t)lrintf(s);
    }
    return out;
}

int h3_image_generate(const h3_image_request *request,
                      uint8_t **rgb, int *out_width, int *out_height,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (rgb) *rgb = NULL;
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!request || !rgb) {
        set_error(error, error_size, "invalid image request");
        return 0;
    }

    h3_denoised latents;
    if (!run_denoise(request->fl2va_directory, request->shader_source_path,
                     request->conditioning, request->conditioning_tokens,
                     request->width, request->height, H3_IMAGE_FRAMES,
                     request->steps, request->seed, &latents, NULL, progress,
                     progress_opaque, error, error_size))
        return 0;

    int ok = 0;
    h3_video_frames frames = {0};
    char *vvae_path = join_path(request->fl2va_directory, "video_vae/source");
    if (!vvae_path) {
        set_error(error, error_size, "out of memory");
        goto done;
    }
    if (!h3_video_vae_decode(vvae_path, request->shader_source_path,
                             latents.video, latents.video_t, latents.latent_h,
                             latents.latent_w, NULL, NULL, &frames, error,
                             error_size))
        goto done;
    if (frames.frames < 1 || frames.width < 1 || frames.height < 1 ||
        !frames.rgb) {
        set_error(error, error_size, "video decode produced no frame");
        goto done;
    }

    size_t pixels = (size_t)frames.width * (size_t)frames.height * 3;
    uint8_t *frame0 = rgb_f32_to_u8(frames.rgb, pixels);
    if (!frame0) {
        set_error(error, error_size, "out of memory converting frame");
        goto done;
    }
    *rgb = frame0;
    if (out_width) *out_width = frames.width;
    if (out_height) *out_height = frames.height;
    ok = 1;

done:
    h3_video_frames_free(&frames);
    denoised_free(&latents);
    free(vvae_path);
    return ok;
}

int h3_video_generate(const h3_video_request *request, h3_video_timing *timing,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (timing) memset(timing, 0, sizeof(*timing));
    if (!request || !request->output_path || !request->output_path[0]) {
        set_error(error, error_size, "invalid video request");
        return 0;
    }
    int frames = request->frames > 0 ? request->frames : H3_IMAGE_FRAMES;

    h3_denoised latents;
    if (!run_denoise(request->fl2va_directory, request->shader_source_path,
                     request->conditioning, request->conditioning_tokens,
                     request->width, request->height, frames, request->steps,
                     request->seed, &latents, timing, progress, progress_opaque,
                     error, error_size))
        return 0;

    int ok = 0;
    h3_video_frames video = {0};
    h3_audio_waveform wave = {0};
    uint8_t *rgb = NULL;
    char *vvae_path = join_path(request->fl2va_directory, "video_vae/source");
    char *avae_path = join_path(request->fl2va_directory, "audio_vae");
    if (!vvae_path || !avae_path) {
        set_error(error, error_size, "out of memory");
        goto done;
    }
    double audio_start = now_seconds();
    if (!h3_audio_vae_decode(avae_path, request->shader_source_path,
                             latents.audio, latents.audio_t, NULL, NULL, &wave,
                             error, error_size))
        goto done;
    if (timing) timing->audio_decode_s = now_seconds() - audio_start;
    double video_start = now_seconds();
    if (!h3_video_vae_decode(vvae_path, request->shader_source_path,
                             latents.video, latents.video_t, latents.latent_h,
                             latents.latent_w, NULL, NULL, &video, error,
                             error_size))
        goto done;
    if (timing) timing->video_decode_s = now_seconds() - video_start;
    if (video.frames < 1 || video.width < 1 || video.height < 1 || !video.rgb) {
        set_error(error, error_size, "video decode produced no frames");
        goto done;
    }

    size_t count = (size_t)video.frames * video.height * video.width * 3;
    rgb = rgb_f32_to_u8(video.rgb, count);
    if (!rgb) {
        set_error(error, error_size, "out of memory converting frames");
        goto done;
    }
    double mux_start = now_seconds();
    if (!h3_ffmpeg_write_av_rgb24_f32(request->output_path, rgb, video.frames,
                                      video.width, video.height, H3_FPS,
                                      wave.pcm, wave.samples, wave.channels,
                                      wave.sample_rate, error, error_size))
        goto done;
    if (timing) timing->mux_s = now_seconds() - mux_start;
    ok = 1;

done:
    free(rgb);
    h3_video_frames_free(&video);
    h3_audio_waveform_free(&wave);
    denoised_free(&latents);
    free(vvae_path);
    free(avae_path);
    return ok;
}
