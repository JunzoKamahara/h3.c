#include "h3_image_gen.h"

#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Qwen backbone hidden width -- the conditioning row stride. */
#define H3_IMAGE_HIDDEN 5120u

/* The joint transformer has no single-frame mode; 5 is the shortest releasable
 * clip (h3_align_frame_count floor). We denoise it and keep frame 0. */
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

int h3_image_generate(const h3_image_request *request,
                      uint8_t **rgb, int *out_width, int *out_height,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (rgb) *rgb = NULL;
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;

    if (!request || !request->fl2va_directory || !request->shader_source_path ||
        !request->conditioning || request->conditioning_tokens == 0 || !rgb) {
        set_error(error, error_size, "invalid image request");
        return 0;
    }
    if (request->width < 32 || request->height < 32 ||
        request->width % H3_CANVAS_MULTIPLE || request->height % H3_CANVAS_MULTIPLE) {
        set_error(error, error_size,
                  "image size must be a multiple of 32, at least 32");
        return 0;
    }
    if (request->steps < 1 || request->steps > H3_MAX_STEPS) {
        set_error(error, error_size, "step count out of range");
        return 0;
    }

    int ok = 0;
    char *dit_path = join_path(request->fl2va_directory, "transformer");
    char *vvae_path = join_path(request->fl2va_directory, "video_vae/source");
    float *video = NULL, *audio = NULL;
    uint8_t *frame0 = NULL;
    h3_dit *dit = NULL;
    h3_layout layout = {0};
    int layout_built = 0;
    h3_video_frames frames = {0};
    if (!dit_path || !vvae_path) {
        set_error(error, error_size, "out of memory");
        goto done;
    }

    h3_temporal_shape temporal = h3_temporal(H3_IMAGE_FRAMES);
    int lw = 0, lh = 0;
    h3_latent_canvas(request->width, request->height, &lw, &lh);

    h3_text_embedding text = {0};
    text.tokens = request->conditioning_tokens;
    text.width = H3_IMAGE_HIDDEN;
    text.values = (uint16_t *)request->conditioning;

    h3_layout_spec spec = {(int)request->conditioning_tokens, temporal.video_t,
                           lh, lw, temporal.audio_t, temporal.frame_count,
                           NULL, 0, NULL, 0};
    if (!h3_layout_build(&spec, &layout, error, error_size)) goto done;
    layout_built = 1;

    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(request->steps, &sigmas)) {
        set_error(error, error_size, "cannot build serving schedule");
        goto done;
    }

    float spatial_rope_scale =
        (request->width == 256 && request->height == 256) ? 0.5f : 1.0f;

    dit = h3_dit_load_t2va(dit_path, request->shader_source_path, &text, &layout,
                           &sigmas, 50, 1, 0, 1 /* ssd_streaming */,
                           spatial_rope_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           progress, progress_opaque, error, error_size);
    if (!dit) goto done;

    size_t nv = h3_dit_video_elements(dit), na = h3_dit_audio_elements(dit);
    video = malloc(nv * sizeof(*video));
    audio = malloc(na * sizeof(*audio));
    if (!video || !audio) {
        set_error(error, error_size, "latent allocation failed");
        goto done;
    }
    h3_rng vr, ar;
    h3_rng_seed(&vr, request->seed);
    h3_rng_seed(&ar, request->seed);
    h3_rng_fill_normal(&vr, video, nv);
    h3_rng_fill_normal(&ar, audio, na);

    if (!h3_dit_denoise_euler(dit, video, audio, 1, progress, progress_opaque,
                              error, error_size))
        goto done;
    h3_dit_free(dit);
    dit = NULL;

    /* The audio latent is a joint-model byproduct; an image ignores it. */
    if (!h3_video_vae_decode(vvae_path, request->shader_source_path, video,
                             temporal.video_t, lh, lw, NULL, NULL, &frames,
                             error, error_size))
        goto done;
    if (frames.frames < 1 || frames.width < 1 || frames.height < 1 ||
        !frames.rgb) {
        set_error(error, error_size, "video decode produced no frame");
        goto done;
    }

    size_t pixels = (size_t)frames.width * (size_t)frames.height * 3;
    frame0 = malloc(pixels ? pixels : 1);
    if (!frame0) {
        set_error(error, error_size, "out of memory converting frame");
        goto done;
    }
    for (size_t i = 0; i < pixels; i++) {
        float s = frames.rgb[i] * 255.0f;
        s = s < 0.0f ? 0.0f : (s > 255.0f ? 255.0f : s);
        frame0[i] = (uint8_t)lrintf(s);
    }

    *rgb = frame0;
    frame0 = NULL;
    if (out_width) *out_width = frames.width;
    if (out_height) *out_height = frames.height;
    ok = 1;

done:
    if (dit) h3_dit_free(dit);
    if (layout_built) h3_layout_free(&layout);
    h3_video_frames_free(&frames);
    free(frame0);
    free(video);
    free(audio);
    free(dit_path);
    free(vvae_path);
    return ok;
}
