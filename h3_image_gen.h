#ifndef H3_IMAGE_GEN_H
#define H3_IMAGE_GEN_H

#include "h3_dit.h"

#include <stddef.h>
#include <stdint.h>

/* P8: text-to-media generation core.
 *
 * The caller supplies the canonical BF16 layer-49 conditioning
 * ([conditioning_tokens, 5120]) already computed from a Qwen session -- this
 * module never loads the language weights. It loads the FL2VA transformer (SSD
 * streaming) and the VAEs, denoises with the given seed and serving-step
 * count, and returns pixels (image) or writes an MP4 (video).
 *
 * `fl2va_directory` is the release ".../FL2VA" directory; the transformer and
 * "video_vae/source" / "audio_vae" subtrees are found beneath it. `width` and
 * `height` must be positive multiples of 32. */

typedef struct {
    const char *fl2va_directory;
    const char *shader_source_path;
    const uint16_t *conditioning;   /* BF16 [conditioning_tokens, 5120] */
    size_t conditioning_tokens;
    int width;
    int height;
    int steps;
    uint64_t seed;
} h3_image_request;

/* P8-IMG-01: denoise the shortest releasable clip and return its first frame.
 * On success *rgb owns `*out_width * *out_height * 3` bytes of row-major
 * interleaved RGB in [0,255]; release it with free(). */
int h3_image_generate(const h3_image_request *request,
                      uint8_t **rgb, int *out_width, int *out_height,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size);

typedef struct {
    const char *fl2va_directory;
    const char *shader_source_path;
    const uint16_t *conditioning;   /* BF16 [conditioning_tokens, 5120] */
    size_t conditioning_tokens;
    int width;
    int height;
    int frames;                     /* requested; aligned up internally */
    int steps;
    uint64_t seed;
    const char *output_path;        /* MP4 written here (video + audio) */
} h3_video_request;

/* P8-VID-01: denoise a clip, decode video + audio, and mux an MP4 to
 * request->output_path. */
int h3_video_generate(const h3_video_request *request,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size);

#endif
