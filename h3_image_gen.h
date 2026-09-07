#ifndef H3_IMAGE_GEN_H
#define H3_IMAGE_GEN_H

#include "h3_dit.h"

#include <stddef.h>
#include <stdint.h>

/* P8-IMG-01: synchronous text-to-image.
 *
 * The caller supplies the canonical BF16 layer-49 conditioning
 * ([conditioning_tokens, 5120]) already computed from the resident Qwen
 * session -- this module never loads the language weights. It loads the FL2VA
 * transformer (SSD streaming) and the video VAE, denoises the shortest
 * releasable clip at a fixed serving-step count with the given seed, decodes
 * it, and returns the first frame.
 *
 * `fl2va_directory` is the release ".../FL2VA" directory; the transformer and
 * "video_vae/source" subtrees are found beneath it. `width` and `height` must
 * be positive multiples of 32. `steps` is the serving evaluation count.
 *
 * On success *rgb owns `*out_width * *out_height * 3` bytes of frame-major,
 * row-major interleaved RGB in [0,255]; release it with free(). */
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

int h3_image_generate(const h3_image_request *request,
                      uint8_t **rgb, int *out_width, int *out_height,
                      h3_dit_progress progress, void *progress_opaque,
                      char *error, size_t error_size);

#endif
