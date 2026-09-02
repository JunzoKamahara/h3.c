#ifndef H3_MULTIMODAL_H
#define H3_MULTIMODAL_H

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stddef.h>
#include <stdint.h>

/* P7-004 front-end: assemble a Qwen3-VL chat multimodal token sequence for the
 * runtime path (qwen_input). Layout: `pre_text`, then per image a
 * `<|vision_start|>` + tokens x `<|image_pad|>` + `<|vision_end|>` block, then
 * `post_text`. On success fills:
 *   *ids / *id_count  -- token sequence (free with h3_tokenizer_ids_free())
 *   *positions        -- axis-major [3, id_count] mRoPE ids (free())
 *   *tags             -- [id_count], 1 for text rows, 0 for vision (free())
 *   *spans            -- [image_count]; each `start` is the offset of the first
 *                        pad token, `tokens` the pad count, and embeddings /
 *                        deepstack pointers are borrowed from `images` (free())
 * The three arrays and *spans line up with qwen_input.{token_ids, position_ids,
 * tags, vision_spans}. */
int h3_multimodal_build_chat_input(
        const h3_tokenizer *tokenizer,
        const char *pre_text, const char *post_text,
        const h3_vision_output *images, size_t image_count,
        uint32_t **ids, size_t *id_count,
        uint32_t **positions, uint8_t **tags,
        h3_text_vision_span **spans,
        char *error, size_t error_size);

/* Construct the released `<Picture n>` presentation around already encoded
 * Qwen vision outputs, then run the language decoder. FL2VA anchors and
 * image-only Ref2VA entries intentionally share this exact presentation.
 * Keeping the builder separate lets vision and text release their large
 * temporary weights independently. */
int h3_multimodal_encode_fl2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

typedef enum {
    H3_PRESENTATION_IMAGE = 1,
    H3_PRESENTATION_VIDEO = 2,
    H3_PRESENTATION_AUDIO = 3
} h3_presentation_kind;

/* A video owns one Qwen vision output per sampled two-frame block and one
 * timestamp per block. Images own exactly one output; audio owns none. */
typedef struct {
    h3_presentation_kind kind;
    int has_audio;
    const h3_vision_output *vision;
    size_t vision_count;
    const double *timestamps;
} h3_reference_presentation;

/* Construct Ref2VA labels in exact request order, including pending audio
 * labels and per-block video timestamps, then run the Qwen decoder. */
int h3_multimodal_encode_ref2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_reference_presentation *references,
                        size_t reference_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

#endif
