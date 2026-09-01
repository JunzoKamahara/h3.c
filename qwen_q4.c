#include "qwen_q4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int qwen_q4_enabled(void) {
    const char *value = getenv("H3_QWEN_Q4");
    return value && value[0] && !(value[0] == '0' && value[1] == '\0');
}

void qwen_q4_weight_free(qwen_q4_weight *weight) {
    if (!weight) return;
    h3_gpu_tensor_free(weight->packed);
    h3_gpu_tensor_free(weight->scales);
    memset(weight, 0, sizeof(*weight));
}

int qwen_q4_quantize(h3_gpu *gpu, const h3_gpu_tensor *src_bf16, uint32_t rows,
                     uint32_t cols, qwen_q4_weight *out, char *error,
                     size_t error_size) {
    memset(out, 0, sizeof(*out));
    if (cols == 0 || cols % QWEN_Q4_GROUP != 0 || (cols & 1u) != 0) {
        if (error) snprintf(error, error_size,
                            "qwen_q4_quantize: cols=%u not a multiple of %u",
                            cols, QWEN_Q4_GROUP);
        return 0;
    }
    size_t count = (size_t)rows * cols;
    size_t groups_per_row = cols / QWEN_Q4_GROUP;
    size_t packed_bytes = count / 2;
    size_t scale_count = (size_t)rows * groups_per_row;

    uint16_t *src = malloc(count * sizeof(*src));
    uint8_t *packed = malloc(packed_bytes);
    uint16_t *scales = malloc(scale_count * sizeof(*scales));
    if (!src || !packed || !scales) {
        if (error) snprintf(error, error_size,
                            "qwen_q4_quantize: out of memory (%.2f GiB)",
                            (double)(count * 2 + packed_bytes +
                                     scale_count * 2) / (1024.0 * 1024 * 1024));
        goto fail;
    }
    if (!h3_gpu_tensor_read_bf16(src_bf16, src, count)) {
        if (error) snprintf(error, error_size,
                            "qwen_q4_quantize: cannot read source weight");
        goto fail;
    }

    for (uint32_t r = 0; r < rows; r++) {
        for (size_t g = 0; g < groups_per_row; g++) {
            const uint16_t *block = src + (size_t)r * cols + g * QWEN_Q4_GROUP;
            float amax = 0.0f;
            for (uint32_t i = 0; i < QWEN_Q4_GROUP; i++) {
                float w = fabsf(bf16_to_f32(block[i]));
                if (w > amax) amax = w;
            }
            float scale = amax > 0.0f ? amax / 8.0f : 1.0f;
            scales[(size_t)r * groups_per_row + g] = f32_to_bf16(scale);
            float inv = 1.0f / scale;
            for (uint32_t i = 0; i < QWEN_Q4_GROUP; i += 2) {
                size_t col = g * QWEN_Q4_GROUP + i;
                long q0 = lroundf(bf16_to_f32(block[i]) * inv);
                long q1 = lroundf(bf16_to_f32(block[i + 1]) * inv);
                if (q0 < -8) q0 = -8; else if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8; else if (q1 > 7) q1 = 7;
                uint8_t lo = (uint8_t)(q0 + 8) & 0x0Fu;
                uint8_t hi = (uint8_t)(q1 + 8) & 0x0Fu;
                packed[((size_t)r * cols + col) / 2] =
                    (uint8_t)(lo | (hi << 4));
            }
        }
    }

    out->packed = h3_gpu_tensor_from_i8(gpu, packed, packed_bytes);
    out->scales = h3_gpu_tensor_from_bf16(gpu, scales, scale_count);
    if (!out->packed || !out->scales) {
        if (error) snprintf(error, error_size,
                            "qwen_q4_quantize: cannot allocate GPU buffers: %s",
                            h3_gpu_error(gpu));
        qwen_q4_weight_free(out);
        goto fail;
    }
    out->rows = rows;
    out->cols = cols;
    free(src);
    free(packed);
    free(scales);
    return 1;

fail:
    free(src);
    free(packed);
    free(scales);
    return 0;
}
