/* exp/grouped-int8-weights, Stage 2 (see the experiment writeup): compares
 * h3_gpu_quantize_weight_int8_grouped's GPU output against a CPU reference
 * for a small, hand-built matrix, covering the cases the plan's section
 * 13.1 calls out: a genuinely partial final group (group_size doesn't
 * divide columns), a zero row, and a row whose largest-magnitude value is
 * negative. Test values are exact in BF16 (small multiples of 0.25) so the
 * host reference does not need to reproduce Metal's F32->BF16 rounding to
 * get a bit-exact match.
 *
 * Usage: h3_int8_weight_group_test
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 4,
    COLUMNS = 300,      /* deliberately not a multiple of GROUP_SIZE */
    GROUP_SIZE = 128,
    GROUPS = (COLUMNS + GROUP_SIZE - 1) / GROUP_SIZE  /* 3: 128, 128, 44 */
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_int8_weight_group.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void require_gpu(h3_gpu *gpu, int condition, const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_int8_weight_group.c: %s: %s\n",
            operation, h3_gpu_error(gpu));
    exit(1);
}

/* Mirrors h3_quantize_bf16_weight_int8_groups_scalar exactly: max_abs/127
 * symmetric scale per group, round-to-nearest-even, clamp to [-127,127]. */
static void reference_quantize(const float *input, int8_t *output,
                               float *scales) {
    for (uint32_t row = 0; row < ROWS; row++) {
        const float *row_in = input + (size_t)row * COLUMNS;
        int8_t *row_out = output + (size_t)row * COLUMNS;
        for (uint32_t group = 0; group < GROUPS; group++) {
            uint32_t start = group * GROUP_SIZE;
            uint32_t width = GROUP_SIZE < COLUMNS - start ?
                (uint32_t)GROUP_SIZE : COLUMNS - start;
            float max_abs = 0.0f;
            for (uint32_t i = 0; i < width; i++)
                max_abs = fmaxf(max_abs, fabsf(row_in[start + i]));
            float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
            float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
            scales[row * GROUPS + group] = scale;
            for (uint32_t i = 0; i < width; i++) {
                float quantized = nearbyintf(row_in[start + i] * inverse);
                if (quantized > 127.0f) quantized = 127.0f;
                if (quantized < -127.0f) quantized = -127.0f;
                row_out[start + i] = (int8_t)quantized;
            }
        }
    }
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);

    float *host_input = malloc((size_t)ROWS * COLUMNS * sizeof(float));
    require(host_input != NULL, "host allocation failed");
    for (uint32_t col = 0; col < COLUMNS; col++) {
        /* Row 0: all zero (degenerate scale = 1/127, all-zero output). */
        host_input[0 * COLUMNS + col] = 0.0f;
        /* Row 1: ramps up so each group's positive max sits at its end. */
        host_input[1 * COLUMNS + col] = (float)(col % GROUP_SIZE) * 0.25f;
        /* Row 2: negative max (most negative value has the largest |x|). */
        host_input[2 * COLUMNS + col] =
            -(float)((col % GROUP_SIZE) + 1) * 0.25f;
        /* Row 3: generic mixed-sign values, still exact in BF16. */
        host_input[3 * COLUMNS + col] =
            ((float)((col * 7 + 3) % 41) - 20.0f) * 0.25f;
    }

    h3_gpu_tensor *input_f32 =
        h3_gpu_tensor_from_f32(gpu, host_input, (size_t)ROWS * COLUMNS);
    require(input_f32 != NULL, "input tensor allocation failed");
    h3_gpu_tensor *input_bf16 =
        h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * COLUMNS);
    h3_gpu_tensor *output_i8 =
        h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * COLUMNS);
    h3_gpu_tensor *scales_f32 =
        h3_gpu_tensor_new_f32(gpu, (size_t)ROWS * GROUPS);
    require(input_bf16 && output_i8 && scales_f32,
           "tensor allocation failed");

    require_gpu(gpu, h3_gpu_begin(gpu), "begin command stream");
    require_gpu(gpu,
        h3_gpu_cast_f32_to_bf16(gpu, input_bf16, input_f32,
                                (uint32_t)(ROWS * COLUMNS)),
        "cast input to bf16");
    require_gpu(gpu,
        h3_gpu_quantize_weight_int8_grouped(
            gpu, output_i8, scales_f32, input_bf16, ROWS, COLUMNS,
            GROUP_SIZE),
        "grouped weight quantization");
    require_gpu(gpu, h3_gpu_submit(gpu), "submit command stream");

    int8_t *gpu_output = malloc((size_t)ROWS * COLUMNS * sizeof(int8_t));
    float *gpu_scales = malloc((size_t)ROWS * GROUPS * sizeof(float));
    require(gpu_output && gpu_scales, "readback allocation failed");
    require(h3_gpu_tensor_read_i8(output_i8, gpu_output,
                                  (size_t)ROWS * COLUMNS),
           "readback quantized weight");
    require(h3_gpu_tensor_read_f32(scales_f32, gpu_scales,
                                   (size_t)ROWS * GROUPS),
           "readback scales");

    int8_t *reference_output = malloc((size_t)ROWS * COLUMNS * sizeof(int8_t));
    float *reference_scales = malloc((size_t)ROWS * GROUPS * sizeof(float));
    require(reference_output && reference_scales,
           "reference allocation failed");
    reference_quantize(host_input, reference_output, reference_scales);

    size_t scale_mismatches = 0, value_mismatches = 0;
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t group = 0; group < GROUPS; group++) {
            size_t index = (size_t)row * GROUPS + group;
            if (gpu_scales[index] != reference_scales[index]) {
                if (!scale_mismatches)
                    fprintf(stderr,
                        "  scale mismatch row=%u group=%u gpu=%.9g ref=%.9g\n",
                        row, group, (double)gpu_scales[index],
                        (double)reference_scales[index]);
                scale_mismatches++;
            }
        }
        for (uint32_t col = 0; col < COLUMNS; col++) {
            size_t index = (size_t)row * COLUMNS + col;
            if (gpu_output[index] != reference_output[index]) {
                if (!value_mismatches)
                    fprintf(stderr,
                        "  value mismatch row=%u col=%u gpu=%d ref=%d\n",
                        row, col, gpu_output[index], reference_output[index]);
                value_mismatches++;
            }
        }
    }

    /* Sanity-check the degenerate all-zero row explicitly, not just via the
     * generic comparison above. */
    for (uint32_t group = 0; group < GROUPS; group++)
        require(reference_scales[group] == 1.0f / 127.0f,
               "zero row should fall back to the 1/127 scale");
    for (uint32_t col = 0; col < COLUMNS; col++)
        require(reference_output[col] == 0,
               "zero row should quantize to all zeros");
    /* And that the partial final group (44-wide, not 128) was exercised
     * at all - a bug that only ever touched the full groups would still
     * pass an all-equal-widths comparison. */
    require(COLUMNS - (GROUPS - 1) * GROUP_SIZE == 44,
           "test setup: final group should be 44 wide");

    if (scale_mismatches || value_mismatches) {
        fprintf(stderr,
            "FAIL tests/test_int8_weight_group.c: %zu/%u scale and "
            "%zu/%u value mismatches (rows=%u columns=%u group_size=%u, "
            "groups=%u, final group width=%u)\n",
            scale_mismatches, ROWS * GROUPS, value_mismatches,
            ROWS * COLUMNS, ROWS, COLUMNS, GROUP_SIZE, GROUPS,
            COLUMNS - (GROUPS - 1) * GROUP_SIZE);
        return 1;
    }

    printf("OK tests/test_int8_weight_group.c: %u rows x %u columns, "
           "group_size=%u (%u groups, final group %u wide), byte-exact "
           "against the CPU reference\n",
           ROWS, COLUMNS, GROUP_SIZE, GROUPS,
           COLUMNS - (GROUPS - 1) * GROUP_SIZE);

    free(host_input);
    free(gpu_output);
    free(gpu_scales);
    free(reference_output);
    free(reference_scales);
    h3_gpu_tensor_free(input_f32);
    h3_gpu_tensor_free(input_bf16);
    h3_gpu_tensor_free(output_i8);
    h3_gpu_tensor_free(scales_f32);
    h3_gpu_free(gpu);
    return 0;
}
