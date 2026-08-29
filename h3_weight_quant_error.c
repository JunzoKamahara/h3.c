/* exp/grouped-int8-weights: weight-level BF16 reconstruction error analysis.
 *
 * Measures, for the real attn.out_proj.weight of several DiT blocks, how far
 * each INT8 quantization scheme's dequantized reconstruction sits from the
 * original BF16 values - independent of any denoising/generation run. This
 * isolates "is the quantization itself more accurate?" from "does that
 * translate into an output closer to BF16?" (the latter needs the full
 * generation pipeline and is confounded by chaotic amplification over very
 * few denoising steps - see the experiment writeup's discussion).
 *
 * Usage: h3_weight_quant_error <FL2VA/transformer dir> [block indices...]
 * (defaults to blocks 0 9 24 39 49, matching the plan's recommended
 * measurement points)
 */
#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,   /* 7168 - out_proj's K dimension */
};

typedef struct {
    const char *label;
    uint32_t group_size;   /* 0 = row-wise (one scale for the whole row) */
} quant_mode;

static const quant_mode MODES[] = {
    {"Row-wise", 0},
    {"G=1024",   1024},
    {"G=512",    512},
    {"G=256",    256},
    {"G=128",    128},
};
#define MODE_COUNT (sizeof(MODES) / sizeof(MODES[0]))

typedef struct {
    double rmse;
    double mae;
    double max_abs;
    double cosine;
    double relative_frobenius;
} error_stats;

/* IEEE754 BF16 -> double: same widening h3_shaders.metal's h3_bf16_to_f32
 * does (BF16 is just the top 16 bits of an F32), only carried to double. */
static double bf16_to_double(uint16_t bits) {
    uint32_t f32_bits = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &f32_bits, sizeof(value));
    return (double)value;
}

static void fail(const char *message) {
    fprintf(stderr, "h3: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

/* Dequantizes `int8_values` (row-major [rows, columns]) against `scales`
 * ([rows] for row-wise, [rows, groups_per_row] for grouped - group_size==0
 * selects row-wise) and accumulates error statistics against the true BF16
 * values (already widened to double in `reference`). */
static error_stats compare_reconstruction(const int8_t *int8_values,
                                          const float *scales,
                                          const double *reference,
                                          uint32_t rows, uint32_t columns,
                                          uint32_t group_size) {
    uint32_t groups_per_row = group_size ?
        (columns + group_size - 1) / group_size : 1;
    double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
    double dot = 0.0, norm_recon = 0.0, norm_ref = 0.0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t col = 0; col < columns; col++) {
            size_t index = (size_t)row * columns + col;
            uint32_t group = group_size ? col / group_size : 0;
            float scale = scales[(size_t)row * groups_per_row + group];
            double reconstructed = (double)int8_values[index] * (double)scale;
            double truth = reference[index];
            double error = reconstructed - truth;
            sum_sq += error * error;
            sum_abs += fabs(error);
            if (fabs(error) > max_abs) max_abs = fabs(error);
            dot += reconstructed * truth;
            norm_recon += reconstructed * reconstructed;
            norm_ref += truth * truth;
        }
    }
    size_t total = (size_t)rows * columns;
    error_stats stats;
    stats.rmse = sqrt(sum_sq / (double)total);
    stats.mae = sum_abs / (double)total;
    stats.max_abs = max_abs;
    stats.cosine = (norm_recon > 0.0 && norm_ref > 0.0) ?
        dot / (sqrt(norm_recon) * sqrt(norm_ref)) : 0.0;
    stats.relative_frobenius = norm_ref > 0.0 ?
        sqrt(sum_sq) / sqrt(norm_ref) : 0.0;
    return stats;
}

static void analyze_block(h3_weight_store *store, h3_gpu *gpu,
                          uint32_t block_index) {
    char name[160], error[512];
    snprintf(name, sizeof(name), "blocks.%u.attn.out_proj.weight",
             block_index);
    uint64_t shape[] = {HIDDEN, INNER};
    h3_gpu_tensor *bf16_weight = h3_weight_load_bf16(
        store, gpu, name, 2, shape, error, sizeof(error));
    if (!bf16_weight) fail(error);

    size_t elements = (size_t)HIDDEN * INNER;
    uint16_t *raw_bf16 = malloc(elements * sizeof(*raw_bf16));
    double *reference = malloc(elements * sizeof(*reference));
    require(raw_bf16 && reference, "out of memory reading reference weight");
    require(h3_gpu_tensor_read_bf16(bf16_weight, raw_bf16, elements),
           "cannot read back BF16 weight");
    for (size_t i = 0; i < elements; i++)
        reference[i] = bf16_to_double(raw_bf16[i]);

    printf("=== block %u: blocks.%u.attn.out_proj.weight [%u x %u] ===\n",
           block_index, block_index, HIDDEN, INNER);
    printf("%-10s %12s %12s %12s %12s %14s\n",
           "Mode", "RMSE", "MAE", "MaxAbsErr", "Cosine", "RelFrobErr");

    for (size_t m = 0; m < MODE_COUNT; m++) {
        uint32_t group_size = MODES[m].group_size;
        uint32_t groups_per_row = group_size ?
            (INNER + group_size - 1) / group_size : 1;
        h3_gpu_tensor *int8_out = h3_gpu_tensor_new_i8(gpu, elements);
        h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(
            gpu, (size_t)HIDDEN * groups_per_row);
        require(int8_out && scales, "out of memory allocating quant buffers");

        require(h3_gpu_begin(gpu), "cannot begin command stream");
        int ok = group_size ?
            h3_gpu_quantize_weight_int8_grouped(
                gpu, int8_out, scales, bf16_weight, HIDDEN, INNER,
                group_size) :
            h3_gpu_quantize_weight_int8(
                gpu, int8_out, scales, bf16_weight, HIDDEN, INNER);
        if (!ok) fail(h3_gpu_error(gpu));
        require(h3_gpu_submit(gpu), "cannot submit command stream");

        int8_t *host_int8 = malloc(elements * sizeof(*host_int8));
        float *host_scales = malloc(
            (size_t)HIDDEN * groups_per_row * sizeof(*host_scales));
        require(host_int8 && host_scales, "out of memory reading back quant");
        require(h3_gpu_tensor_read_i8(int8_out, host_int8, elements),
               "cannot read back int8 weight");
        require(h3_gpu_tensor_read_f32(
                    scales, host_scales, (size_t)HIDDEN * groups_per_row),
               "cannot read back scales");

        error_stats stats = compare_reconstruction(
            host_int8, host_scales, reference, HIDDEN, INNER, group_size);
        printf("%-10s %12.6f %12.6f %12.6f %12.8f %14.8f\n",
               MODES[m].label, stats.rmse, stats.mae, stats.max_abs,
               stats.cosine, stats.relative_frobenius);

        free(host_int8);
        free(host_scales);
        h3_gpu_tensor_free(int8_out);
        h3_gpu_tensor_free(scales);
    }
    printf("\n");

    free(raw_bf16);
    free(reference);
    h3_gpu_tensor_free(bf16_weight);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> [block indices...]\n",
                argv[0]);
        return 1;
    }
    char error[512];
    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                   sizeof(error));
    if (!store) fail(error);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);

    if (argc > 2) {
        for (int i = 2; i < argc; i++)
            analyze_block(store, gpu, (uint32_t)strtoul(argv[i], NULL, 10));
    } else {
        uint32_t defaults[] = {0, 9, 24, 39, 49};
        for (size_t i = 0; i < sizeof(defaults) / sizeof(defaults[0]); i++)
            analyze_block(store, gpu, defaults[i]);
    }

    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
