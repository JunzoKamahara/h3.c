/* Decode-matmul microbenchmark for the Chat runtime.
 *
 * Decode is a stack of GEMVs (row count 1) and is weight-bandwidth bound. This
 * measures, per Qwen3-VL projection:
 *   - the real decode case: rows=1 bf16 via the existing h3_gpu_linear_bf16
 *     (effective GB/s says how close the kernel gets to Unified Memory peak);
 *   - a rows=128 bf16-vs-int8 GEMM (h3_gpu_linear_int8_bf16, weights quantised
 *     once). h3.c's int8/TensorOps kernels are GEMM-only (min 128 rows), so
 *     this is the closest proxy for the quantisation ceiling -- there is no
 *     int8 GEMV path today.
 *
 *   ./h3_qwen_matmul_bench [iters]     (H3_FORCE_TENSOROPS=1 to try the int8 GEMM)
 *
 * No model weights needed. GPU-kernel env knobs are honoured by h3_gpu_create.
 */

#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char *message, h3_gpu *gpu) {
    fprintf(stderr, "FAIL bench_qwen_matmul: %s%s%s\n", message,
            gpu ? ": " : "", gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static double time_loop(h3_gpu *gpu, int iters, int (*body)(void *), void *c) {
    for (int w = 0; w < 4; w++) body(c);
    double t0 = now();
    for (int it = 0; it < iters; it++)
        if (!body(c)) die("kernel", gpu);
    return (now() - t0) / iters * 1e6; /* us/call */
}

typedef struct {
    const char *name;
    uint32_t k, n;
    int per_layer; /* x64 vs x1 in the per-token projection */
} shape;

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *output, *input, *weight;       /* bf16 */
    h3_gpu_tensor *weight_i8, *weight_scales;
    h3_gpu_tensor *q_input, *input_scales;
    uint32_t k, n, rows;
} ctx;

static int run_bf16(void *v) {
    ctx *c = v;
    return h3_gpu_begin(c->gpu) &&
           h3_gpu_linear_bf16(c->gpu, c->output, c->input, c->weight, NULL,
                              c->rows, c->k, c->n) &&
           h3_gpu_submit(c->gpu);
}
static int run_int8(void *v) {
    ctx *c = v;
    return h3_gpu_begin(c->gpu) &&
           h3_gpu_linear_int8_bf16(c->gpu, c->output, c->q_input,
                                   c->input_scales, c->input, c->weight_i8,
                                   c->weight_scales, c->rows, c->k, c->n, 0) &&
           h3_gpu_submit(c->gpu);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 40;
    if (iters < 4) iters = 4;

    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error, NULL);
    int int8_available = h3_gpu_has_int8_mlp(gpu);
    printf("int8 / TensorOps GEMM path: %s%s\n\n",
           int8_available ? "enabled" : "unavailable",
           int8_available ? "" : " (H3_FORCE_TENSOROPS=1 on a Metal 4 GPU)");

    const shape shapes[] = {
        {"q_proj",    5120,   8192, 1}, {"k_proj",    5120,   1024, 1},
        {"v_proj",    5120,   1024, 1}, {"o_proj",    8192,   5120, 1},
        {"gate_proj", 5120,  25600, 1}, {"up_proj",   5120,  25600, 1},
        {"down_proj", 25600,  5120, 1}, {"lm_head",   5120, 151936, 0},
    };
    const size_t nshapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("%-10s %7s %7s | rows=1 bf16 GEMV  | rows=128 GEMM  bf16 vs int8\n",
           "shape", "K", "N");
    printf("%-10s %7s %7s | %9s %7s | %8s %8s %5s\n", "", "", "", "us", "GB/s",
           "bf16 us", "int8 us", "x");
    printf("-------------------------------------------------------------------"
           "----------\n");

    double tok_gemv_ms = 0.0;

    for (size_t s = 0; s < nshapes; s++) {
        uint32_t k = shapes[s].k, n = shapes[s].n;
        size_t welems = (size_t)n * k;
        uint16_t *wh = malloc(welems * sizeof(*wh));
        uint16_t *ih = malloc((size_t)k * 128 * sizeof(*ih));
        if (!wh || !ih) die("host alloc", NULL);
        for (size_t i = 0; i < welems; i++)
            wh[i] = f32_to_bf16((float)((i * 2654435761u) % 512) / 512.0f -
                                0.5f);
        for (size_t i = 0; i < (size_t)k * 128; i++)
            ih[i] = f32_to_bf16((float)(i % 97) / 97.0f - 0.5f);

        ctx c = {0};
        c.gpu = gpu;
        c.k = k;
        c.n = n;
        c.weight = h3_gpu_tensor_from_bf16(gpu, wh, welems);
        c.input = h3_gpu_tensor_from_bf16(gpu, ih, (size_t)k * 128);
        c.output = h3_gpu_tensor_new_bf16(gpu, (size_t)n * 128);
        c.weight_i8 = h3_gpu_tensor_new_i8(gpu, welems);
        c.weight_scales = h3_gpu_tensor_new_f32(gpu, n);
        c.q_input = h3_gpu_tensor_new_i8(gpu, (size_t)k * 128);
        c.input_scales = h3_gpu_tensor_new_f32(gpu, 128);
        if (!c.weight || !c.input || !c.output || !c.weight_i8 ||
            !c.weight_scales || !c.q_input || !c.input_scales)
            die("gpu alloc", gpu);
        free(wh);
        free(ih);

        /* rows=1 bf16 GEMV -- the real decode step */
        c.rows = 1;
        double gemv_us = time_loop(gpu, iters, run_bf16, &c);
        double gemv_gbs = (double)welems * 2.0 / (gemv_us * 1e-6) / 1e9;

        /* rows=128 bf16 vs int8 GEMM -- quantisation ceiling proxy */
        c.rows = 128;
        double g128_bf16 = time_loop(gpu, iters, run_bf16, &c);
        double g128_int8 = NAN;
        if (int8_available) {
            if (h3_gpu_begin(gpu) &&
                h3_gpu_quantize_weight_int8(gpu, c.weight_i8, c.weight_scales,
                                            c.weight, n, k) &&
                h3_gpu_submit(gpu) && run_int8(&c))
                g128_int8 = time_loop(gpu, iters, run_int8, &c);
            else
                fprintf(stderr, "  [%s] int8 GEMM failed: %s\n",
                        shapes[s].name, h3_gpu_error(gpu));
        }

        printf("%-10s %7u %7u | %9.1f %7.1f | %8.1f %8.1f %5.2f\n",
               shapes[s].name, k, n, gemv_us, gemv_gbs, g128_bf16, g128_int8,
               isnan(g128_int8) ? NAN : g128_bf16 / g128_int8);

        tok_gemv_ms +=
            gemv_us * (shapes[s].per_layer ? 64 : 1) / 1000.0;

        h3_gpu_tensor_free(c.weight);
        h3_gpu_tensor_free(c.input);
        h3_gpu_tensor_free(c.output);
        h3_gpu_tensor_free(c.weight_i8);
        h3_gpu_tensor_free(c.weight_scales);
        h3_gpu_tensor_free(c.q_input);
        h3_gpu_tensor_free(c.input_scales);
    }

    printf("\nrows=1 bf16 linears, one submit each: %.0f ms/token -> %.2f "
           "tok/s\n(the real decode overlaps a layer's ops in one submit, so "
           "measured\ndecode is a little faster; this is the weight-movement "
           "floor.)\n",
           tok_gemv_ms, 1000.0 / tok_gemv_ms);

    h3_gpu_free(gpu);
    return 0;
}
