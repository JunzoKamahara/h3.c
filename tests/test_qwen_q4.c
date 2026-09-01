/* Unit test for the group-wise INT4 decode GEMV (chat-speedup step #2).
 *
 * No model weights: builds random BF16 projection matrices, quantises them with
 * qwen_q4_quantize(), and checks h3_gpu_linear_q4_gemv() against
 *   (a) a host dot over the *dequantised* weights  -- kernel correctness,
 *   (b) h3_gpu_linear_bf16() over the original BF16 -- quantisation loss.
 *
 *   ./h3_qwen_q4_test [h3_shaders.metal]
 */

#include "h3_gpu.h"
#include "qwen_q4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_q4.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

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

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static float next_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 11) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
}
static float next_gaussian(void) {
    /* sum of 4 uniforms in [-0.5,0.5] ~ approx normal, var 4/12 */
    return (next_uniform() + next_uniform() + next_uniform() + next_uniform()) *
           1.7320508f;
}

static void check_shape(h3_gpu *gpu, uint32_t N, uint32_t K, float weight_std) {
    size_t wcount = (size_t)N * K;
    uint16_t *w = malloc(wcount * sizeof(*w));
    uint16_t *x = malloc((size_t)K * sizeof(*x));
    float *xf = malloc((size_t)K * sizeof(*xf));
    require(w && x && xf, "host alloc");

    for (size_t i = 0; i < wcount; i++)
        w[i] = f32_to_bf16(next_gaussian() * weight_std);
    for (uint32_t i = 0; i < K; i++) {
        xf[i] = next_gaussian();
        x[i] = f32_to_bf16(xf[i]);
        xf[i] = bf16_to_f32(x[i]);
    }

    h3_gpu_tensor *w_bf16 = h3_gpu_tensor_from_bf16(gpu, w, wcount);
    h3_gpu_tensor *x_t = h3_gpu_tensor_from_bf16(gpu, x, K);
    h3_gpu_tensor *out_q4 = h3_gpu_tensor_new_bf16(gpu, N);
    h3_gpu_tensor *out_bf16 = h3_gpu_tensor_new_bf16(gpu, N);
    require(w_bf16 && x_t && out_q4 && out_bf16, "gpu alloc");

    char error[256];
    qwen_q4_weight q4;
    require(qwen_q4_quantize(gpu, w_bf16, N, K, &q4, error, sizeof(error)),
            error);

    require(h3_gpu_begin(gpu), "begin");
    require(h3_gpu_linear_q4_gemv(gpu, out_q4, x_t, q4.packed, q4.scales,
                                  q4.awq_inv_scale, NULL, K, N, QWEN_Q4_GROUP),
            "q4 gemv dispatch");
    require(h3_gpu_linear_bf16(gpu, out_bf16, x_t, w_bf16, NULL, 1, K, N),
            "bf16 linear dispatch");
    require(h3_gpu_submit(gpu), "submit");

    uint16_t *got_q4 = malloc((size_t)N * sizeof(*got_q4));
    uint16_t *got_bf16 = malloc((size_t)N * sizeof(*got_bf16));
    uint8_t *packed = malloc((size_t)N * K / 2);
    uint16_t *scales = malloc((size_t)N * (K / QWEN_Q4_GROUP) * sizeof(*scales));
    require(got_q4 && got_bf16 && packed && scales, "host alloc 2");
    require(h3_gpu_tensor_read_bf16(out_q4, got_q4, N), "read q4");
    require(h3_gpu_tensor_read_bf16(out_bf16, got_bf16, N), "read bf16");
    require(h3_gpu_tensor_read_bf16(q4.scales, scales,
                                   (size_t)N * (K / QWEN_Q4_GROUP)),
            "read scales");
    /* packed lives in an I8 tensor; read raw bytes through the f32 path is not
     * available, so recompute the host reference from w + scales directly. */

    double se_kernel = 0, se_quant = 0, sr = 0, dot_ref_ss = 0, dot_q4_ss = 0;
    double cross = 0;
    uint32_t gpr = K / QWEN_Q4_GROUP;
    for (uint32_t n = 0; n < N; n++) {
        /* dequantised-weight reference: repeat the quantiser's rounding. */
        double acc_deq = 0.0, acc_bf16 = 0.0;
        for (uint32_t g = 0; g < gpr; g++) {
            float scale = bf16_to_f32(scales[n * gpr + g]);
            float inv = 1.0f / scale;
            for (uint32_t j = 0; j < QWEN_Q4_GROUP; j++) {
                uint32_t k = g * QWEN_Q4_GROUP + j;
                float wv = bf16_to_f32(w[(size_t)n * K + k]);
                long q = lroundf(wv * inv);
                if (q < -8) q = -8; else if (q > 7) q = 7;
                acc_deq += (double)xf[k] * (double)((float)q * scale);
                acc_bf16 += (double)xf[k] * (double)wv;
            }
        }
        float ref = (float)acc_deq;
        float gq = bf16_to_f32(got_q4[n]);
        float gb = bf16_to_f32(got_bf16[n]);
        se_kernel += (double)(gq - ref) * (gq - ref);
        se_quant += (double)(gq - gb) * (gq - gb);
        sr += (double)ref * ref;
        dot_ref_ss += (double)acc_bf16 * acc_bf16;
        dot_q4_ss += (double)gq * gq;
        cross += (double)gq * (double)gb;
        (void)packed;
    }
    double rel_kernel = sqrt(se_kernel / (sr > 1e-30 ? sr : 1e-30));
    double rel_quant = sqrt(se_quant / (dot_ref_ss > 1e-30 ? dot_ref_ss : 1e-30));
    double cosine = cross / (sqrt(dot_q4_ss) * sqrt(dot_ref_ss) + 1e-30);

    printf("  N=%-5u K=%-6u  kernel_rel=%.2e  quant_rel=%.3f  cos=%.5f\n",
           N, K, rel_kernel, rel_quant, cosine);
    /* Hard check: the kernel reproduces sum(x . dequant(w)) (+bias). */
    require(rel_kernel < 3e-2,
            "q4 GEMV disagrees with its own dequantised-weight math");
    /* Soft check: iid-Gaussian weights are a worst case for relative INT4
     * error (~1/sqrt(12) of the quant step ~= 0.12); real LLM weights with a
     * per-group scale land far lower. Wide bounds here just catch a broken
     * quantiser (wrong packing, wrong scale). Real accuracy is proven by
     * resident-check (Q4 decode vs BF16) and the phase3-6 functional tests. */
    require(rel_quant < 0.22, "INT4 quantisation error is unexpectedly large");
    require(cosine > 0.98, "INT4 output direction drifted from BF16");

    free(w); free(x); free(xf);
    free(got_q4); free(got_bf16); free(packed); free(scales);
    qwen_q4_weight_free(&q4);
    h3_gpu_tensor_free(w_bf16);
    h3_gpu_tensor_free(x_t);
    h3_gpu_tensor_free(out_q4);
    h3_gpu_tensor_free(out_bf16);
}

int main(int argc, char **argv) {
    const char *shaders = argc > 1 ? argv[1] : "h3_shaders.metal";
    char error[256];
    h3_gpu *gpu = h3_gpu_create(shaders, error, sizeof(error));
    if (!gpu) fail(error);

    /* Exercises qwen_q4_quantize() + the kernel directly, independent of the
     * H3_QWEN_Q4 resident opt-in. */
    printf("group-wise INT4 decode GEMV:\n");
    check_shape(gpu, 64, 256, 0.02f);      /* tiny, N multiple of 8   */
    check_shape(gpu, 100, 384, 0.02f);     /* N not a multiple of 8   */
    check_shape(gpu, 512, 5120, 0.02f);    /* q/o/gate/up/down K      */
    check_shape(gpu, 300, 25600, 0.015f);  /* down_proj K             */
    check_shape(gpu, 1024, 5120, 0.03f);   /* k/v_proj-ish            */

    h3_gpu_free(gpu);
    puts("ok: qwen INT4 decode GEMV (kernel + quantisation error)");
    return 0;
}
