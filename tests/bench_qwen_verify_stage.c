/* QINT-015e-1 -- verify-stage microbenchmark. No model weights.
 *
 * QINT-015d-3 showed verify-M costs ~2x scalar-1 already at M=2 and only ~80 ms
 * per extra row. This isolates *which projection* on the Mixed-W4/BF16 verify
 * path carries that fixed cost, at the real model shapes, for M in 1..5:
 *
 *   BF16 stages  (h3_gpu_linear_bf16): k/v on layers 0..49, every projection
 *                on layers 50..63, and the lm_head -- rows==1 takes the
 *                dedicated GEMV, rows 2..5 fall into the 16x16 tiled kernel.
 *   W4 stages    (h3_gpu_linear_q4_gemv / _decode_batch): q/o/gate/up/down on
 *                layers 0..49.
 *
 * Per stage and M it reports scalar-1, scalar-M (M sequential rows==1 calls)
 * and batch-M (one rows==M call). Warm-up first; median of 20 reps.
 *
 *   ./h3_qwen_verify_stage_bench [h3_shaders.metal]
 */

#include "h3_gpu.h"
#include "qwen_q4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NREP 20
#define WARM 4
#define MAXM 5

static void fail(const char *m) {
    fprintf(stderr, "FAIL bench_qwen_verify_stage: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a - *(const double *)b;
    return x < 0 ? -1 : x > 0 ? 1 : 0;
}
static double median(double *s, int n) {
    qsort(s, (size_t)n, sizeof(double), cmp_d);
    return s[n / 2];
}

static uint64_t rng = 0x2545F4914F6CDD1DULL;
static float g(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    float u = (float)((rng >> 11) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    float v = (float)((rng >> 11) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
    return (u + v) * 1.5f;
}
static uint16_t f32_bf16(float x) {
    uint32_t b;
    memcpy(&b, &x, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

typedef struct {
    h3_gpu *gpu;
    const char *name;
    uint32_t K, N;
    int is_q4;
    h3_gpu_tensor *w_bf16;   /* [N,K] */
    qwen_q4_weight q4;       /* when is_q4 */
    h3_gpu_tensor *x[MAXM];  /* MAXM separate [K] inputs */
    h3_gpu_tensor *xM;       /* one [MAXM,K] input */
    h3_gpu_tensor *o1;       /* [N] */
    h3_gpu_tensor *oM;       /* [MAXM,N] */
} stage;

static void stage_init(stage *s, h3_gpu *gpu, const char *name, uint32_t K,
                       uint32_t N, int is_q4) {
    char err[256];
    memset(s, 0, sizeof(*s));
    s->gpu = gpu;
    s->name = name;
    s->K = K;
    s->N = N;
    s->is_q4 = is_q4;

    size_t wc = (size_t)N * K;
    uint16_t *w = malloc(wc * sizeof(*w));
    require(w != NULL, "alloc w");
    for (size_t i = 0; i < wc; i++) w[i] = f32_bf16(g() * 0.02f);
    s->w_bf16 = h3_gpu_tensor_from_bf16(gpu, w, wc);
    require(s->w_bf16 != NULL, "gpu w");
    if (is_q4)
        require(qwen_q4_quantize(gpu, s->w_bf16, N, K, &s->q4, err, sizeof(err)),
                err);
    free(w);

    uint16_t *xm = malloc((size_t)MAXM * K * sizeof(*xm));
    require(xm != NULL, "alloc x");
    for (size_t i = 0; i < (size_t)MAXM * K; i++) xm[i] = f32_bf16(g());
    s->xM = h3_gpu_tensor_from_bf16(gpu, xm, (size_t)MAXM * K);
    for (int m = 0; m < MAXM; m++)
        s->x[m] = h3_gpu_tensor_from_bf16(gpu, xm + (size_t)m * K, K);
    free(xm);
    s->o1 = h3_gpu_tensor_new_bf16(gpu, N);
    s->oM = h3_gpu_tensor_new_bf16(gpu, (size_t)MAXM * N);
    require(s->xM && s->o1 && s->oM, "gpu x/o");
}

static int do_scalar1(stage *s, h3_gpu_tensor *xrow) {
    if (s->is_q4)
        return h3_gpu_linear_q4_gemv(s->gpu, s->o1, xrow, s->q4.packed,
                                     s->q4.scales, s->q4.awq_inv_scale, NULL,
                                     s->K, s->N, QWEN_Q4_GROUP);
    return h3_gpu_linear_bf16(s->gpu, s->o1, xrow, s->w_bf16, NULL, 1, s->K,
                              s->N);
}
static int do_batchM(stage *s, uint32_t M) {
    if (s->is_q4)
        return h3_gpu_linear_q4_decode_batch(s->gpu, s->oM, s->xM, s->q4.packed,
                                             s->q4.scales, s->q4.awq_inv_scale,
                                             NULL, M, s->K, s->N,
                                             QWEN_Q4_GROUP);
    return h3_gpu_linear_bf16(s->gpu, s->oM, s->xM, s->w_bf16, NULL, M, s->K,
                              s->N);
}

/* time one submitted-and-waited dispatch group */
static double timed(h3_gpu *gpu, int (*body)(stage *, uint32_t), stage *s,
                    uint32_t arg, int reps) {
    char err[64];
    (void)err;
    for (int w = 0; w < WARM; w++) {
        require(h3_gpu_begin(gpu), "begin");
        require(body(s, arg), "warm dispatch");
        require(h3_gpu_submit(gpu), "submit");
    }
    double t[NREP];
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        require(h3_gpu_begin(gpu), "begin");
        require(body(s, arg), "dispatch");
        require(h3_gpu_submit(gpu), "submit");
        t[r] = now_ms() - t0;
    }
    return median(t, reps);
}

static int body_scalar1(stage *s, uint32_t m) { return do_scalar1(s, s->x[m]); }
static int body_scalarM(stage *s, uint32_t M) {
    for (uint32_t i = 0; i < M; i++)
        if (!do_scalar1(s, s->x[i])) return 0;
    return 1;
}
static int body_batchM(stage *s, uint32_t M) { return do_batchM(s, M); }

static void run_stage(h3_gpu *gpu, const char *name, uint32_t K, uint32_t N,
                      int is_q4) {
    stage s;
    stage_init(&s, gpu, name, K, N, is_q4);
    double sc1 = timed(gpu, body_scalar1, &s, 0, NREP);
    printf("  %-16s K=%-6u N=%-7u %s  scalar-1 %.3f ms\n", name, K, N,
           is_q4 ? "W4  " : "BF16", sc1);
    printf("      %-5s %10s %10s %9s %9s\n", "M", "scalar-M", "batch-M",
           "b / s1M", "b vs sM");
    for (uint32_t M = 2; M <= MAXM; M++) {
        double scM = timed(gpu, body_scalarM, &s, M, NREP);
        double bM = 0.0;
        int have_batch = 1;
        /* q4 decode-batch only exists for M 2..5; bf16 always has a rows=M
         * path (the 16x16 tile). */
        {
            require(h3_gpu_begin(gpu), "begin");
            have_batch = body_batchM(&s, M);
            require(h3_gpu_submit(gpu), "submit");
        }
        if (have_batch) bM = timed(gpu, body_batchM, &s, M, NREP);
        printf("      M=%-3u %10.3f %10s %9s %9s\n", M, scM,
               have_batch ? "" : "n/a", "", "");
        if (have_batch)
            printf("            %20.3f %9.2f %9.2f  (ms)\n", bM,
                   bM / (sc1 * M), bM / scM);
    }
    (void)s;
}

int main(int argc, char **argv) {
    const char *shaders = argc > 1 ? argv[1] : "h3_shaders.metal";
    char err[256];
    h3_gpu *gpu = h3_gpu_create(shaders, err, sizeof(err));
    if (!gpu) fail(err);

    printf("QINT-015e-1 verify-stage microbench (no model), reps=%d warm=%d\n",
           NREP, WARM);
    printf("b/s1M = batch-M / (scalar-1 * M)   (1.0 = as slow as M scalars)\n");
    printf("b vs sM = batch-M / scalar-M       (<1.0 = batch faster)\n\n");

    /* Mixed-W4/BF16 layer 0..49 W4 projections */
    run_stage(gpu, "W4 q_proj",  5120,  8192, 1);
    run_stage(gpu, "W4 o_proj",  8192,  5120, 1);
    run_stage(gpu, "W4 gate/up", 5120, 25600, 1);
    run_stage(gpu, "W4 down",   25600,  5120, 1);
    /* Mixed BF16: K/V (0..49), and everything on 50..63 */
    run_stage(gpu, "BF16 k/v",   5120,  1024, 0);
    run_stage(gpu, "BF16 tail-q",5120,  8192, 0);
    run_stage(gpu, "BF16 tail-o",8192,  5120, 0);
    run_stage(gpu, "BF16 tailMLP",5120,25600, 0);
    run_stage(gpu, "BF16 lm_head",5120,151936,0);

    h3_gpu_free(gpu);
    puts("\nok: verify-stage microbench");
    return 0;
}
