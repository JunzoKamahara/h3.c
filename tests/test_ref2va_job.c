/* P10-REF2VA-01 (slow): one real Ref2VA video job through the SAME job
 * manager + generation engine the server uses -- not the standalone CLI
 * generator that P10-REF2VA-00 validated. Proves the engine/job integration
 * (own conditioning path, own Ref2VA checkpoint selection, refcounted
 * condition-row lifetime) produces the same kind of result: swapping the
 * reference image measurably changes the output, through h3_job_submit().
 *
 *   ./h3_ref2va_job_test MiniMax-H3
 *
 * Loads the Ref2VA transformer (SSD streaming) + both VAEs, twice. Not in
 * `make test`.
 */

#include "h3_ffmpeg.h"
#include "h3_generation.h"
#include "h3_job.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_ref2va_job.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

#define REF_SIZE 256
#define REF_FRAMES 22 /* Ref2VA generation's floor -- see h3_generation.c */

static void write_solid_reference(const char *path, uint8_t r, uint8_t g,
                                  uint8_t b) {
    size_t count = (size_t)REF_SIZE * REF_SIZE;
    uint8_t *pixels = malloc(count * 3);
    if (!pixels) fail("out of memory building a synthetic reference image");
    for (size_t index = 0; index < count; index++) {
        pixels[index * 3 + 0] = r;
        pixels[index * 3 + 1] = g;
        pixels[index * 3 + 2] = b;
    }
    char error[256];
    if (!h3_ffmpeg_write_png_rgb24(path, pixels, REF_SIZE, REF_SIZE, error,
                                   sizeof(error)))
        fail(error);
    free(pixels);
}

static void run_job(h3_job_manager *manager, const char *reference_path,
                    h3_job_info *info_out) {
    h3_job_request request = {H3_JOB_VIDEO, "A calm still scene.", 42,
                              REF_SIZE, REF_SIZE, REF_FRAMES, reference_path};
    char id[H3_JOB_ID_SIZE];
    char error[512];
    require(h3_job_submit(manager, &request, id, sizeof(id), error,
                         sizeof(error)),
            error);
    h3_job_info info;
    for (int waited = 0; waited < 1200000; waited += 500) {
        require(h3_job_get(manager, id, &info), "job vanished");
        if (info.status == H3_JOB_SUCCEEDED || info.status == H3_JOB_FAILED)
            break;
        usleep(500000);
    }
    require(info.status == H3_JOB_SUCCEEDED, info.error);
    *info_out = info;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];

    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tokenizer_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *fl2va = path_join(root, "FL2VA");
    char *ref2va = path_join(root, "Ref2VA");

    struct stat probe;
    char *ref2va_transformer = path_join(ref2va, "transformer");
    if (stat(ref2va_transformer, &probe) != 0) {
        fprintf(stderr,
                "skip: Ref2VA transformer checkpoint is not installed\n");
        return 0;
    }
    free(ref2va_transformer);

    qwen_engine *engine = NULL;
    require(qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                            sizeof(error)),
            error);
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    require(tokenizer != NULL, error);
    (void)tokenizer; /* only needed if this test grows a chat-concurrency leg */

    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);

    h3_generation_engine *gen = h3_generation_engine_acquire(
        engine, fl2va, ref2va, "h3_shaders.metal", &lock, error,
        sizeof(error));
    require(gen != NULL, error);

    char artifact_dir[] = "/tmp/h3-ref2va-job-XXXXXX";
    require(mkdtemp(artifact_dir) != NULL, "mkdtemp");
    h3_job_manager *manager =
        h3_job_manager_new(artifact_dir, error, sizeof(error));
    require(manager != NULL, error);
    require(h3_job_manager_start(manager, h3_generation_run_job, gen, error,
                                sizeof(error)),
            error);

    char ref_a[1024], ref_b[1024];
    snprintf(ref_a, sizeof(ref_a), "%s/ref-a.png", artifact_dir);
    snprintf(ref_b, sizeof(ref_b), "%s/ref-b.png", artifact_dir);
    write_solid_reference(ref_a, 220, 40, 40);
    write_solid_reference(ref_b, 40, 60, 220);

    printf("ref2va-job: submitting clip A (red reference)\n");
    h3_job_info info_a;
    run_job(manager, ref_a, &info_a);
    printf("ref2va-job: submitting clip B (blue reference)\n");
    h3_job_info info_b;
    run_job(manager, ref_b, &info_b);

    float *pixels_a = NULL, *pixels_b = NULL;
    int frames_a = 0, frames_b = 0;
    require(h3_ffmpeg_read_video_f32(info_a.output_path, REF_SIZE, REF_SIZE,
                                     REF_FRAMES, &pixels_a, &frames_a, error,
                                     sizeof(error)),
           error);
    require(h3_ffmpeg_read_video_f32(info_b.output_path, REF_SIZE, REF_SIZE,
                                     REF_FRAMES, &pixels_b, &frames_b, error,
                                     sizeof(error)),
           error);
    require(frames_a == frames_b,
           "the two clips decoded to different frame counts");

    size_t total = (size_t)3 * (size_t)frames_a * REF_SIZE * REF_SIZE;
    double sum_abs_diff = 0.0, mean_a = 0.0;
    for (size_t index = 0; index < total; index++) {
        require(isfinite(pixels_a[index]) && isfinite(pixels_b[index]),
               "generated output contains a non-finite pixel");
        sum_abs_diff += fabs((double)pixels_a[index] - (double)pixels_b[index]);
        mean_a += pixels_a[index];
    }
    mean_a /= (double)total;
    double sum_var_a = 0.0;
    for (size_t index = 0; index < total; index++) {
        double d = (double)pixels_a[index] - mean_a;
        sum_var_a += d * d;
    }
    double mean_abs_diff = sum_abs_diff / (double)total;
    double variance_a = sum_var_a / (double)total;

    printf("ref2va-job: pixel variance (clip A)     = %.6f\n", variance_a);
    printf("ref2va-job: mean abs diff (A vs B)       = %.6f\n",
          mean_abs_diff);
    require(variance_a > 1e-4, "clip A is degenerate (near-constant output)");
    require(mean_abs_diff > 1e-3,
           "swapping the reference image through h3_job_submit() produced no "
           "measurable change in the output");

    free(pixels_a);
    free(pixels_b);
    h3_job_manager_free(manager);
    h3_generation_engine_release(gen);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    unlink(info_a.output_path);
    unlink(info_b.output_path);
    unlink(ref_a);
    unlink(ref_b);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    free(ref2va);
    puts("ok: P10-REF2VA-01 real Ref2VA job through h3_job_submit()");
    return 0;
}
