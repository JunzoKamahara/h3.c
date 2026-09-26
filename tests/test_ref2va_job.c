/* P10-REF2VA-01/02 (slow): real Ref2VA video jobs through the SAME job
 * manager + generation engine the server uses -- not the standalone CLI
 * generator that P10-REF2VA-00 validated. Proves the engine/job integration
 * (own conditioning path, own Ref2VA checkpoint selection, refcounted
 * condition-row lifetime) produces the same kind of result for BOTH
 * reference kinds the job engine accepts: swapping the reference measurably
 * changes the output, through h3_job_submit().
 *
 *   ./h3_ref2va_job_test MiniMax-H3
 *
 * Loads the Ref2VA transformer (SSD streaming) + both VAEs, four times (two
 * reference kinds x two references each). Not in `make test`.
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
#define IMAGE_REF_FRAMES 22 /* Ref2VA's floor -- see h3_generation.c */
#define VIDEO_REF_FRAMES 39 /* aligned; yields 2 Qwen vision blocks, not 1 */

static void write_solid_reference_image(const char *path, uint8_t r,
                                        uint8_t g, uint8_t b) {
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

static void write_solid_reference_video(const char *path, uint8_t r,
                                        uint8_t g, uint8_t b, int frames) {
    size_t frame_bytes = (size_t)REF_SIZE * REF_SIZE * 3;
    uint8_t *pixels = malloc(frame_bytes * (size_t)frames);
    if (!pixels) fail("out of memory building a synthetic reference video");
    for (size_t f = 0; f < (size_t)frames; f++)
        for (size_t index = 0; index < (size_t)REF_SIZE * REF_SIZE; index++) {
            uint8_t *px = pixels + f * frame_bytes + index * 3;
            px[0] = r; px[1] = g; px[2] = b;
        }
    char error[256];
    if (!h3_ffmpeg_write_rgb24(path, pixels, frames, REF_SIZE, REF_SIZE, 24,
                              error, sizeof(error)))
        fail(error);
    free(pixels);
}

static void run_job(h3_job_manager *manager, h3_job_reference_kind kind,
                    const char *reference_path, int frames,
                    h3_job_info *info_out) {
    h3_job_request request = {H3_JOB_VIDEO, "A calm still scene.", 42,
                              REF_SIZE, REF_SIZE, frames, kind,
                              reference_path};
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

/* Decodes both clips and requires they differ by a real margin -- the same
 * wiring check P10-REF2VA-00 established, reused for each reference kind. */
static void compare_and_report(const char *label, const h3_job_info *info_a,
                               const h3_job_info *info_b, int frames) {
    float *pixels_a = NULL, *pixels_b = NULL;
    int frames_a = 0, frames_b = 0;
    char error[512];
    require(h3_ffmpeg_read_video_f32(info_a->output_path, REF_SIZE, REF_SIZE,
                                     frames, &pixels_a, &frames_a, error,
                                     sizeof(error)),
           error);
    require(h3_ffmpeg_read_video_f32(info_b->output_path, REF_SIZE, REF_SIZE,
                                     frames, &pixels_b, &frames_b, error,
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

    printf("ref2va-job (%s): pixel variance (clip A) = %.6f\n", label,
          variance_a);
    printf("ref2va-job (%s): mean abs diff (A vs B)   = %.6f\n", label,
          mean_abs_diff);
    require(variance_a > 1e-4, "clip A is degenerate (near-constant output)");
    require(mean_abs_diff > 1e-3,
           "swapping the reference through h3_job_submit() produced no "
           "measurable change in the output");

    free(pixels_a);
    free(pixels_b);
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

    /* -- IMAGE reference (P10-REF2VA-01) -- */
    char img_a[1024], img_b[1024];
    snprintf(img_a, sizeof(img_a), "%s/img-a.png", artifact_dir);
    snprintf(img_b, sizeof(img_b), "%s/img-b.png", artifact_dir);
    write_solid_reference_image(img_a, 220, 40, 40);
    write_solid_reference_image(img_b, 40, 60, 220);

    printf("ref2va-job: submitting IMAGE clip A (red reference)\n");
    h3_job_info img_info_a;
    run_job(manager, H3_JOB_REF_IMAGE, img_a, IMAGE_REF_FRAMES, &img_info_a);
    printf("ref2va-job: submitting IMAGE clip B (blue reference)\n");
    h3_job_info img_info_b;
    run_job(manager, H3_JOB_REF_IMAGE, img_b, IMAGE_REF_FRAMES, &img_info_b);
    compare_and_report("image", &img_info_a, &img_info_b, IMAGE_REF_FRAMES);

    /* -- VIDEO reference (P10-REF2VA-02) -- */
    char vid_a[1024], vid_b[1024];
    snprintf(vid_a, sizeof(vid_a), "%s/vid-a.mp4", artifact_dir);
    snprintf(vid_b, sizeof(vid_b), "%s/vid-b.mp4", artifact_dir);
    write_solid_reference_video(vid_a, 220, 40, 40, VIDEO_REF_FRAMES);
    write_solid_reference_video(vid_b, 40, 60, 220, VIDEO_REF_FRAMES);

    printf("ref2va-job: submitting VIDEO clip A (red reference)\n");
    h3_job_info vid_info_a;
    run_job(manager, H3_JOB_REF_VIDEO, vid_a, VIDEO_REF_FRAMES, &vid_info_a);
    printf("ref2va-job: submitting VIDEO clip B (blue reference)\n");
    h3_job_info vid_info_b;
    run_job(manager, H3_JOB_REF_VIDEO, vid_b, VIDEO_REF_FRAMES, &vid_info_b);
    compare_and_report("video", &vid_info_a, &vid_info_b, VIDEO_REF_FRAMES);

    h3_job_manager_free(manager);
    h3_generation_engine_release(gen);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    unlink(img_info_a.output_path);
    unlink(img_info_b.output_path);
    unlink(vid_info_a.output_path);
    unlink(vid_info_b.output_path);
    unlink(img_a);
    unlink(img_b);
    unlink(vid_a);
    unlink(vid_b);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    free(ref2va);
    puts("ok: P10-REF2VA-01/02 real Ref2VA jobs (image + video reference) "
        "through h3_job_submit()");
    return 0;
}
