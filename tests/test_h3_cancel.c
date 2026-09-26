/* P10-CANCEL-01 (slow): a REAL video job cancelled mid-diffusion, through
 * the same job manager + generation engine the server uses.
 *
 *   ./h3_cancel_test MiniMax-H3
 *
 * Proves three things the mock-executor gate (make job-check) cannot:
 *   1. cancelling a job actually stops it well before its natural ~70-80 s
 *      completion (P8-MEM-01) -- not just a status label flip.
 *   2. the cancellation check-in point (the diffusion Euler loop, see
 *      cancel_check() in h3_image_gen.c) is real code on the hot path, not
 *      dead plumbing.
 *   3. the shared generation engine (GPU scheduler, keep-alive thread,
 *      conditioning lock) is still healthy afterwards -- a second job
 *      submitted right after runs to completion normally.
 *
 * Loads the FL2VA transformer (SSD streaming) twice: once for the
 * cancelled job (short), once for the follow-up (full length). Not in
 * `make test`.
 */

#include "h3_ffmpeg.h"
#include "h3_generation.h"
#include "h3_job.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_cancel.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];

    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tokenizer_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *fl2va = path_join(root, "FL2VA");

    qwen_engine *engine = NULL;
    require(qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                            sizeof(error)),
            error);
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    require(tokenizer != NULL, error);
    h3_tokenizer_free(tokenizer); /* not needed past validating it loads */

    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);

    h3_generation_engine *gen = h3_generation_engine_acquire(
        engine, fl2va, NULL /* no Ref2VA checkpoint needed for this gate */,
        "h3_shaders.metal", &lock, error, sizeof(error));
    require(gen != NULL, error);

    char artifact_dir[] = "/tmp/h3-cancel-XXXXXX";
    require(mkdtemp(artifact_dir) != NULL, "mkdtemp");
    h3_job_manager *manager =
        h3_job_manager_new(artifact_dir, error, sizeof(error));
    require(manager != NULL, error);
    require(h3_job_manager_start(manager, h3_generation_run_job, gen, error,
                                sizeof(error)),
            error);

    /* 1. submit a real video job -- same size/frames as p8-vid-job-check,
     * which measured this taking ~70-80 s to run to completion. */
    h3_job_request request = {H3_JOB_VIDEO, "A slow-moving glacier calving",
                              42, 256, 256, 25, H3_JOB_REF_NONE, NULL, NULL,
                              NULL};
    char id[H3_JOB_ID_SIZE];
    double t_submit = now_seconds();
    require(h3_job_submit(manager, &request, id, sizeof(id), error,
                         sizeof(error)),
            error);
    printf("(1) submitted %s\n", id);

    /* 2. wait for it to actually start, then let it run a couple of seconds
     * into the diffusion body before cancelling. */
    h3_job_info info;
    int saw_running = 0;
    for (int waited = 0; waited < 30000 && !saw_running; waited += 100) {
        require(h3_job_get(manager, id, &info), "job vanished");
        if (info.status == H3_JOB_RUNNING) saw_running = 1;
        else usleep(100000);
    }
    require(saw_running, "job entered RUNNING within 30s");
    printf("(2) job is RUNNING (%.1fs after submit)\n",
          now_seconds() - t_submit);
    sleep(2);

    /* 3. cancel it, and time how long it takes to actually stop. */
    double t_cancel = now_seconds();
    require(h3_job_cancel(manager, id, error, sizeof(error)), error);
    int cancelled = 0;
    for (int waited = 0; waited < 60000 && !cancelled; waited += 200) {
        require(h3_job_get(manager, id, &info), "job vanished");
        if (info.status == H3_JOB_CANCELLED) cancelled = 1;
        else if (info.status == H3_JOB_SUCCEEDED || info.status == H3_JOB_FAILED) {
            fprintf(stderr, "job finished as %s instead of being cancelled: %s\n",
                    h3_job_status_name(info.status), info.error);
            fail("job did not honor the cancel request");
        } else {
            usleep(200000);
        }
    }
    double stop_latency = now_seconds() - t_cancel;
    double total_elapsed = now_seconds() - t_submit;
    require(cancelled, "job reached CANCELLED within 60s of the cancel call");
    require(!strcmp(info.error, "cancelled"),
           "job carries a clean cancellation message");
    printf("(3) cancelled %.1fs after the cancel call (%.1fs total, vs the "
          "~70-80s this job normally takes) -- error=\"%s\"\n",
          stop_latency, total_elapsed, info.error);
    /* Generous bound: conditioning + transformer load + a handful of
     * diffusion blocks, well under the ~70-80s natural completion time. */
    require(total_elapsed < 45.0,
           "cancellation did not meaningfully shorten the job");

    /* 4. cancelling it again is rejected -- it is already terminal. */
    require(!h3_job_cancel(manager, id, error, sizeof(error)),
           "cancelling an already-cancelled job is rejected");
    printf("(4) re-cancelling the same job is rejected\n");

    /* 5. the engine is still healthy: a follow-up job runs to completion. */
    h3_job_request request2 = {H3_JOB_VIDEO, "A calm still scene.", 43, 256,
                               256, 25, H3_JOB_REF_NONE, NULL, NULL, NULL};
    char id2[H3_JOB_ID_SIZE];
    require(h3_job_submit(manager, &request2, id2, sizeof(id2), error,
                         sizeof(error)),
            error);
    h3_job_info info2;
    for (int waited = 0; waited < 1200000; waited += 500) {
        require(h3_job_get(manager, id2, &info2), "job vanished");
        if (info2.status == H3_JOB_SUCCEEDED || info2.status == H3_JOB_FAILED)
            break;
        usleep(500000);
    }
    require(info2.status == H3_JOB_SUCCEEDED,
           info2.error[0] ? info2.error
                          : "follow-up job did not succeed after a cancel");
    int w = 0, h = 0;
    require(h3_ffprobe_visual_size(info2.output_path, &w, &h, error,
                                   sizeof(error)),
           error);
    require(w == 256 && h == 256,
           "follow-up job produced a real 256x256 video");
    printf("(5) engine still healthy: a follow-up job ran to completion "
          "after the cancellation\n");

    h3_job_manager_free(manager);
    h3_generation_engine_release(gen);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    unlink(info2.output_path);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    puts("ok: P10-CANCEL-01 real job cancellation mid-diffusion");
    return 0;
}
