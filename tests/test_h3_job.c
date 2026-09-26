/* P8-VID-01: the FIFO job manager, exercised with a fast mock executor (no
 * model). Verifies submission order, single-worker serialisation, per-job
 * artifacts, failure reporting, and a clean stop.
 *
 * P10-CANCEL-01: cancelling a QUEUED job (never runs) and a RUNNING one
 * (mock_executor polls job->cancel_requested the way the real diffusion
 * loop polls it via h3_video_request.cancel_requested).
 *
 *   ./h3_job_test
 */

#include "h3_job.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_job.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

typedef struct {
    pthread_mutex_t mu;
    int running_now;
    int running_peak;
    char order[256];
} probe;

/* Mock: records concurrency + start order, writes a tiny artifact, and fails
 * the job whose seed is the sentinel. Polls job->cancel_requested in small
 * increments while "working" -- the same check-in granularity the real
 * diffusion loop uses via h3_video_request.cancel_requested -- so a RUNNING
 * job can be cancelled mid-flight instead of only ever finishing. */
static int mock_executor(h3_job *job, void *ctx) {
    probe *p = ctx;
    pthread_mutex_lock(&p->mu);
    p->running_now++;
    if (p->running_now > p->running_peak) p->running_peak = p->running_now;
    strncat(p->order, job->id + 4, sizeof(p->order) - strlen(p->order) - 1);
    pthread_mutex_unlock(&p->mu);

    int cancelled = 0;
    for (int waited = 0; waited < 50000; waited += 5000) {
        if (atomic_load(&job->cancel_requested)) {
            cancelled = 1;
            break;
        }
        usleep(5000);
    }

    int should_fail = 0;
    if (!cancelled) {
        FILE *f = fopen(job->output_path, "wb");
        if (f) {
            fputs("artifact", f);
            fclose(f);
        }
        should_fail = job->seed == 0xDEAD;
        if (should_fail)
            snprintf(job->error, sizeof(job->error), "forced failure");
    }

    pthread_mutex_lock(&p->mu);
    p->running_now--;
    pthread_mutex_unlock(&p->mu);
    return (cancelled || should_fail) ? 0 : 1;
}

static h3_job_status wait_terminal(h3_job_manager *m, const char *id,
                                   h3_job_info *out) {
    for (int waited_ms = 0; waited_ms < 10000; waited_ms += 10) {
        require(h3_job_get(m, id, out), "job vanished");
        if (out->status == H3_JOB_SUCCEEDED || out->status == H3_JOB_FAILED ||
            out->status == H3_JOB_CANCELLED)
            return out->status;
        usleep(10000);
    }
    fail("job did not finish in time");
    return H3_JOB_FAILED;
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && st.st_size > 0;
}

int main(void) {
    char dir[] = "/tmp/h3-job-test-XXXXXX";
    require(mkdtemp(dir) != NULL, "mkdtemp");

    char error[256];
    h3_job_manager *m = h3_job_manager_new(dir, error, sizeof(error));
    require(m != NULL, error);

    probe p;
    memset(&p, 0, sizeof(p));
    pthread_mutex_init(&p.mu, NULL);
    require(h3_job_manager_start(m, mock_executor, &p, error, sizeof(error)),
            error);

    require(!h3_job_manager_start(m, mock_executor, &p, error, sizeof(error)),
            "double start must be rejected");

    char id_a[H3_JOB_ID_SIZE], id_b[H3_JOB_ID_SIZE], id_c[H3_JOB_ID_SIZE];
    h3_job_request a = {H3_JOB_VIDEO, "clip a", 1, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    h3_job_request b = {H3_JOB_VIDEO, "clip b", 2, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    h3_job_request c = {H3_JOB_VIDEO, "clip c", 0xDEAD, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    require(h3_job_submit(m, &a, id_a, sizeof(id_a), error, sizeof(error)),
            error);
    require(h3_job_submit(m, &b, id_b, sizeof(id_b), error, sizeof(error)),
            error);
    require(h3_job_submit(m, &c, id_c, sizeof(id_c), error, sizeof(error)),
            error);
    require(strcmp(id_a, id_b) && strcmp(id_b, id_c), "ids are distinct");
    printf("(1) submitted %s %s %s\n", id_a, id_b, id_c);

    h3_job_info ia, ib, ic;
    require(wait_terminal(m, id_a, &ia) == H3_JOB_SUCCEEDED, "A succeeded");
    require(wait_terminal(m, id_b, &ib) == H3_JOB_SUCCEEDED, "B succeeded");
    require(wait_terminal(m, id_c, &ic) == H3_JOB_FAILED, "C failed");
    require(!strcmp(ic.error, "forced failure"), "C carries its error");
    printf("(2) A/B succeeded, C failed with its error\n");

    require(p.running_peak == 1, "never more than one job ran at a time");
    char expect[64];
    snprintf(expect, sizeof(expect), "%s%s%s", id_a + 4, id_b + 4, id_c + 4);
    require(!strcmp(p.order, expect), "jobs ran in submission order");
    printf("(3) single worker, FIFO order (%s)\n", p.order);

    require(file_exists(ia.output_path) && file_exists(ib.output_path),
            "A and B wrote artifacts");
    require(strcmp(ia.output_path, ib.output_path) &&
                strcmp(ib.output_path, ic.output_path),
            "each job has its own artifact path");
    require(ia.started_at <= ib.started_at && ib.started_at <= ic.started_at,
            "start timestamps are ordered");
    printf("(4) per-job artifacts, ordered timestamps\n");

    /* 5. cancel while QUEUED: D is picked up almost immediately; E, submitted
     * right behind it, should still be sitting in the queue. */
    char id_d[H3_JOB_ID_SIZE], id_e[H3_JOB_ID_SIZE];
    h3_job_request d = {H3_JOB_VIDEO, "clip d", 4, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    h3_job_request e = {H3_JOB_VIDEO, "clip e", 5, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    require(h3_job_submit(m, &d, id_d, sizeof(id_d), error, sizeof(error)),
            error);
    require(h3_job_submit(m, &e, id_e, sizeof(id_e), error, sizeof(error)),
            error);
    require(h3_job_cancel(m, id_e, error, sizeof(error)), error);
    h3_job_info info_e;
    require(h3_job_get(m, id_e, &info_e), "E exists");
    require(info_e.status == H3_JOB_CANCELLED, "E cancelled while queued");
    h3_job_info info_d;
    require(wait_terminal(m, id_d, &info_d) == H3_JOB_SUCCEEDED,
           "D still succeeded (E's cancellation did not disturb it)");
    printf("(5) cancel while QUEUED: E never ran, D unaffected\n");

    /* 6. cancel while RUNNING: F's mock_executor polls cancel_requested in
     * 5 ms increments, the same granularity the real diffusion loop uses. */
    char id_f[H3_JOB_ID_SIZE];
    h3_job_request f = {H3_JOB_VIDEO, "clip f", 6, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    require(h3_job_submit(m, &f, id_f, sizeof(id_f), error, sizeof(error)),
            error);
    h3_job_info info_f;
    int saw_running = 0;
    for (int waited = 0; waited < 200 && !saw_running; waited++) {
        require(h3_job_get(m, id_f, &info_f), "F exists");
        if (info_f.status == H3_JOB_RUNNING) saw_running = 1;
        else usleep(1000);
    }
    require(saw_running, "F entered RUNNING before being cancelled");
    require(h3_job_cancel(m, id_f, error, sizeof(error)), error);
    require(wait_terminal(m, id_f, &info_f) == H3_JOB_CANCELLED,
           "F stopped mid-flight instead of running to completion");
    require(!strcmp(info_f.error, "cancelled"),
           "F carries a clean cancellation message, not the mock's internals");
    printf("(6) cancel while RUNNING: F stopped mid-flight\n");

    /* 7. cancelling an unknown id or an already-terminal job is rejected. */
    require(!h3_job_cancel(m, id_f, error, sizeof(error)),
           "cancelling an already-cancelled job is rejected");
    require(!h3_job_cancel(m, "job-does-not-exist", error, sizeof(error)),
           "cancelling an unknown id is rejected");
    printf("(7) cancel rejected for an unknown id / an already-terminal job\n");

    /* 8. the worker is still healthy after a cancellation. */
    char id_g[H3_JOB_ID_SIZE];
    h3_job_request g = {H3_JOB_VIDEO, "clip g", 7, 256, 256, 25,
                        H3_JOB_REF_NONE, NULL, NULL, NULL};
    require(h3_job_submit(m, &g, id_g, sizeof(id_g), error, sizeof(error)),
            error);
    h3_job_info info_g;
    require(wait_terminal(m, id_g, &info_g) == H3_JOB_SUCCEEDED,
           "G still runs to completion after a cancellation");
    printf("(8) worker still healthy: G ran to completion after a cancel\n");

    h3_job_manager_stop(m);
    h3_job_manager_stop(m); /* idempotent */
    h3_job_manager_free(m);
    pthread_mutex_destroy(&p.mu);

    unlink(ia.output_path);
    unlink(ib.output_path);
    unlink(ic.output_path);
    unlink(info_d.output_path);
    unlink(info_g.output_path);
    rmdir(dir);
    puts("ok: P8-VID-01 job manager + P10-CANCEL-01 cancellation (mock "
        "executor)");
    return 0;
}
