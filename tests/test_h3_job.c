/* P8-VID-01: the FIFO job manager, exercised with a fast mock executor (no
 * model). Verifies submission order, single-worker serialisation, per-job
 * artifacts, failure reporting, and a clean stop.
 *
 *   ./h3_job_test
 */

#include "h3_job.h"

#include <pthread.h>
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
 * the job whose seed is the sentinel. */
static int mock_executor(h3_job *job, void *ctx) {
    probe *p = ctx;
    pthread_mutex_lock(&p->mu);
    p->running_now++;
    if (p->running_now > p->running_peak) p->running_peak = p->running_now;
    strncat(p->order, job->id + 4, sizeof(p->order) - strlen(p->order) - 1);
    pthread_mutex_unlock(&p->mu);

    usleep(50000);
    FILE *f = fopen(job->output_path, "wb");
    if (f) {
        fputs("artifact", f);
        fclose(f);
    }
    int should_fail = job->seed == 0xDEAD;
    if (should_fail)
        snprintf(job->error, sizeof(job->error), "forced failure");

    pthread_mutex_lock(&p->mu);
    p->running_now--;
    pthread_mutex_unlock(&p->mu);
    return should_fail ? 0 : 1;
}

static h3_job_status wait_terminal(h3_job_manager *m, const char *id,
                                   h3_job_info *out) {
    for (int waited_ms = 0; waited_ms < 10000; waited_ms += 10) {
        require(h3_job_get(m, id, out), "job vanished");
        if (out->status == H3_JOB_SUCCEEDED || out->status == H3_JOB_FAILED)
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
    h3_job_request a = {H3_JOB_VIDEO, "clip a", 1, 256, 256, 25};
    h3_job_request b = {H3_JOB_VIDEO, "clip b", 2, 256, 256, 25};
    h3_job_request c = {H3_JOB_VIDEO, "clip c", 0xDEAD, 256, 256, 25};
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

    h3_job_manager_stop(m);
    h3_job_manager_stop(m); /* idempotent */
    h3_job_manager_free(m);
    pthread_mutex_destroy(&p.mu);

    unlink(ia.output_path);
    unlink(ib.output_path);
    unlink(ic.output_path);
    rmdir(dir);
    puts("ok: P8-VID-01 job manager (mock executor)");
    return 0;
}
