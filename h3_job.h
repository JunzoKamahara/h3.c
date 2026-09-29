#ifndef H3_JOB_H
#define H3_JOB_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

/* P8-VID-01: a process-local, non-persistent FIFO job manager for media
 * generation. One background worker runs at most one job at a time in
 * submission order. No progress, no priority, no persistence yet.
 *
 * The manager knows nothing about diffusion or the language model: the actual
 * work is an injected executor callback (`h3_job_executor`). Tests inject a
 * fast mock; the server injects the real generation path.
 *
 * P10-CANCEL-01: cancellation is cooperative. A QUEUED job is cancelled
 * immediately (it never runs). A RUNNING job only stops at the diffusion
 * loop's per-step boundary (h3_job.cancel_requested, checked by the
 * generation code via the request's cancel_requested pointer) -- not during
 * conditioning or the final VAE decode/mux, both short relative to the
 * diffusion body. */

typedef enum {
    H3_JOB_QUEUED,
    H3_JOB_RUNNING,
    H3_JOB_SUCCEEDED,
    H3_JOB_FAILED,
    H3_JOB_CANCELLED
} h3_job_status;

typedef enum {
    H3_JOB_IMAGE,
    H3_JOB_VIDEO,
    H3_JOB_AUDIO
} h3_job_type;

/* P10-MULTIREF-01: one ordered Ref2VA reference. IMAGE/VIDEO are visual;
 * AUDIO is a standalone audio reference -- the canonical model never accepts
 * an all-audio reference set (h3.c's own h3_valid_params()), so a non-empty
 * reference array with no IMAGE/VIDEO entry is invalid, checked where the
 * array is consumed (compute_ref2va_conditioning() in h3_generation.c), not
 * here -- this manager knows nothing about diffusion semantics. */
typedef enum {
    H3_JOB_REF_IMAGE,
    H3_JOB_REF_VIDEO,
    H3_JOB_REF_AUDIO
} h3_job_reference_kind;

#define H3_JOB_ID_SIZE 24
#define H3_JOB_ERROR_SIZE 512
#define H3_JOB_PATH_SIZE 1024
/* Ref2VA's own limit (h3.c's h3_valid_params()): at most 12 ordered
 * references total (also capped per kind -- 9 image / 3 video / 3 audio --
 * re-validated in compute_ref2va_conditioning() where the per-kind counts
 * are actually known). */
#define H3_JOB_MAX_REFERENCES 12

/* One ordered reference, owned by the job; its path is freed with it. */
typedef struct {
    h3_job_reference_kind kind;
    char *path;
} h3_job_reference;

/* Same shape as h3_job_reference, but the path is borrowed -- used only in
 * h3_job_request, which h3_job_submit() copies from. */
typedef struct {
    h3_job_reference_kind kind;
    const char *path;
} h3_job_reference_request;

/* One unit of work. The worker fills `output_path` before the executor runs
 * (it is `<artifact_dir>/<id>.<ext>`); the executor must write its artifact
 * there. On failure the executor fills `error` and returns 0. */
typedef struct {
    char id[H3_JOB_ID_SIZE];
    h3_job_type type;
    h3_job_status status;

    char *prompt;
    uint64_t seed;
    int width;
    int height;
    int frames;             /* video only; 0 otherwise */
    int steps;               /* P10-PARAMS-02: 0 = use the engine's own default */
    h3_job_reference *references; /* owned array; NULL/0 = plain T2VA */
    size_t reference_count;

    /* P10-CANCEL-01: set by h3_job_cancel() while RUNNING; the generation
     * code polls this (via h3_job_request.cancel_requested, a borrowed
     * pointer to this field) at each diffusion step. Not touched while
     * QUEUED -- h3_job_cancel() flips a queued job straight to CANCELLED
     * instead, since it never started. */
    _Atomic int cancel_requested;

    char output_path[H3_JOB_PATH_SIZE];
    char error[H3_JOB_ERROR_SIZE];

    int64_t created_at;     /* unix seconds */
    int64_t started_at;     /* 0 until RUNNING */
    int64_t finished_at;    /* 0 until terminal (SUCCEEDED / FAILED / CANCELLED) */
} h3_job;

/* Runs on the worker thread, one job at a time. Return 1 on success (artifact
 * written to job->output_path) or 0 on failure (job->error filled). `ctx` is
 * the pointer passed to h3_job_manager_start(). */
typedef int (*h3_job_executor)(h3_job *job, void *ctx);

typedef struct {
    h3_job_type type;
    const char *prompt;
    uint64_t seed;
    int width;
    int height;
    int frames;             /* video only */
    int steps;               /* P10-PARAMS-02: 0 = use the engine's own default */
    /* P10-MULTIREF-01: an ordered array of Ref2VA references (borrowed;
     * h3_job_submit() copies it); NULL/0 keeps the plain T2VA path. */
    const h3_job_reference_request *references;
    size_t reference_count;
    /* P10-CANCEL-01: borrowed pointer the generation code polls during the
     * diffusion loop; NULL means not cancellable (e.g. a caller not going
     * through h3_generation_run_job()). h3_generation_run_job() sets this to
     * &job->cancel_requested when it rebuilds a request from a live job. */
    const _Atomic int *cancel_requested;
} h3_job_request;

/* Read-only snapshot returned by h3_job_get(). */
typedef struct {
    char id[H3_JOB_ID_SIZE];
    h3_job_type type;
    h3_job_status status;
    char output_path[H3_JOB_PATH_SIZE];
    char error[H3_JOB_ERROR_SIZE];
    int64_t created_at;
    int64_t started_at;
    int64_t finished_at;
} h3_job_info;

typedef struct h3_job_manager h3_job_manager;

/* `artifact_dir` must already exist; the worker writes finished artifacts
 * into it. The manager does not own or clean the directory. */
h3_job_manager *h3_job_manager_new(const char *artifact_dir,
                                   char *error, size_t error_size);

/* Spawn the worker. `executor` must be non-NULL. Call exactly once. */
int h3_job_manager_start(h3_job_manager *manager, h3_job_executor executor,
                         void *ctx, char *error, size_t error_size);

/* Signal the worker to finish the current job (if any) and exit, then join
 * it. Queued-but-unstarted jobs are left QUEUED. Idempotent. */
void h3_job_manager_stop(h3_job_manager *manager);

/* Stop (if needed) and release everything, including every job record. */
void h3_job_manager_free(h3_job_manager *manager);

/* Enqueue a job. On success writes the new id into `id_out` (>= H3_JOB_ID_SIZE)
 * and returns 1. */
int h3_job_submit(h3_job_manager *manager, const h3_job_request *request,
                  char *id_out, size_t id_size, char *error, size_t error_size);

/* Copy the current state of job `id` into `*out`. Returns 1 if found. */
int h3_job_get(h3_job_manager *manager, const char *id, h3_job_info *out);

/* P10-CANCEL-01: cancel job `id`. A QUEUED job is marked CANCELLED
 * immediately, before it ever runs. A RUNNING job has its cancel flag set
 * and returns 1 right away -- the transition to CANCELLED happens
 * asynchronously once the generation code next checks in (the diffusion
 * loop's per-step boundary); poll h3_job_get() to observe it. Returns 0 with
 * `error` filled if the job is unknown or already in a terminal state
 * (SUCCEEDED / FAILED / CANCELLED). */
int h3_job_cancel(h3_job_manager *manager, const char *id, char *error,
                  size_t error_size);

const char *h3_job_status_name(h3_job_status status);

#endif
