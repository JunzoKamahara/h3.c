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

/* P10-REF2VA: the visual half of a Ref2VA reference attached to a video job.
 * An optional reference_audio_path (below) rides alongside it -- the
 * canonical model never accepts a reference audio without an image or video
 * reference, so H3_JOB_REF_NONE with a non-NULL reference_audio_path is
 * invalid, not "audio only". */
typedef enum {
    H3_JOB_REF_NONE = 0,
    H3_JOB_REF_IMAGE,
    H3_JOB_REF_VIDEO
} h3_job_reference_kind;

#define H3_JOB_ID_SIZE 24
#define H3_JOB_ERROR_SIZE 512
#define H3_JOB_PATH_SIZE 1024

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
    h3_job_reference_kind reference_kind; /* P10-REF2VA: NONE = plain T2VA */
    char *reference_path;   /* local file; meaningful iff reference_kind set */
    char *reference_audio_path; /* P10-REF2VA-04: optional, needs reference_kind set */

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
    /* P10-REF2VA: an optional local IMAGE or VIDEO reference file for Ref2VA
     * conditioning; H3_JOB_REF_NONE (the default) keeps the plain T2VA path.
     * `reference_audio_path` (P10-REF2VA-04) is an optional second, audio-only
     * reference that must accompany a non-NONE reference_kind -- the
     * canonical model never accepts a reference audio standalone. */
    h3_job_reference_kind reference_kind;
    const char *reference_path;
    const char *reference_audio_path;
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
