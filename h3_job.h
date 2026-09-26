#ifndef H3_JOB_H
#define H3_JOB_H

#include <stddef.h>
#include <stdint.h>

/* P8-VID-01: a process-local, non-persistent FIFO job manager for media
 * generation. One background worker runs at most one job at a time in
 * submission order. No progress, no cancel, no priority, no persistence yet.
 *
 * The manager knows nothing about diffusion or the language model: the actual
 * work is an injected executor callback (`h3_job_executor`). Tests inject a
 * fast mock; the server injects the real generation path. */

typedef enum {
    H3_JOB_QUEUED,
    H3_JOB_RUNNING,
    H3_JOB_SUCCEEDED,
    H3_JOB_FAILED
} h3_job_status;

typedef enum {
    H3_JOB_IMAGE,
    H3_JOB_VIDEO,
    H3_JOB_AUDIO
} h3_job_type;

/* P10-REF2VA: a Ref2VA reference attached to a video job. Audio references
 * are always paired with an image/video reference in the canonical model
 * and are not supported standalone -- not added yet. */
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
    h3_job_reference_kind reference_kind; /* P10-REF2VA: NONE = plain T2VA */
    char *reference_path;   /* local file; meaningful iff reference_kind set */

    char output_path[H3_JOB_PATH_SIZE];
    char error[H3_JOB_ERROR_SIZE];

    int64_t created_at;     /* unix seconds */
    int64_t started_at;     /* 0 until RUNNING */
    int64_t finished_at;    /* 0 until SUCCEEDED / FAILED */
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
    /* P10-REF2VA: an optional local IMAGE or VIDEO reference file for Ref2VA
     * conditioning; H3_JOB_REF_NONE (the default) keeps the plain T2VA path.
     * Audio references are not accepted standalone yet. */
    h3_job_reference_kind reference_kind;
    const char *reference_path;
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

const char *h3_job_status_name(h3_job_status status);

#endif
