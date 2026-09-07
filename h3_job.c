#include "h3_job.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct h3_job_manager {
    char artifact_dir[H3_JOB_PATH_SIZE];

    pthread_mutex_t mu;
    pthread_cond_t cv;          /* signalled on submit and on stop */
    pthread_t worker;
    int worker_running;
    int stopping;

    h3_job_executor executor;
    void *executor_ctx;

    h3_job **jobs;
    size_t count;
    size_t capacity;
    size_t next_unstarted;     /* FIFO cursor */
    unsigned long id_counter;
};

const char *h3_job_status_name(h3_job_status status) {
    switch (status) {
        case H3_JOB_QUEUED: return "queued";
        case H3_JOB_RUNNING: return "running";
        case H3_JOB_SUCCEEDED: return "succeeded";
        case H3_JOB_FAILED: return "failed";
    }
    return "unknown";
}

static int64_t now_seconds(void) { return (int64_t)time(NULL); }

static const char *type_extension(h3_job_type type) {
    switch (type) {
        case H3_JOB_IMAGE: return "png";
        case H3_JOB_VIDEO: return "mp4";
        case H3_JOB_AUDIO: return "wav";
    }
    return "bin";
}

h3_job_manager *h3_job_manager_new(const char *artifact_dir, char *error,
                                   size_t error_size) {
    if (!artifact_dir || !*artifact_dir) {
        if (error && error_size)
            snprintf(error, error_size, "h3_job_manager_new needs an artifact "
                                        "directory");
        return NULL;
    }
    h3_job_manager *manager = calloc(1, sizeof(*manager));
    if (!manager) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    snprintf(manager->artifact_dir, sizeof(manager->artifact_dir), "%s",
             artifact_dir);
    pthread_mutex_init(&manager->mu, NULL);
    pthread_cond_init(&manager->cv, NULL);
    return manager;
}

/* Caller holds manager->mu. */
static h3_job *find_job_locked(h3_job_manager *manager, const char *id) {
    for (size_t i = 0; i < manager->count; i++)
        if (!strcmp(manager->jobs[i]->id, id)) return manager->jobs[i];
    return NULL;
}

static void *worker_main(void *opaque) {
    h3_job_manager *manager = opaque;
    pthread_mutex_lock(&manager->mu);
    for (;;) {
        while (!manager->stopping &&
               manager->next_unstarted == manager->count)
            pthread_cond_wait(&manager->cv, &manager->mu);
        if (manager->stopping) break;

        h3_job *job = manager->jobs[manager->next_unstarted++];
        job->status = H3_JOB_RUNNING;
        job->started_at = now_seconds();
        h3_job_executor executor = manager->executor;
        void *ctx = manager->executor_ctx;
        pthread_mutex_unlock(&manager->mu);

        int ok = executor ? executor(job, ctx) : 0;
        if (!executor)
            snprintf(job->error, sizeof(job->error), "no executor configured");

        pthread_mutex_lock(&manager->mu);
        job->status = ok ? H3_JOB_SUCCEEDED : H3_JOB_FAILED;
        job->finished_at = now_seconds();
        pthread_cond_broadcast(&manager->cv);
    }
    pthread_mutex_unlock(&manager->mu);
    return NULL;
}

int h3_job_manager_start(h3_job_manager *manager, h3_job_executor executor,
                         void *ctx, char *error, size_t error_size) {
    if (!manager || !executor) {
        if (error && error_size)
            snprintf(error, error_size, "h3_job_manager_start needs an "
                                        "executor");
        return 0;
    }
    pthread_mutex_lock(&manager->mu);
    if (manager->worker_running) {
        pthread_mutex_unlock(&manager->mu);
        if (error && error_size)
            snprintf(error, error_size, "job worker already started");
        return 0;
    }
    manager->executor = executor;
    manager->executor_ctx = ctx;
    manager->stopping = 0;
    if (pthread_create(&manager->worker, NULL, worker_main, manager) != 0) {
        pthread_mutex_unlock(&manager->mu);
        if (error && error_size)
            snprintf(error, error_size, "cannot start the job worker");
        return 0;
    }
    manager->worker_running = 1;
    pthread_mutex_unlock(&manager->mu);
    return 1;
}

void h3_job_manager_stop(h3_job_manager *manager) {
    if (!manager) return;
    pthread_mutex_lock(&manager->mu);
    if (!manager->worker_running) {
        pthread_mutex_unlock(&manager->mu);
        return;
    }
    manager->stopping = 1;
    pthread_cond_broadcast(&manager->cv);
    pthread_mutex_unlock(&manager->mu);
    pthread_join(manager->worker, NULL);
    pthread_mutex_lock(&manager->mu);
    manager->worker_running = 0;
    pthread_mutex_unlock(&manager->mu);
}

void h3_job_manager_free(h3_job_manager *manager) {
    if (!manager) return;
    h3_job_manager_stop(manager);
    for (size_t i = 0; i < manager->count; i++) {
        free(manager->jobs[i]->prompt);
        free(manager->jobs[i]);
    }
    free(manager->jobs);
    pthread_mutex_destroy(&manager->mu);
    pthread_cond_destroy(&manager->cv);
    free(manager);
}

int h3_job_submit(h3_job_manager *manager, const h3_job_request *request,
                  char *id_out, size_t id_size, char *error,
                  size_t error_size) {
    if (!manager || !request || !id_out || id_size < H3_JOB_ID_SIZE) {
        if (error && error_size)
            snprintf(error, error_size, "invalid job submission");
        return 0;
    }
    h3_job *job = calloc(1, sizeof(*job));
    if (!job) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    job->type = request->type;
    job->status = H3_JOB_QUEUED;
    job->prompt = strdup(request->prompt ? request->prompt : "");
    job->seed = request->seed;
    job->width = request->width;
    job->height = request->height;
    job->frames = request->frames;
    job->created_at = now_seconds();
    if (!job->prompt) {
        free(job);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }

    pthread_mutex_lock(&manager->mu);
    if (manager->count == manager->capacity) {
        size_t grown = manager->capacity ? manager->capacity * 2 : 8;
        h3_job **jobs = realloc(manager->jobs, grown * sizeof(*jobs));
        if (!jobs) {
            pthread_mutex_unlock(&manager->mu);
            free(job->prompt);
            free(job);
            if (error && error_size)
                snprintf(error, error_size, "out of memory");
            return 0;
        }
        manager->jobs = jobs;
        manager->capacity = grown;
    }
    snprintf(job->id, sizeof(job->id), "job-%08lx", ++manager->id_counter);
    snprintf(job->output_path, sizeof(job->output_path), "%s/%s.%s",
             manager->artifact_dir, job->id, type_extension(job->type));
    manager->jobs[manager->count++] = job;
    pthread_cond_broadcast(&manager->cv);
    snprintf(id_out, id_size, "%s", job->id);
    pthread_mutex_unlock(&manager->mu);
    return 1;
}

int h3_job_get(h3_job_manager *manager, const char *id, h3_job_info *out) {
    if (!manager || !id || !out) return 0;
    pthread_mutex_lock(&manager->mu);
    h3_job *job = find_job_locked(manager, id);
    if (job) {
        memcpy(out->id, job->id, sizeof(out->id));
        out->type = job->type;
        out->status = job->status;
        snprintf(out->output_path, sizeof(out->output_path), "%s",
                 job->output_path);
        snprintf(out->error, sizeof(out->error), "%s", job->error);
        out->created_at = job->created_at;
        out->started_at = job->started_at;
        out->finished_at = job->finished_at;
    }
    pthread_mutex_unlock(&manager->mu);
    return job != NULL;
}
