#ifndef H3_GENERATION_H
#define H3_GENERATION_H

#include "h3_job.h"
#include "qwen_engine.h"

#include <pthread.h>
#include <stddef.h>

/* P8-VID-01: a stateful media-generation context.
 *
 * It shares the immutable Qwen weights with the chat engine (same qwen_engine)
 * but keeps its OWN qwen_session, so the background job worker never touches
 * the chat request's KV / sampling state. Layers 0..49 (conditioning) run on
 * the shared resident weights, so they are briefly serialised against chat
 * evals through `conditioning_lock`; the diffusion transformer and the VAEs
 * run on their own Metal contexts with no shared mutable state.
 *
 * The 62 GB transformer is still loaded per generation (SSD streaming). The
 * acquire / release seam is where a resident cache goes later (P8-GEN-CACHE)
 * without changing callers. */

typedef struct h3_generation_engine h3_generation_engine;

/* `language_engine` is the shared Qwen engine (borrowed, not owned).
 * `fl2va_directory` is the release ".../FL2VA" directory. `conditioning_lock`
 * (borrowed) is held only around the layers-0..49 forward; pass the same lock
 * the chat path serialises on. */
h3_generation_engine *h3_generation_engine_acquire(
        qwen_engine *language_engine, const char *fl2va_directory,
        const char *shader_source_path, pthread_mutex_t *conditioning_lock,
        char *error, size_t error_size);

void h3_generation_engine_release(h3_generation_engine *engine);

/* Run one job to completion: compute conditioning from job->prompt, then
 * generate, writing the artifact to job->output_path. Returns 1 on success;
 * on failure fills job->error and returns 0. Safe to use as an
 * h3_job_executor with the engine pointer as ctx. */
int h3_generation_run_job(h3_job *job, void *engine);

#endif
