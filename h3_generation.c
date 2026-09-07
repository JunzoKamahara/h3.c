#include "h3_generation.h"

#include "h3_ffmpeg.h"
#include "h3_image_gen.h"
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Fixed serving-step count for this build (matches P8-IMG-01). */
#define H3_GENERATION_STEPS 12

struct h3_generation_engine {
    qwen_engine *language_engine;       /* borrowed */
    qwen_session *session;              /* owned -- separate KV / sampling state */
    h3_tokenizer *tokenizer;           /* owned */
    pthread_mutex_t *conditioning_lock; /* borrowed; may be NULL */
    char *fl2va_directory;
    char *shader_source_path;
    int steps;
};

static char *join_path(const char *root, const char *suffix) {
    size_t n = strlen(root) + strlen(suffix) + 2;
    char *r = malloc(n);
    if (r) snprintf(r, n, "%s/%s", root, suffix);
    return r;
}

h3_generation_engine *h3_generation_engine_acquire(
        qwen_engine *language_engine, const char *fl2va_directory,
        const char *shader_source_path, pthread_mutex_t *conditioning_lock,
        char *error, size_t error_size) {
    if (!language_engine || !fl2va_directory || !shader_source_path) {
        if (error && error_size)
            snprintf(error, error_size, "invalid generation-engine request");
        return NULL;
    }
    h3_generation_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    engine->language_engine = language_engine;
    engine->conditioning_lock = conditioning_lock;
    engine->fl2va_directory = strdup(fl2va_directory);
    engine->shader_source_path = strdup(shader_source_path);
    engine->steps = H3_GENERATION_STEPS;

    char *tokenizer_path = join_path(fl2va_directory, "tokenizer/tokenizer.json");
    if (!engine->fl2va_directory || !engine->shader_source_path ||
        !tokenizer_path) {
        free(tokenizer_path);
        h3_generation_engine_release(engine);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    engine->tokenizer = h3_tokenizer_load(tokenizer_path, error, error_size);
    free(tokenizer_path);
    if (!engine->tokenizer) {
        h3_generation_engine_release(engine);
        return NULL;
    }
    if (!qwen_session_create(&engine->session, language_engine, error,
                             error_size)) {
        h3_generation_engine_release(engine);
        return NULL;
    }
    return engine;
}

void h3_generation_engine_release(h3_generation_engine *engine) {
    if (!engine) return;
    if (engine->session) qwen_session_free(engine->session);
    if (engine->tokenizer) h3_tokenizer_free(engine->tokenizer);
    free(engine->fl2va_directory);
    free(engine->shader_source_path);
    free(engine);
}

/* Layers 0..49 for `prompt`, serialised against the chat path. */
static int compute_conditioning(h3_generation_engine *engine, const char *prompt,
                                qwen_intermediate_state *out, char *error,
                                size_t error_size) {
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(engine->tokenizer,
                             prompt && prompt[0] ? prompt : " ", 1, &ids,
                             &count, error, error_size))
        return 0;

    qwen_input in = {0};
    in.token_ids = ids;
    in.token_count = count;

    if (engine->conditioning_lock)
        pthread_mutex_lock(engine->conditioning_lock);
    int ok = qwen_session_get_h3_conditioning(engine->session, &in, out, NULL,
                                              NULL, error, error_size);
    if (engine->conditioning_lock)
        pthread_mutex_unlock(engine->conditioning_lock);

    h3_tokenizer_ids_free(ids);
    if (ok && !h3_conditioning_accepts(out)) {
        snprintf(error, error_size, "conditioning is not BF16-canonical");
        qwen_intermediate_state_free(out);
        return 0;
    }
    return ok;
}

int h3_generation_generate_video(h3_generation_engine *engine,
                                 const h3_job_request *request,
                                 const char *output_path,
                                 double *conditioning_seconds,
                                 h3_video_timing *timing,
                                 h3_dit_progress progress, void *progress_opaque,
                                 char *error, size_t error_size) {
    if (conditioning_seconds) *conditioning_seconds = 0.0;
    if (!engine || !request || !output_path || !output_path[0]) {
        if (error && error_size)
            snprintf(error, error_size, "invalid video generation request");
        return 0;
    }
    qwen_intermediate_state cond = {0};
    double conditioning_start = now_seconds();
    if (!compute_conditioning(engine, request->prompt, &cond, error, error_size))
        return 0;
    if (conditioning_seconds)
        *conditioning_seconds = now_seconds() - conditioning_start;

    h3_video_request req = {0};
    req.fl2va_directory = engine->fl2va_directory;
    req.shader_source_path = engine->shader_source_path;
    req.conditioning = cond.values;
    req.conditioning_tokens = cond.tokens;
    req.width = request->width > 0 ? request->width : 256;
    req.height = request->height > 0 ? request->height : 256;
    req.frames = request->frames;
    req.steps = engine->steps;
    req.seed = request->seed;
    req.output_path = output_path;
    int ok = h3_video_generate(&req, timing, progress, progress_opaque, error,
                               error_size);
    qwen_intermediate_state_free(&cond);
    return ok;
}

int h3_generation_run_job(h3_job *job, void *engine_ptr) {
    h3_generation_engine *engine = engine_ptr;
    if (!engine || !job) return 0;
    char *error = job->error;
    size_t error_size = sizeof(job->error);
    error[0] = '\0';

    if (job->type == H3_JOB_VIDEO) {
        h3_job_request request = {job->type, job->prompt, job->seed,
                                  job->width, job->height, job->frames};
        return h3_generation_generate_video(engine, &request, job->output_path,
                                            NULL, NULL, NULL, NULL, error,
                                            error_size);
    }
    if (job->type == H3_JOB_IMAGE) {
        qwen_intermediate_state cond = {0};
        if (!compute_conditioning(engine, job->prompt, &cond, error, error_size))
            return 0;
        h3_image_request req = {0};
        req.fl2va_directory = engine->fl2va_directory;
        req.shader_source_path = engine->shader_source_path;
        req.conditioning = cond.values;
        req.conditioning_tokens = cond.tokens;
        req.width = job->width > 0 ? job->width : 256;
        req.height = job->height > 0 ? job->height : 256;
        req.steps = engine->steps;
        req.seed = job->seed;
        uint8_t *rgb = NULL;
        int gw = 0, gh = 0;
        int ok = h3_image_generate(&req, &rgb, &gw, &gh, NULL, NULL, error,
                                   error_size) &&
                 h3_ffmpeg_write_png_rgb24(job->output_path, rgb, gw, gh, error,
                                           error_size);
        free(rgb);
        qwen_intermediate_state_free(&cond);
        return ok;
    }
    snprintf(error, error_size, "audio generation is not implemented yet");
    return 0;
}
