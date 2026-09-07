#include "h3_generation.h"

#include "h3_ffmpeg.h"
#include "h3_image_gen.h"
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int h3_generation_run_job(h3_job *job, void *engine_ptr) {
    h3_generation_engine *engine = engine_ptr;
    if (!engine || !job) return 0;
    char *error = job->error;
    size_t error_size = sizeof(job->error);
    error[0] = '\0';

    int w = job->width > 0 ? job->width : 256;
    int h = job->height > 0 ? job->height : 256;

    qwen_intermediate_state cond = {0};
    if (!compute_conditioning(engine, job->prompt, &cond, error, error_size))
        return 0;

    int ok;
    if (job->type == H3_JOB_VIDEO) {
        h3_video_request req = {0};
        req.fl2va_directory = engine->fl2va_directory;
        req.shader_source_path = engine->shader_source_path;
        req.conditioning = cond.values;
        req.conditioning_tokens = cond.tokens;
        req.width = w;
        req.height = h;
        req.frames = job->frames;
        req.steps = engine->steps;
        req.seed = job->seed;
        req.output_path = job->output_path;
        ok = h3_video_generate(&req, NULL, NULL, error, error_size);
    } else if (job->type == H3_JOB_IMAGE) {
        h3_image_request req = {0};
        req.fl2va_directory = engine->fl2va_directory;
        req.shader_source_path = engine->shader_source_path;
        req.conditioning = cond.values;
        req.conditioning_tokens = cond.tokens;
        req.width = w;
        req.height = h;
        req.steps = engine->steps;
        req.seed = job->seed;
        uint8_t *rgb = NULL;
        int gw = 0, gh = 0;
        ok = h3_image_generate(&req, &rgb, &gw, &gh, NULL, NULL, error,
                               error_size) &&
             h3_ffmpeg_write_png_rgb24(job->output_path, rgb, gw, gh, error,
                                       error_size);
        free(rgb);
    } else {
        snprintf(error, error_size, "audio generation is not implemented yet");
        ok = 0;
    }

    qwen_intermediate_state_free(&cond);
    return ok;
}
