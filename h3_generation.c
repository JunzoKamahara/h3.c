#include "h3_generation.h"

#include "h3_ffmpeg.h"
#include "h3_gpu_sched.h"
#include "h3_host.h"
#include "h3_image_gen.h"
#include "h3_multimodal.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_vision_encoder.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Fixed serving-step count for this build (matches P8-IMG-01). */
#define H3_GENERATION_STEPS 12

/* P8-SCHED-01e1B: how often the keep-alive thread runs a one-token decode
 * while a job is in flight, to keep the shared chat weights out of the VM
 * compressor (an MTLResidencySet did not stop the compressor). 0 disables. */
#define H3_GENERATION_KEEPALIVE_MS 2000

struct h3_generation_engine {
    qwen_engine *language_engine;       /* borrowed */
    qwen_session *session;              /* owned -- separate KV / sampling state */
    qwen_session *keepalive_session;    /* owned -- tiny decodes, keeps weights hot */
    h3_tokenizer *tokenizer;           /* owned; FL2VA tokenizer (plain T2VA) */
    h3_tokenizer *ref2va_tokenizer;    /* owned; NULL unless ref2va_directory is set */
    pthread_mutex_t *conditioning_lock; /* borrowed; may be NULL */
    char *fl2va_directory;
    char *ref2va_directory;            /* NULL if that checkpoint isn't installed */
    char *shader_source_path;
    int steps;
    long keepalive_ms;
    _Atomic int keepalive_run;
    pthread_t keepalive_thread;
};

/* One throwaway one-token decode on the keep-alive session. Touches every
 * decoder layer's weights, so the shared resident set stays hot for the next
 * real chat prefill. Serialised against chat + conditioning through the lock;
 * the 01b scheduler makes the diffusion transformer yield to it. */
static void keepalive_tick(h3_generation_engine *engine) {
    char error[256];
    uint32_t token = 1; /* any valid id; content does not matter */
    if (engine->conditioning_lock) pthread_mutex_lock(engine->conditioning_lock);
    int ok = qwen_session_rewind(engine->keepalive_session, 0, error,
                                 sizeof(error));
    if (ok) {
        h3_gpu_sched_chat_enter();
        ok = qwen_session_eval(engine->keepalive_session, &token, 1, error,
                               sizeof(error));
        h3_gpu_sched_chat_leave();
    }
    if (engine->conditioning_lock)
        pthread_mutex_unlock(engine->conditioning_lock);
    (void)ok;
}

static void *keepalive_main(void *opaque) {
    h3_generation_engine *engine = opaque;
    while (atomic_load(&engine->keepalive_run)) {
        keepalive_tick(engine);
        long slept = 0;
        while (atomic_load(&engine->keepalive_run) &&
               slept < engine->keepalive_ms) {
            usleep(50000);
            slept += 50;
        }
    }
    return NULL;
}

static void keepalive_start(h3_generation_engine *engine) {
    if (engine->keepalive_ms <= 0 || !engine->keepalive_session) return;
    atomic_store(&engine->keepalive_run, 1);
    if (pthread_create(&engine->keepalive_thread, NULL, keepalive_main,
                       engine) != 0)
        atomic_store(&engine->keepalive_run, 0);
}

static void keepalive_stop(h3_generation_engine *engine) {
    if (!atomic_load(&engine->keepalive_run)) return;
    atomic_store(&engine->keepalive_run, 0);
    pthread_join(engine->keepalive_thread, NULL);
}

static char *join_path(const char *root, const char *suffix) {
    size_t n = strlen(root) + strlen(suffix) + 2;
    char *r = malloc(n);
    if (r) snprintf(r, n, "%s/%s", root, suffix);
    return r;
}

h3_generation_engine *h3_generation_engine_acquire(
        qwen_engine *language_engine, const char *fl2va_directory,
        const char *ref2va_directory,
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
    engine->ref2va_directory = ref2va_directory ? strdup(ref2va_directory) :
                                                  NULL;
    engine->shader_source_path = strdup(shader_source_path);
    engine->steps = H3_GENERATION_STEPS;

    char *tokenizer_path = join_path(fl2va_directory, "tokenizer/tokenizer.json");
    if (!engine->fl2va_directory ||
        (ref2va_directory && !engine->ref2va_directory) ||
        !engine->shader_source_path || !tokenizer_path) {
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
    if (engine->ref2va_directory) {
        char *ref2va_tokenizer_path = join_path(engine->ref2va_directory,
                                                "tokenizer/tokenizer.json");
        if (!ref2va_tokenizer_path) {
            h3_generation_engine_release(engine);
            if (error && error_size) snprintf(error, error_size, "out of memory");
            return NULL;
        }
        engine->ref2va_tokenizer = h3_tokenizer_load(ref2va_tokenizer_path,
                                                     error, error_size);
        free(ref2va_tokenizer_path);
        if (!engine->ref2va_tokenizer) {
            h3_generation_engine_release(engine);
            return NULL;
        }
    }
    if (!qwen_session_create(&engine->session, language_engine, error,
                             error_size)) {
        h3_generation_engine_release(engine);
        return NULL;
    }

    engine->keepalive_ms = H3_GENERATION_KEEPALIVE_MS;
    const char *ms = getenv("H3_GEN_KEEPALIVE_MS");
    if (ms) {
        long v = atol(ms);
        engine->keepalive_ms = (v >= 0 && v <= 60000) ? v : 0;
    }
    if (engine->keepalive_ms > 0 &&
        !qwen_session_create(&engine->keepalive_session, language_engine, error,
                             error_size)) {
        h3_generation_engine_release(engine);
        return NULL;
    }
    return engine;
}

void h3_generation_engine_release(h3_generation_engine *engine) {
    if (!engine) return;
    keepalive_stop(engine);
    if (engine->keepalive_session) qwen_session_free(engine->keepalive_session);
    if (engine->session) qwen_session_free(engine->session);
    if (engine->tokenizer) h3_tokenizer_free(engine->tokenizer);
    if (engine->ref2va_tokenizer) h3_tokenizer_free(engine->ref2va_tokenizer);
    free(engine->fl2va_directory);
    free(engine->ref2va_directory);
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

/* P10-REF2VA-02: Ref2VA conditioning for one IMAGE or VIDEO reference (audio
 * references are always paired with a visual reference in the canonical
 * model and are not supported standalone -- not added yet). Mirrors the
 * sequence h3.c's h3_generate() runs for a single reference: probe -> resolve
 * the reference canvas -> decode -> encode through BOTH the video VAE (DiT
 * condition rows) and the Qwen vision tower (the <Picture 1> / <Video 1>
 * presentation for the text pass -- a video becomes floor(frames/12) time
 * samples grouped into ceil(samples/2) two-frame blocks, one Qwen vision pass
 * each) -> the Ref2VA text conditioning. Serialised against chat through
 * conditioning_lock for the whole sequence, matching compute_conditioning();
 * a video reference's multiple vision passes make that critical section
 * longer than an image's, but there is exactly one reference, so it stays
 * bounded. On success `*layout_ref_out`, `*condition_video_rows_out` and
 * `*condition_video_elements_out` describe the packed condition the DiT
 * needs (see h3_video_condition in h3_image_gen.h); the caller owns the
 * returned rows and frees them with free(). */
static int compute_ref2va_conditioning(h3_generation_engine *engine,
                                       const char *prompt,
                                       h3_job_reference_kind kind,
                                       const char *reference_path,
                                       int target_width, int target_height,
                                       int target_frames,
                                       qwen_intermediate_state *text_out,
                                       h3_layout_ref *layout_ref_out,
                                       float **condition_video_rows_out,
                                       size_t *condition_video_elements_out,
                                       char *error, size_t error_size) {
    memset(text_out, 0, sizeof(*text_out));
    memset(layout_ref_out, 0, sizeof(*layout_ref_out));
    *condition_video_rows_out = NULL;
    *condition_video_elements_out = 0;

    if (!engine->ref2va_directory) {
        snprintf(error, error_size,
                "this server has no Ref2VA transformer checkpoint installed");
        return 0;
    }
    int is_video = kind == H3_JOB_REF_VIDEO;

    int source_width = 0, source_height = 0;
    if (!h3_ffprobe_visual_size(reference_path, &source_width, &source_height,
                                error, error_size))
        return 0;
    int media_width = 0, media_height = 0;
    if (is_video) {
        if (!h3_reference_video_canvas(source_width, source_height,
                                       &media_width, &media_height)) {
            snprintf(error, error_size,
                    "cannot resolve reference video canvas");
            return 0;
        }
    } else if (!h3_reference_image_canvas(source_width, source_height,
                                          target_width, target_height, 0,
                                          &media_width, &media_height)) {
        snprintf(error, error_size, "cannot resolve reference image canvas");
        return 0;
    }

    float *pixels = NULL;
    int ref_frames = 1;
    if (is_video) {
        int max_frames = h3_temporal(target_frames).frame_count;
        if (!h3_ffmpeg_read_video_f32(reference_path, media_width,
                                      media_height, max_frames, &pixels,
                                      &ref_frames, error, error_size))
            return 0;
    } else if (!h3_ffmpeg_read_image_f32(reference_path, media_width,
                                         media_height, H3_IMAGE_FIT_STRETCH,
                                         &pixels, error, error_size)) {
        return 0;
    }

    char *text_dir = join_path(engine->ref2va_directory, "text_encoder");
    char *vae_dir = join_path(engine->ref2va_directory, "video_vae/source");
    if (!text_dir || !vae_dir) {
        free(pixels);
        free(text_dir);
        free(vae_dir);
        snprintf(error, error_size, "out of memory");
        return 0;
    }

    if (engine->conditioning_lock)
        pthread_mutex_lock(engine->conditioning_lock);

    h3_video_latent latent = {0};
    int ok = h3_video_vae_encode(vae_dir, engine->shader_source_path, pixels,
                                 ref_frames, media_height, media_width, NULL,
                                 NULL, &latent, error, error_size);
    int ref_latent_w = 0, ref_latent_h = 0, ref_latent_t = 0;
    float *rows = NULL;
    size_t row_elements = 0;
    if (ok) {
        h3_latent_canvas(media_width, media_height, &ref_latent_w,
                         &ref_latent_h);
        ref_latent_t = h3_video_encoder_latent_t(ref_frames);
        if (latent.time != ref_latent_t || latent.height != ref_latent_h ||
            latent.width != ref_latent_w) {
            snprintf(error, error_size,
                    "reference VAE produced unexpected latent geometry");
            ok = 0;
        }
    }
    if (ok) {
        row_elements = (size_t)ref_latent_t * (size_t)ref_latent_h *
                      (size_t)ref_latent_w / 4 * 96;
        rows = malloc(row_elements * sizeof(*rows));
        if (!rows) {
            snprintf(error, error_size, "out of memory");
            ok = 0;
        } else {
            ok = h3_dit_patchify_video(latent.values, 24, ref_latent_t,
                                       ref_latent_h, ref_latent_w, rows,
                                       row_elements);
            if (!ok)
                snprintf(error, error_size,
                        "cannot patchify reference condition");
        }
    }
    h3_video_latent_free(&latent);

    /* Qwen sees an image as one frame; a video as blocks of two frames each
     * (the released cadence: floor(frames/12) samples, ceil(samples/2)
     * blocks), one timestamp per block. */
    size_t blocks = 1;
    double *timestamps = NULL;
    if (is_video && ok) {
        size_t samples = ((size_t)ref_frames + 11) / 12;
        blocks = (samples + 1) / 2;
        if (blocks < 1) blocks = 1;
        timestamps = malloc(blocks * sizeof(*timestamps));
        if (!timestamps) {
            snprintf(error, error_size, "out of memory");
            ok = 0;
        } else {
            for (size_t block = 0; block < blocks; block++) {
                size_t first = 2 * block;
                size_t second = first + 1 < samples ? first + 1 : first;
                timestamps[block] = ((double)first + (double)second) / 4.0;
            }
        }
    }

    h3_vision_output *vision = ok ? calloc(blocks, sizeof(*vision)) : NULL;
    size_t vision_count = 0;
    if (ok && !vision) {
        snprintf(error, error_size, "out of memory");
        ok = 0;
    }
    if (ok && !is_video) {
        ok = h3_vision_encode_bf16(text_dir, engine->shader_source_path,
                                   pixels, 1, media_height, media_width, NULL,
                                   NULL, &vision[0], error, error_size);
        if (ok) vision_count = 1;
    } else if (ok) {
        size_t samples = ((size_t)ref_frames + 11) / 12;
        for (size_t block = 0; block < blocks && ok; block++) {
            size_t first_sample = 2 * block;
            size_t second_sample = first_sample + 1 < samples ?
                                   first_sample + 1 : first_sample;
            int first = (int)(first_sample * 12);
            int second = (int)(second_sample * 12);
            float *pair = h3_extract_vision_pair(pixels, ref_frames,
                                                 media_height, media_width,
                                                 first, second);
            if (!pair) {
                snprintf(error, error_size,
                        "out of memory extracting a reference video block");
                ok = 0;
                break;
            }
            ok = h3_vision_encode_bf16(text_dir, engine->shader_source_path,
                                       pair, 2, media_height, media_width,
                                       NULL, NULL, &vision[block], error,
                                       error_size);
            free(pair);
            if (ok) vision_count = block + 1;
        }
    }

    h3_text_embedding text = {0};
    int have_text = 0;
    if (ok) {
        h3_reference_presentation presentation = {0};
        presentation.kind = is_video ? H3_PRESENTATION_VIDEO :
                                       H3_PRESENTATION_IMAGE;
        presentation.vision = vision;
        presentation.vision_count = blocks;
        presentation.timestamps = timestamps;
        ok = h3_multimodal_encode_ref2va_bf16(
            engine->ref2va_tokenizer, text_dir, engine->shader_source_path,
            prompt, &presentation, 1, NULL, NULL, &text, error, error_size);
        have_text = ok;
    }

    if (engine->conditioning_lock)
        pthread_mutex_unlock(engine->conditioning_lock);

    free(pixels);
    free(text_dir);
    free(vae_dir);
    free(timestamps);
    for (size_t index = 0; index < vision_count; index++)
        h3_vision_output_free(&vision[index]);
    free(vision);

    if (!ok) {
        free(rows);
        if (have_text) h3_text_embedding_free(&text);
        return 0;
    }

    text_out->tokens = text.tokens;
    text_out->hidden_size = text.width;
    text_out->values = text.values;
    text_out->tags = text.tags;
    text_out->policy = QWEN_EXEC_BF16_CANONICAL;
    if (!h3_conditioning_accepts(text_out)) {
        snprintf(error, error_size,
                "Ref2VA conditioning is not BF16-canonical");
        qwen_intermediate_state_free(text_out);
        free(rows);
        return 0;
    }

    *layout_ref_out = (h3_layout_ref){
        is_video ? H3_LAYOUT_REF_VIDEO : H3_LAYOUT_REF_IMAGE, ref_latent_t,
        ref_latent_h, ref_latent_w, 0};
    *condition_video_rows_out = rows;
    *condition_video_elements_out = row_elements;
    return 1;
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
    int width = request->width > 0 ? request->width : 256;
    int height = request->height > 0 ? request->height : 256;
    int ref2va = request->reference_kind != H3_JOB_REF_NONE &&
                request->reference_path && request->reference_path[0];
    /* Matches h3_generate()'s own floor for a conditioned run (one trained
     * 22-frame decoder chunk); plain T2VA keeps its proven 5-frame floor
     * (P8-IMG-01 / P8-VID-01), unvalidated at 22 frames here. */
    if (ref2va && h3_align_frame_count(request->frames) < 22) {
        if (error && error_size)
            snprintf(error, error_size,
                    "Ref2VA generation requires at least a 22-frame clip");
        return 0;
    }

    qwen_intermediate_state cond = {0};
    h3_layout_ref layout_ref = {0};
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
    double conditioning_start = now_seconds();
    int cond_ok = ref2va ?
        compute_ref2va_conditioning(
            engine, request->prompt, request->reference_kind,
            request->reference_path, width, height, request->frames, &cond,
            &layout_ref, &condition_video_rows, &condition_video_elements,
            error, error_size) :
        compute_conditioning(engine, request->prompt, &cond, error,
                             error_size);
    if (!cond_ok) return 0;
    if (conditioning_seconds)
        *conditioning_seconds = now_seconds() - conditioning_start;

    h3_video_condition condition = {0};
    if (ref2va) {
        condition.release_directory = engine->ref2va_directory;
        condition.layout_references = &layout_ref;
        condition.reference_count = 1;
        condition.condition_video_rows = condition_video_rows;
        condition.condition_video_elements = condition_video_elements;
    }

    h3_video_request req = {0};
    req.fl2va_directory = engine->fl2va_directory;
    req.shader_source_path = engine->shader_source_path;
    req.conditioning = cond.values;
    req.conditioning_tokens = cond.tokens;
    req.width = width;
    req.height = height;
    req.frames = request->frames;
    req.steps = engine->steps;
    req.seed = request->seed;
    req.output_path = output_path;
    req.condition = ref2va ? &condition : NULL;
    keepalive_start(engine);
    int ok = h3_video_generate(&req, timing, progress, progress_opaque, error,
                               error_size);
    keepalive_stop(engine);
    qwen_intermediate_state_free(&cond);
    free(condition_video_rows);
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
                                  job->width, job->height, job->frames,
                                  job->reference_kind, job->reference_path};
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
