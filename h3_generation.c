#include "h3_generation.h"

#include "h3_audio_vae.h"
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

/* P10-MULTIREF-01: the packed conditioning a Ref2VA job needs -- one layout
 * reference per input reference (image/video get real VAE-latent geometry;
 * audio gets zeroed visual geometry + its own audio_t), in the same order
 * the references were given, plus the two packed condition-row arrays
 * h3_video_condition (h3_image_gen.h) wants directly. Free with
 * h3_ref2va_condition_free(). */
typedef struct {
    h3_layout_ref *refs; /* one per reference, in order; owned */
    size_t ref_count;
    float *video_rows;
    size_t video_elements;
    float *audio_rows;
    size_t audio_elements;
} h3_ref2va_condition;

static void h3_ref2va_condition_free(h3_ref2va_condition *condition) {
    free(condition->refs);
    free(condition->video_rows);
    free(condition->audio_rows);
    memset(condition, 0, sizeof(*condition));
}

/* Per-reference bookkeeping compute_ref2va_conditioning() needs between its
 * encode pass and the final h3_reference_presentation array it builds --
 * kept separate from h3_layout_ref (written once in place, no aliasing
 * concern) because a presentation's `vision` pointer must point into the
 * FINAL, no-longer-growing vision-output array, which is only stable once
 * every reference has been encoded (the array grows via realloc as each
 * reference's outputs are appended). */
typedef struct {
    size_t vision_cursor;
    size_t vision_count;
    double *timestamps; /* owned; video only */
} h3_ref2va_ref_meta;

/* P10-MULTIREF-01: Ref2VA conditioning for an ordered list of up to
 * H3_JOB_MAX_REFERENCES references (any mix of IMAGE/VIDEO/AUDIO, capped at
 * 9 image / 3 video / 3 audio and a combined 15s across every audio input --
 * h3.c's own h3_valid_params()). Mirrors h3_generate()'s own two-pass
 * structure for the reason noted on h3_ref2va_ref_meta above: every vision
 * output is produced (and the array holding them stops growing) before any
 * h3_reference_presentation is built from it. For each visual reference:
 * probe -> resolve the reference canvas -> decode -> encode through BOTH the
 * video VAE (DiT condition rows) and the Qwen vision tower (the <Picture N>
 * / <Video N> presentation for the text pass -- a video becomes
 * floor(frames/12) time samples grouped into ceil(samples/2) two-frame
 * blocks, one Qwen vision pass each). For each audio reference: decode (15s
 * cap, combined across every audio reference) -> audio VAE encode -> its own
 * <Audio N> presentation entry, not attached to any visual one -> the Ref2VA
 * text conditioning. Serialised against chat through conditioning_lock for
 * the whole sequence, matching compute_conditioning(). On success
 * `*condition_out` describes the packed condition the DiT needs (see
 * h3_video_condition in h3_image_gen.h); the caller owns it and frees it
 * with h3_ref2va_condition_free(). */
static int compute_ref2va_conditioning(
        h3_generation_engine *engine, const char *prompt,
        const h3_job_reference_request *references, size_t reference_count,
        int target_width, int target_height, int target_frames,
        qwen_intermediate_state *text_out, h3_ref2va_condition *condition_out,
        char *error, size_t error_size) {
    memset(text_out, 0, sizeof(*text_out));
    memset(condition_out, 0, sizeof(*condition_out));

    if (!engine->ref2va_directory) {
        snprintf(error, error_size,
                "this server has no Ref2VA transformer checkpoint installed");
        return 0;
    }
    if (reference_count > H3_JOB_MAX_REFERENCES) {
        snprintf(error, error_size,
                "at most %d ordered references are supported",
                H3_JOB_MAX_REFERENCES);
        return 0;
    }
    size_t images = 0, videos = 0, audio_refs = 0;
    for (size_t i = 0; i < reference_count; i++) {
        if (!references[i].path || !references[i].path[0]) {
            snprintf(error, error_size, "reference %zu has no input path",
                    i + 1);
            return 0;
        }
        switch (references[i].kind) {
        case H3_JOB_REF_IMAGE: images++; break;
        case H3_JOB_REF_VIDEO: videos++; break;
        case H3_JOB_REF_AUDIO: audio_refs++; break;
        }
    }
    if (images > 9 || videos > 3 || audio_refs > 3) {
        snprintf(error, error_size,
                "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    if (reference_count && !(images + videos)) {
        snprintf(error, error_size,
                "a reference audio requires an image or video reference");
        return 0;
    }

    char *text_dir = join_path(engine->ref2va_directory, "text_encoder");
    char *vae_dir = join_path(engine->ref2va_directory, "video_vae/source");
    char *audio_vae_dir = join_path(engine->ref2va_directory, "audio_vae");
    if (!text_dir || !vae_dir || !audio_vae_dir) {
        free(text_dir);
        free(vae_dir);
        free(audio_vae_dir);
        snprintf(error, error_size, "out of memory");
        return 0;
    }

    size_t alloc_count = reference_count ? reference_count : 1;
    h3_layout_ref *layout_refs = calloc(alloc_count, sizeof(*layout_refs));
    h3_ref2va_ref_meta *meta = calloc(alloc_count, sizeof(*meta));
    float *video_rows = NULL;
    size_t video_elements = 0;
    float *audio_rows = NULL;
    size_t audio_elements = 0;
    h3_vision_output *vision = NULL;
    size_t vision_count = 0;
    size_t total_audio_samples = 0;
    int ok = layout_refs && meta;
    if (!ok) snprintf(error, error_size, "out of memory");

    if (engine->conditioning_lock)
        pthread_mutex_lock(engine->conditioning_lock);

    /* Pass 1: every visual (image/video) reference, in order. */
    for (size_t i = 0; ok && i < reference_count; i++) {
        if (references[i].kind == H3_JOB_REF_AUDIO) continue;
        int is_video = references[i].kind == H3_JOB_REF_VIDEO;
        const char *path = references[i].path;

        int source_width = 0, source_height = 0;
        ok = h3_ffprobe_visual_size(path, &source_width, &source_height,
                                    error, error_size);
        int media_width = 0, media_height = 0;
        if (ok) {
            ok = is_video ?
                h3_reference_video_canvas(source_width, source_height,
                                          &media_width, &media_height) :
                h3_reference_image_canvas(source_width, source_height,
                                          target_width, target_height, 0,
                                          &media_width, &media_height);
            if (!ok)
                snprintf(error, error_size,
                        "cannot resolve reference %zu canvas", i + 1);
        }
        float *pixels = NULL;
        int ref_frames = 1;
        if (ok) {
            if (is_video) {
                int max_frames = h3_temporal(target_frames).frame_count;
                ok = h3_ffmpeg_read_video_f32(path, media_width, media_height,
                                              max_frames, &pixels, &ref_frames,
                                              error, error_size);
            } else {
                ok = h3_ffmpeg_read_image_f32(path, media_width, media_height,
                                              H3_IMAGE_FIT_STRETCH, &pixels,
                                              error, error_size);
            }
        }
        h3_video_latent latent = {0};
        int ref_latent_w = 0, ref_latent_h = 0, ref_latent_t = 0;
        if (ok)
            ok = h3_video_vae_encode(vae_dir, engine->shader_source_path,
                                     pixels, ref_frames, media_height,
                                     media_width, NULL, NULL, &latent, error,
                                     error_size);
        if (ok) {
            h3_latent_canvas(media_width, media_height, &ref_latent_w,
                             &ref_latent_h);
            ref_latent_t = h3_video_encoder_latent_t(ref_frames);
            if (latent.time != ref_latent_t || latent.height != ref_latent_h ||
                latent.width != ref_latent_w) {
                snprintf(error, error_size,
                        "reference %zu VAE produced unexpected latent geometry",
                        i + 1);
                ok = 0;
            }
        }
        size_t row_elements = 0;
        if (ok) {
            row_elements = (size_t)ref_latent_t * (size_t)ref_latent_h *
                          (size_t)ref_latent_w / 4 * 96;
            float *grown = realloc(
                video_rows, (video_elements + row_elements) * sizeof(*grown));
            ok = grown != NULL;
            if (ok) {
                video_rows = grown;
                ok = h3_dit_patchify_video(latent.values, 24, ref_latent_t,
                                          ref_latent_h, ref_latent_w,
                                          video_rows + video_elements,
                                          row_elements);
                if (!ok)
                    snprintf(error, error_size,
                            "cannot patchify reference %zu condition", i + 1);
            } else {
                snprintf(error, error_size, "out of memory");
            }
        }
        h3_video_latent_free(&latent);
        if (ok) video_elements += row_elements;

        /* Qwen sees an image as one frame; a video as blocks of two frames
         * each (the released cadence: floor(frames/12) samples,
         * ceil(samples/2) blocks), one timestamp per block. */
        size_t blocks = 1;
        double *timestamps = NULL;
        if (ok && is_video) {
            size_t samples = ((size_t)ref_frames + 11) / 12;
            blocks = (samples + 1) / 2;
            if (blocks < 1) blocks = 1;
            timestamps = malloc(blocks * sizeof(*timestamps));
            ok = timestamps != NULL;
            if (ok) {
                for (size_t block = 0; block < blocks; block++) {
                    size_t first = 2 * block;
                    size_t second = first + 1 < samples ? first + 1 : first;
                    timestamps[block] = ((double)first + (double)second) / 4.0;
                }
            } else {
                snprintf(error, error_size, "out of memory");
            }
        }

        if (ok) {
            h3_vision_output *grown = realloc(
                vision, (vision_count + blocks) * sizeof(*grown));
            ok = grown != NULL;
            if (ok) vision = grown;
            else snprintf(error, error_size, "out of memory");
        }
        if (ok && !is_video) {
            ok = h3_vision_encode_bf16(text_dir, engine->shader_source_path,
                                       pixels, 1, media_height, media_width,
                                       NULL, NULL, &vision[vision_count],
                                       error, error_size);
            if (ok) vision_count++;
        } else if (ok) {
            size_t samples = ((size_t)ref_frames + 11) / 12;
            for (size_t block = 0; block < blocks && ok; block++) {
                size_t first_sample = 2 * block;
                size_t second_sample = first_sample + 1 < samples ?
                                       first_sample + 1 : first_sample;
                int first = (int)(first_sample * 12);
                int second = (int)(second_sample * 12);
                float *pair = h3_extract_vision_pair(
                    pixels, ref_frames, media_height, media_width, first,
                    second);
                if (!pair) {
                    snprintf(error, error_size,
                            "out of memory extracting reference %zu video "
                            "block", i + 1);
                    ok = 0;
                    break;
                }
                ok = h3_vision_encode_bf16(
                    text_dir, engine->shader_source_path, pair, 2,
                    media_height, media_width, NULL, NULL,
                    &vision[vision_count], error, error_size);
                free(pair);
                if (ok) vision_count++;
            }
        }
        free(pixels);

        if (ok) {
            meta[i].vision_cursor = vision_count - blocks;
            meta[i].vision_count = blocks;
            meta[i].timestamps = is_video ? timestamps : NULL;
            layout_refs[i] = (h3_layout_ref){
                is_video ? H3_LAYOUT_REF_VIDEO : H3_LAYOUT_REF_IMAGE,
                ref_latent_t, ref_latent_h, ref_latent_w, 0};
        } else {
            free(timestamps);
        }
    }

    /* Pass 2: every audio reference, in order. Standalone only (no video's
     * own embedded soundtrack in this build), 15s cap per read, combined
     * across every audio reference -- matches h3.c's own "ordered reference
     * audio exceeds 15 seconds in total" rule. */
    for (size_t i = 0; ok && i < reference_count; i++) {
        if (references[i].kind != H3_JOB_REF_AUDIO) continue;
        float *pcm = NULL;
        int samples = 0;
        ok = h3_ffmpeg_read_audio_f32(references[i].path, 32000 * 15, 0, &pcm,
                                      &samples, error, error_size);
        if (ok && (size_t)samples > (size_t)32000 * 15 - total_audio_samples) {
            free(pcm);
            snprintf(error, error_size,
                    "ordered reference audio exceeds 15 seconds in total");
            ok = 0;
        }
        h3_audio_latent latent = {0};
        if (ok) {
            ok = h3_audio_vae_encode(audio_vae_dir, engine->shader_source_path,
                                     pcm, samples, NULL, NULL, &latent, error,
                                     error_size);
            free(pcm);
        }
        if (ok && (latent.channels != 32 || latent.stereo != 2 ||
                  latent.length < 1)) {
            snprintf(error, error_size,
                    "reference %zu audio VAE produced unexpected geometry",
                    i + 1);
            ok = 0;
        }
        size_t elements = 0;
        if (ok) {
            elements = (size_t)latent.length * 2 * 32;
            float *grown = realloc(
                audio_rows, (audio_elements + elements) * sizeof(*grown));
            ok = grown != NULL;
            if (ok) {
                audio_rows = grown;
                float *rows = audio_rows + audio_elements;
                for (int stereo = 0; stereo < 2; stereo++)
                    for (int time = 0; time < latent.length; time++)
                        for (int channel = 0; channel < 32; channel++) {
                            size_t source = ((size_t)channel * 2 +
                                             (size_t)stereo) *
                                            (size_t)latent.length +
                                            (size_t)time;
                            size_t destination =
                                ((size_t)stereo * (size_t)latent.length +
                                 (size_t)time) * 32 + (size_t)channel;
                            rows[destination] = latent.values[source];
                        }
            } else {
                snprintf(error, error_size, "out of memory");
            }
        }
        if (ok) {
            audio_elements += elements;
            total_audio_samples += (size_t)samples;
            layout_refs[i] = (h3_layout_ref){
                H3_LAYOUT_REF_AUDIO, 0, 0, 0, latent.length};
        }
        h3_audio_latent_free(&latent);
    }

    h3_text_embedding text = {0};
    int have_text = 0;
    h3_reference_presentation *presentations = NULL;
    if (ok && reference_count) {
        presentations = calloc(reference_count, sizeof(*presentations));
        ok = presentations != NULL;
        if (!ok) snprintf(error, error_size, "out of memory");
        for (size_t i = 0; ok && i < reference_count; i++) {
            if (references[i].kind == H3_JOB_REF_AUDIO) {
                presentations[i].kind = H3_PRESENTATION_AUDIO;
                presentations[i].has_audio = 1;
            } else {
                presentations[i].kind =
                    references[i].kind == H3_JOB_REF_VIDEO ?
                        H3_PRESENTATION_VIDEO : H3_PRESENTATION_IMAGE;
                presentations[i].vision = &vision[meta[i].vision_cursor];
                presentations[i].vision_count = meta[i].vision_count;
                presentations[i].timestamps = meta[i].timestamps;
            }
        }
    }
    if (ok) {
        ok = h3_multimodal_encode_ref2va_bf16(
            engine->ref2va_tokenizer, text_dir, engine->shader_source_path,
            prompt, presentations, reference_count, NULL, NULL, &text, error,
            error_size);
        have_text = ok;
    }

    if (engine->conditioning_lock)
        pthread_mutex_unlock(engine->conditioning_lock);

    free(text_dir);
    free(vae_dir);
    free(audio_vae_dir);
    free(presentations);
    for (size_t i = 0; i < reference_count; i++) free(meta[i].timestamps);
    free(meta);
    for (size_t i = 0; i < vision_count; i++) h3_vision_output_free(&vision[i]);
    free(vision);

    if (!ok) {
        free(layout_refs);
        free(video_rows);
        free(audio_rows);
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
        free(layout_refs);
        free(video_rows);
        free(audio_rows);
        return 0;
    }

    condition_out->refs = layout_refs;
    condition_out->ref_count = reference_count;
    condition_out->video_rows = video_rows;
    condition_out->video_elements = video_elements;
    condition_out->audio_rows = audio_rows;
    condition_out->audio_elements = audio_elements;
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
    int ref2va = request->reference_count > 0;
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
    h3_ref2va_condition ref_condition = {0};
    double conditioning_start = now_seconds();
    int cond_ok = ref2va ?
        compute_ref2va_conditioning(
            engine, request->prompt, request->references,
            request->reference_count, width, height, request->frames, &cond,
            &ref_condition, error, error_size) :
        compute_conditioning(engine, request->prompt, &cond, error,
                             error_size);
    if (!cond_ok) return 0;
    if (conditioning_seconds)
        *conditioning_seconds = now_seconds() - conditioning_start;

    h3_video_condition condition = {0};
    if (ref2va) {
        condition.release_directory = engine->ref2va_directory;
        condition.layout_references = ref_condition.refs;
        condition.reference_count = ref_condition.ref_count;
        condition.condition_video_rows = ref_condition.video_rows;
        condition.condition_video_elements = ref_condition.video_elements;
        condition.condition_audio_rows = ref_condition.audio_rows;
        condition.condition_audio_elements = ref_condition.audio_elements;
    }

    h3_video_request req = {0};
    req.fl2va_directory = engine->fl2va_directory;
    req.shader_source_path = engine->shader_source_path;
    req.conditioning = cond.values;
    req.conditioning_tokens = cond.tokens;
    req.width = width;
    req.height = height;
    req.frames = request->frames;
    req.steps = request->steps > 0 ? request->steps : engine->steps;
    req.seed = request->seed;
    req.output_path = output_path;
    req.condition = ref2va ? &condition : NULL;
    req.cancel_requested = request->cancel_requested;
    keepalive_start(engine);
    int ok = h3_video_generate(&req, timing, progress, progress_opaque, error,
                               error_size);
    keepalive_stop(engine);
    qwen_intermediate_state_free(&cond);
    h3_ref2va_condition_free(&ref_condition);
    return ok;
}

int h3_generation_run_job(h3_job *job, void *engine_ptr) {
    h3_generation_engine *engine = engine_ptr;
    if (!engine || !job) return 0;
    char *error = job->error;
    size_t error_size = sizeof(job->error);
    error[0] = '\0';

    if (job->type == H3_JOB_VIDEO) {
        /* h3_job_request.references is a borrowed view (const char* paths);
         * job->references owns its paths (char*). job->reference_count is
         * guaranteed <= H3_JOB_MAX_REFERENCES by h3_job_submit(). */
        h3_job_reference_request ref_views[H3_JOB_MAX_REFERENCES];
        for (size_t i = 0; i < job->reference_count; i++) {
            ref_views[i].kind = job->references[i].kind;
            ref_views[i].path = job->references[i].path;
        }
        h3_job_request request = {job->type, job->prompt, job->seed,
                                  job->width, job->height, job->frames,
                                  job->steps, ref_views, job->reference_count,
                                  &job->cancel_requested};
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
        req.steps = job->steps > 0 ? job->steps : engine->steps;
        req.seed = job->seed;
        req.cancel_requested = &job->cancel_requested;
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
