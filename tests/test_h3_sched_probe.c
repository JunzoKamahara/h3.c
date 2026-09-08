/* P8-SCHED-01a (slow, probe only -- no scheduler): measure the diffusion
 * transformer's GPU cadence at the transformer-block boundary and check
 * whether pausing the diffusion thread there lets a concurrent chat decode
 * make progress. Generation output is unchanged; this only reads clocks and,
 * optionally, sleeps the diffusion thread between blocks.
 *
 *   ./h3_sched_probe_test MiniMax-H3
 *
 * Runs three real 256x256 generations. Not in `make test`.
 *   A  baseline    -- H3_DIT_SCHED_PROBE=1, no concurrent chat
 *   B  concurrent  -- chat fixture runs during the job, no yield
 *   C  concurrent  -- same, with a 4 ms diffusion-thread yield per block
 */

#include "h3_generation.h"
#include "h3_gpu_sched.h"
#include "h3_image_gen.h"
#include "h3_job.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHAT_TOKENS 24
#define VIDEO_FRAMES 25
static const char *CHAT_PROMPT = "Explain in one sentence why the sky is blue.";
static const char *VIDEO_PROMPT = "A red fox walking through snow";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_sched_probe.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

typedef struct {
    double ttft_s;
    double tok_s;
    double p95_ms;
    int tokens;
} chatmetrics;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void chat_bench(qwen_session *session, h3_tokenizer *tokenizer,
                       pthread_mutex_t *lock, chatmetrics *out) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    require(h3_tokenizer_encode(tokenizer, CHAT_PROMPT, 1, &ids, &n, error,
                               sizeof(error)),
            error);
    double inter[CHAT_TOKENS];
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock(lock);
    require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
    double start = now_seconds(), last = start;
    require(qwen_session_eval(session, ids, n, error, sizeof(error)), error);
    int produced = 0;
    for (int step = 0; step < CHAT_TOKENS; step++) {
        uint32_t next = 0;
        require(qwen_session_sample(session, &next, error, sizeof(error)),
                error);
        double t = now_seconds();
        if (step == 0) out->ttft_s = t - start;
        else inter[step - 1] = (t - last) * 1000.0;
        last = t;
        produced++;
        if (!qwen_session_eval(session, &next, 1, error, sizeof(error))) break;
    }
    double total = now_seconds() - start;
    pthread_mutex_unlock(lock);
    h3_tokenizer_ids_free(ids);

    out->tokens = produced;
    out->tok_s = total > 0 ? produced / total : 0.0;
    int gaps = produced - 1;
    if (gaps > 0) {
        qsort(inter, (size_t)gaps, sizeof(inter[0]), cmp_double);
        int idx = (int)ceil(gaps * 0.95) - 1;
        if (idx < 0) idx = 0;
        out->p95_ms = inter[idx];
    }
}

typedef struct {
    h3_generation_engine *engine;
    char output_path[1024];
    _Atomic int denoise_started;
    _Atomic int finished;
    int ok;
    char error[512];
    double total_s;
} vidctx;

static void vid_progress(const char *phase, int completed, int total,
                         void *opaque) {
    (void)completed;
    (void)total;
    vidctx *v = opaque;
    if (phase && strstr(phase, "denoise")) atomic_store(&v->denoise_started, 1);
}

static void *vid_thread(void *opaque) {
    vidctx *v = opaque;
    h3_job_request request = {H3_JOB_VIDEO, VIDEO_PROMPT, 42, 256, 256,
                              VIDEO_FRAMES};
    double start = now_seconds();
    v->ok = h3_generation_generate_video(v->engine, &request, v->output_path,
                                         NULL, NULL, vid_progress, v, v->error,
                                         sizeof(v->error));
    v->total_s = now_seconds() - start;
    atomic_store(&v->finished, 1);
    return NULL;
}

static double run_solo(h3_generation_engine *gen, const char *dir,
                       const char *tag) {
    vidctx v;
    memset(&v, 0, sizeof(v));
    v.engine = gen;
    snprintf(v.output_path, sizeof(v.output_path), "%s/%s.mp4", dir, tag);
    vid_thread(&v);
    require(v.ok, v.error);
    unlink(v.output_path);
    return v.total_s;
}

static double run_concurrent(h3_generation_engine *gen, qwen_session *chat,
                             h3_tokenizer *tok, pthread_mutex_t *lock,
                             const char *dir, const char *tag, chatmetrics *cm) {
    vidctx v;
    memset(&v, 0, sizeof(v));
    v.engine = gen;
    snprintf(v.output_path, sizeof(v.output_path), "%s/%s.mp4", dir, tag);
    pthread_t t;
    require(pthread_create(&t, NULL, vid_thread, &v) == 0, "pthread_create");
    for (int waited = 0; waited < 180000 &&
                         !atomic_load(&v.denoise_started) &&
                         !atomic_load(&v.finished);
         waited += 50)
        usleep(50000);
    chat_bench(chat, tok, lock, cm);
    pthread_join(t, NULL);
    require(v.ok, v.error);
    unlink(v.output_path);
    return v.total_s;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];

    /* 01a measures the fixed-sleep probe only; keep the 01b cooperative
     * scheduler out of the way. */
    h3_gpu_sched_set_enabled(0);
    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tokenizer_path = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *fl2va = path_join(root, "FL2VA");

    qwen_engine *engine = NULL;
    require(qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                            sizeof(error)),
            error);
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    require(tokenizer != NULL, error);
    qwen_session *chat = NULL;
    require(qwen_session_create(&chat, engine, error, sizeof(error)), error);
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);

    char dir[] = "/tmp/h3-sched-XXXXXX";
    require(mkdtemp(dir) != NULL, "mkdtemp");

    /* Idle chat baseline. */
    chatmetrics idle;
    {
        h3_generation_engine *g = h3_generation_engine_acquire(
            engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
        require(g != NULL, error);
        chat_bench(chat, tokenizer, &lock, &idle);
        h3_generation_engine_release(g);
    }

    /* A: baseline cadence, probe on, no concurrent chat. */
    setenv("H3_DIT_SCHED_PROBE", "1", 1);
    unsetenv("H3_DIT_SCHED_PROBE_YIELD_US");
    double video_a;
    {
        h3_generation_engine *g = h3_generation_engine_acquire(
            engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
        require(g != NULL, error);
        printf("\n--- A: baseline block cadence (sched-probe lines above) ---\n");
        video_a = run_solo(g, dir, "a");
        h3_generation_engine_release(g);
    }

    /* B: concurrent chat, no yield. */
    chatmetrics chat_b;
    double video_b;
    {
        h3_generation_engine *g = h3_generation_engine_acquire(
            engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
        require(g != NULL, error);
        printf("\n--- B: chat during generation, no yield ---\n");
        video_b = run_concurrent(g, chat, tokenizer, &lock, dir, "b", &chat_b);
        h3_generation_engine_release(g);
    }

    /* C: concurrent chat, 4 ms diffusion-thread yield per block. */
    chatmetrics chat_c;
    double video_c;
    {
        setenv("H3_DIT_SCHED_PROBE_YIELD_US", "4000", 1);
        h3_generation_engine *g = h3_generation_engine_acquire(
            engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
        require(g != NULL, error);
        printf("\n--- C: chat during generation, 4 ms/block yield ---\n");
        video_c = run_concurrent(g, chat, tokenizer, &lock, dir, "c", &chat_c);
        h3_generation_engine_release(g);
        unsetenv("H3_DIT_SCHED_PROBE_YIELD_US");
    }

    printf("\n=== P8-SCHED-01a ======================================\n");
    printf("chat idle:            %5.2f tok/s  TTFT %5.2fs  p95 %6.1f ms\n",
           idle.tok_s, idle.ttft_s, idle.p95_ms);
    printf("A video solo:         %6.1f s\n", video_a);
    printf("B concurrent no-yield: chat %5.2f tok/s (%.2fx)  TTFT %5.2fs  "
           "p95 %6.1f ms   video %6.1f s (%.2fx)\n",
           chat_b.tok_s, idle.tok_s > 0 ? chat_b.tok_s / idle.tok_s : 0.0,
           chat_b.ttft_s, chat_b.p95_ms, video_b,
           video_a > 0 ? video_b / video_a : 0.0);
    printf("C concurrent 4ms/blk:  chat %5.2f tok/s (%.2fx)  TTFT %5.2fs  "
           "p95 %6.1f ms   video %6.1f s (%.2fx)\n",
           chat_c.tok_s, idle.tok_s > 0 ? chat_c.tok_s / idle.tok_s : 0.0,
           chat_c.ttft_s, chat_c.p95_ms, video_c,
           video_a > 0 ? video_c / video_a : 0.0);
    printf("interpretation: if C's chat ratio clearly beats B's, a "
           "block-boundary yield is an effective lever (build P8-SCHED-01b).\n");
    printf("=======================================================\n");

    require(chat_b.tokens == CHAT_TOKENS && chat_c.tokens == CHAT_TOKENS,
            "chat produced the full fixture under load");

    qwen_session_free(chat);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    rmdir(dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    puts("ok: P8-SCHED-01a GPU-cadence probe complete");
    return 0;
}
