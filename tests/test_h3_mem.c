/* P8-MEM-01 (slow, decision task): measure memory and chat latency in four
 * states -- idle, chat only, video generation only, video generation + chat --
 * in one process with one configuration, plus the video pipeline's per-stage
 * timing. Prints a table and the two slowdown ratios; the branch decision
 * (parallel OK / compute-bound / memory-unsafe) is made by reading it.
 *
 *   ./h3_mem_test MiniMax-H3
 *
 * Loads the resident chat weights + the FL2VA transformer + both VAEs. Not in
 * `make test`.
 */

#include "h3_generation.h"
#include "h3_job.h"
#include "h3_image_gen.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define CHAT_TOKENS 24
#define VIDEO_FRAMES 25
static const char *CHAT_PROMPT = "Explain in one sentence why the sky is blue.";
static const char *VIDEO_PROMPT = "A red fox walking through snow";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_mem.c: %s\n", m);
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

/* -------------------------------------------------------------- memory ---- */

typedef struct {
    double footprint_gb;   /* phys_footprint -- Activity Monitor "Memory" */
    double resident_gb;
    double resident_peak_gb;
    double compressed_gb;
    double sys_free_gb;
    double swap_used_gb;
} memsnap;

static double to_gb(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void mem_snapshot(memsnap *s) {
    memset(s, 0, sizeof(*s));

    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info,
                  &count) == KERN_SUCCESS) {
        s->footprint_gb = to_gb(vm_info.phys_footprint);
        s->compressed_gb = to_gb(vm_info.compressed);
    }

    mach_task_basic_info_data_t basic;
    count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&basic,
                  &count) == KERN_SUCCESS) {
        s->resident_gb = to_gb(basic.resident_size);
        s->resident_peak_gb = to_gb(basic.resident_size_max);
    }

    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);
    vm_statistics64_data_t vm_stat;
    count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t free_pages =
            (uint64_t)vm_stat.free_count + vm_stat.inactive_count;
        s->sys_free_gb = to_gb(free_pages * (uint64_t)page);
    }

    struct xsw_usage swap;
    size_t len = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &len, NULL, 0) == 0)
        s->swap_used_gb = to_gb(swap.xsu_used);
}

/* ---------------------------------------------------------------- chat ---- */

typedef struct {
    double ttft_s;
    double tok_s;
    double inter_p50_ms;
    double inter_p95_ms;
    double total_s;
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
    double start = now_seconds();
    require(qwen_session_eval(session, ids, n, error, sizeof(error)), error);
    double last = start;
    int produced = 0;
    for (int step = 0; step < CHAT_TOKENS; step++) {
        uint32_t next = 0;
        require(qwen_session_sample(session, &next, error, sizeof(error)),
                error);
        double t = now_seconds();
        if (step == 0)
            out->ttft_s = t - start;
        else
            inter[step - 1] = (t - last) * 1000.0;
        last = t;
        produced++;
        if (!qwen_session_eval(session, &next, 1, error, sizeof(error))) {
            require(qwen_session_rewind(session, 0, error, sizeof(error)),
                    error);
            break;
        }
    }
    out->total_s = now_seconds() - start;
    pthread_mutex_unlock(lock);
    h3_tokenizer_ids_free(ids);

    out->tokens = produced;
    out->tok_s = out->total_s > 0 ? produced / out->total_s : 0.0;
    int gaps = produced - 1;
    if (gaps > 0) {
        qsort(inter, (size_t)gaps, sizeof(inter[0]), cmp_double);
        out->inter_p50_ms = inter[gaps / 2];
        out->inter_p95_ms = inter[(int)ceil(gaps * 0.95) - 1 < 0
                                      ? 0
                                      : (int)ceil(gaps * 0.95) - 1];
    }
}

/* --------------------------------------------------------------- video --- */

typedef struct {
    h3_generation_engine *engine;
    char output_path[1024];
    _Atomic int denoise_started;
    _Atomic int finished;
    int ok;
    char error[512];
    double conditioning_s;
    h3_video_timing timing;
    double total_s;
} vidctx;

static void vid_progress(const char *phase, int completed, int total,
                         void *opaque) {
    (void)completed;
    (void)total;
    vidctx *v = opaque;
    if (phase && strstr(phase, "denoise"))
        atomic_store(&v->denoise_started, 1);
}

static void *vid_thread(void *opaque) {
    vidctx *v = opaque;
    h3_job_request request = {H3_JOB_VIDEO, VIDEO_PROMPT, 42, 256, 256,
                              VIDEO_FRAMES};
    double start = now_seconds();
    v->ok = h3_generation_generate_video(v->engine, &request, v->output_path,
                                         &v->conditioning_s, &v->timing,
                                         vid_progress, v, v->error,
                                         sizeof(v->error));
    v->total_s = now_seconds() - start;
    atomic_store(&v->finished, 1);
    return NULL;
}

/* --------------------------------------------------------------- report -- */

static void print_mem_row(const char *label, const memsnap *s) {
    printf("  %-18s footprint %6.1f  resident %6.1f  peak %6.1f  "
           "compressed %5.1f  sys-free %6.1f  swap %4.1f  (GB)\n",
           label, s->footprint_gb, s->resident_gb, s->resident_peak_gb,
           s->compressed_gb, s->sys_free_gb, s->swap_used_gb);
}

static void print_chat_row(const char *label, const chatmetrics *m) {
    printf("  %-14s TTFT %5.2fs  %5.1f tok/s  inter p50 %6.1f ms  "
           "p95 %6.1f ms  total %5.2fs  (%d tok)\n",
           label, m->ttft_s, m->tok_s, m->inter_p50_ms, m->inter_p95_ms,
           m->total_s, m->tokens);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];

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
    h3_generation_engine *gen = h3_generation_engine_acquire(
        engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
    require(gen != NULL, error);

    char artifact_dir[] = "/tmp/h3-mem-XXXXXX";
    require(mkdtemp(artifact_dir) != NULL, "mkdtemp");

    /* Warm the resident weights so M0 reflects steady state. */
    {
        chatmetrics warm;
        chat_bench(chat, tokenizer, &lock, &warm);
    }

    memsnap m0;
    mem_snapshot(&m0);

    /* M1: chat only. */
    chatmetrics chat_m1;
    chat_bench(chat, tokenizer, &lock, &chat_m1);
    memsnap m1;
    mem_snapshot(&m1);

    /* M2: video only. */
    vidctx v2;
    memset(&v2, 0, sizeof(v2));
    v2.engine = gen;
    snprintf(v2.output_path, sizeof(v2.output_path), "%s/m2.mp4", artifact_dir);
    vid_thread(&v2);
    require(v2.ok, v2.error);
    memsnap m2;
    mem_snapshot(&m2);

    /* M3: video generation + chat. */
    vidctx v3;
    memset(&v3, 0, sizeof(v3));
    v3.engine = gen;
    snprintf(v3.output_path, sizeof(v3.output_path), "%s/m3.mp4", artifact_dir);
    pthread_t vt;
    require(pthread_create(&vt, NULL, vid_thread, &v3) == 0, "pthread_create");
    for (int waited = 0; waited < 180000 &&
                         !atomic_load(&v3.denoise_started) &&
                         !atomic_load(&v3.finished);
         waited += 50)
        usleep(50000);
    require(atomic_load(&v3.denoise_started) || atomic_load(&v3.finished),
            "video never reached the denoise stage");

    chatmetrics chat_m3;
    chat_bench(chat, tokenizer, &lock, &chat_m3);
    memsnap m3_during;
    mem_snapshot(&m3_during);

    pthread_join(vt, NULL);
    require(v3.ok, v3.error);
    memsnap m3_after;
    mem_snapshot(&m3_after);

    /* ------------------------------------------------------------ output */
    printf("\n=== P8-MEM-01 =========================================\n");
    printf("chat fixture: %d tokens   video: %d frames, 256x256, 12 steps\n\n",
           CHAT_TOKENS, VIDEO_FRAMES);

    printf("memory:\n");
    print_mem_row("M0 idle", &m0);
    print_mem_row("M1 chat", &m1);
    print_mem_row("M2 video", &m2);
    print_mem_row("M3 video+chat", &m3_during);
    print_mem_row("M3 after join", &m3_after);

    printf("\nchat latency:\n");
    print_chat_row("M1 chat-only", &chat_m1);
    print_chat_row("M3 w/ video", &chat_m3);

    printf("\nvideo pipeline (M2, seconds):\n");
    printf("  conditioning %6.1f   transformer-load %6.1f   denoise %6.1f\n",
           v2.conditioning_s, v2.timing.transformer_load_s, v2.timing.denoise_s);
    printf("  video-decode %6.1f   audio-decode     %6.1f   mux     %6.2f\n",
           v2.timing.video_decode_s, v2.timing.audio_decode_s, v2.timing.mux_s);
    printf("  total        %6.1f\n", v2.total_s);
    printf("  M3 video total %6.1f\n", v3.total_s);

    double chat_ratio =
        chat_m1.tok_s > 0 ? chat_m3.tok_s / chat_m1.tok_s : 0.0;
    double video_ratio = v2.total_s > 0 ? v3.total_s / v2.total_s : 0.0;
    printf("\nratios:\n");
    printf("  chat  tok/s  M3/M1 = %.2f  (>= 0.70 -> parallel OK)\n",
           chat_ratio);
    printf("  video time   M3/M2 = %.2f  (<= 1.30 -> parallel OK)\n",
           video_ratio);
    printf("  peak footprint %.1f GB, peak resident %.1f GB, swap %.1f GB\n",
           m3_during.footprint_gb, m3_after.resident_peak_gb,
           m3_during.swap_used_gb);
    printf("=======================================================\n");

    require(chat_m1.tokens == CHAT_TOKENS && chat_m3.tokens == CHAT_TOKENS,
            "chat produced the full fixture in both states");

    h3_generation_engine_release(gen);
    qwen_session_free(chat);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    unlink(v2.output_path);
    unlink(v3.output_path);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    puts("ok: P8-MEM-01 measurement complete");
    return 0;
}
