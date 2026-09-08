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

#include "h3_ffmpeg.h"
#include "h3_generation.h"
#include "h3_gpu_sched.h"
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
    h3_gpu_sched_chat_enter();
    int prefilled = qwen_session_eval(session, ids, n, error, sizeof(error));
    h3_gpu_sched_chat_leave();
    require(prefilled, error);
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
        h3_gpu_sched_chat_enter();
        int advanced = qwen_session_eval(session, &next, 1, error,
                                         sizeof(error));
        h3_gpu_sched_chat_leave();
        if (!advanced) {
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

/* Generate solo (no concurrent chat) and return wall time; fills `*out`. */
static void video_solo(vidctx *out, h3_generation_engine *gen, const char *dir,
                       const char *tag) {
    memset(out, 0, sizeof(*out));
    out->engine = gen;
    snprintf(out->output_path, sizeof(out->output_path), "%s/%s.mp4", dir, tag);
    vid_thread(out);
    require(out->ok, out->error);
}

/* Run a video job while a chat fixture executes once denoising has started. */
static void video_with_chat(h3_generation_engine *gen, qwen_session *chat,
                            h3_tokenizer *tok, pthread_mutex_t *lock,
                            const char *dir, const char *tag, chatmetrics *cm,
                            double *video_s) {
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
    require(atomic_load(&v.denoise_started) || atomic_load(&v.finished),
            "video never reached the denoise stage");
    chat_bench(chat, tok, lock, cm);
    pthread_join(t, NULL);
    require(v.ok, v.error);
    *video_s = v.total_s;
    unlink(v.output_path);
}

/* Max abs difference between frame 0 of two clips (quality-unchanged check). */
static double first_frame_max_diff(const char *path_a, const char *path_b) {
    char error[512];
    float *a = NULL, *b = NULL;
    int fa = 0, fb = 0;
    require(h3_ffmpeg_read_video_f32(path_a, 256, 256, 5, &a, &fa, error,
                                     sizeof(error)),
            error);
    require(h3_ffmpeg_read_video_f32(path_b, 256, 256, 5, &b, &fb, error,
                                     sizeof(error)),
            error);
    int frames = fa < fb ? fa : fb;
    if (frames > 5) frames = 5;
    size_t n = (size_t)3 * (size_t)frames * 256 * 256;
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    free(a);
    free(b);
    return m;
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

    /* M1: chat only (the interactive reference). */
    chatmetrics chat_m1;
    chat_bench(chat, tokenizer, &lock, &chat_m1);
    memsnap m1;
    mem_snapshot(&m1);

    /* Video solo, scheduler OFF -- baseline timing + first frame. */
    h3_gpu_sched_set_enabled(0);
    vidctx v_off;
    video_solo(&v_off, gen, artifact_dir, "solo_off");
    memsnap m2;
    mem_snapshot(&m2);

    /* Video solo, scheduler ON, still no chat -- must match OFF: identical
     * pixels (the scheduler only inserts waits between submissions) and the
     * same wall time (it yields nothing when no chat waits). */
    h3_gpu_sched_set_enabled(1);
    vidctx v_on_solo;
    video_solo(&v_on_solo, gen, artifact_dir, "solo_on");
    double quality_max_diff =
        first_frame_max_diff(v_off.output_path, v_on_solo.output_path);
    double solo_ratio =
        v_off.total_s > 0 ? v_on_solo.total_s / v_off.total_s : 0.0;
    unlink(v_off.output_path);
    unlink(v_on_solo.output_path);

    /* M3 OFF: video + concurrent chat, no cooperative scheduling. */
    h3_gpu_sched_set_enabled(0);
    chatmetrics chat_off;
    double video_off_s;
    video_with_chat(gen, chat, tokenizer, &lock, artifact_dir, "m3_off",
                    &chat_off, &video_off_s);
    memsnap m3_off;
    mem_snapshot(&m3_off);

    /* M3 ON: same, with the cooperative scheduler. */
    h3_gpu_sched_set_enabled(1);
    h3_gpu_sched_reset_stats();
    chatmetrics chat_on;
    double video_on_s;
    video_with_chat(gen, chat, tokenizer, &lock, artifact_dir, "m3_on",
                    &chat_on, &video_on_s);
    memsnap m3_on;
    mem_snapshot(&m3_on);
    unsigned long yield_count = h3_gpu_sched_yield_count();
    double yield_seconds = h3_gpu_sched_yield_seconds();

    /* ------------------------------------------------------------ output */
    printf("\n=== P8-MEM-01 / P8-SCHED-01b ==========================\n");
    printf("chat fixture: %d tokens   video: %d frames, 256x256, 12 steps\n\n",
           CHAT_TOKENS, VIDEO_FRAMES);

    printf("memory:\n");
    print_mem_row("M0 idle", &m0);
    print_mem_row("M1 chat", &m1);
    print_mem_row("video solo", &m2);
    print_mem_row("M3 sched-off", &m3_off);
    print_mem_row("M3 sched-on", &m3_on);

    printf("\nvideo pipeline (solo, scheduler off, seconds):\n");
    printf("  conditioning %6.1f   transformer-load %6.1f   denoise %6.1f\n",
           v_off.conditioning_s, v_off.timing.transformer_load_s,
           v_off.timing.denoise_s);
    printf("  video-decode %6.1f   audio-decode     %6.1f   mux     %6.2f\n",
           v_off.timing.video_decode_s, v_off.timing.audio_decode_s,
           v_off.timing.mux_s);
    printf("  total        %6.1f\n", v_off.total_s);

    printf("\nchat latency:\n");
    print_chat_row("M1 idle", &chat_m1);
    print_chat_row("M3 sched-off", &chat_off);
    print_chat_row("M3 sched-on", &chat_on);

    double off_ratio = chat_m1.tok_s > 0 ? chat_off.tok_s / chat_m1.tok_s : 0;
    double on_ratio = chat_m1.tok_s > 0 ? chat_on.tok_s / chat_m1.tok_s : 0;
    double video_slow_off =
        v_off.total_s > 0 ? video_off_s / v_off.total_s : 0;
    double video_slow_on = v_off.total_s > 0 ? video_on_s / v_off.total_s : 0;

    printf("\nP8-SCHED-01b gates (vs M1 idle: %.2f tok/s, TTFT %.2fs, "
           "p95 %.0f ms):\n",
           chat_m1.tok_s, chat_m1.ttft_s, chat_m1.inter_p95_ms);
    printf("  scheduler OFF: chat %.2f tok/s (%.2fx)  TTFT %.2fs  p95 %.0f ms"
           "   video %.0fs (%.2fx)\n",
           chat_off.tok_s, off_ratio, chat_off.ttft_s, chat_off.inter_p95_ms,
           video_off_s, video_slow_off);
    printf("  scheduler ON : chat %.2f tok/s (%.2fx)  TTFT %.2fs  p95 %.0f ms"
           "   video %.0fs (%.2fx)\n",
           chat_on.tok_s, on_ratio, chat_on.ttft_s, chat_on.inter_p95_ms,
           video_on_s, video_slow_on);
    printf("  target: TTFT <= 2.0s, chat ratio >= 0.70, p95 <= %.0f ms, "
           "video slowdown <= 1.30\n",
           2.0 * chat_m1.inter_p95_ms);
    printf("  scheduler waited %lu times, %.1fs total\n", yield_count,
           yield_seconds);
    printf("  scheduler idle cost: solo video ON/OFF = %.3f  (expect ~1.0)\n",
           solo_ratio);
    printf("  quality: max|frame0_on - frame0_off| = %.2e  (expect ~0)\n",
           quality_max_diff);
    printf("  peak footprint %.1f GB, swap %.1f GB\n", m3_on.footprint_gb,
           m3_on.swap_used_gb);
    printf("=======================================================\n");

    require(chat_m1.tokens == CHAT_TOKENS && chat_off.tokens == CHAT_TOKENS &&
                chat_on.tokens == CHAT_TOKENS,
            "chat produced the full fixture in every state");
    require(quality_max_diff < 1e-3,
            "cooperative scheduling did not change the generated pixels");

    h3_generation_engine_release(gen);
    qwen_session_free(chat);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    puts("ok: P8-MEM-01 measurement complete");
    return 0;
}
