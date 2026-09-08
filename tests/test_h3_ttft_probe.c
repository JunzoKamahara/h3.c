/* P8-SCHED-01e0 (slow, diagnostic only -- no fix): find what makes the first
 * chat token cost ~8 s during a video job when steady decode is native.
 *
 * Splits a chat request into prep / prefill / decode-1 / decode-2 / decode-3 /
 * steady, and samples the kernel's cumulative page-fault + compressor counters
 * across the first-token window vs the steady window. Then the causal test:
 * warm a SEPARATE scratch session by one token right before the real chat and
 * see whether the real TTFT collapses -- which would place the cold cost in
 * shared model / GPU state rather than the chat session's own KV.
 *
 *   ./h3_ttft_probe_test MiniMax-H3
 */

#include "h3_generation.h"
#include "h3_gpu_sched.h"
#include "h3_job.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define STEADY_TOKENS 12
static const char *CHAT_PROMPT = "Explain in one sentence why the sky is blue.";
static const char *VIDEO_PROMPT = "A red fox walking through snow";
static const char *WARM_PROMPT = "Hello.";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_ttft_probe.c: %s\n", m);
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

/* -------- kernel VM counters (cumulative) -------- */
typedef struct {
    uint64_t pageins, decompressions, compressions, swapins, swapouts, faults;
} vmcounters;

static vmcounters vm_now(void) {
    vmcounters v;
    memset(&v, 0, sizeof(v));
    vm_statistics64_data_t s;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s,
                          &count) == KERN_SUCCESS) {
        v.pageins = s.pageins;
        v.decompressions = s.decompressions;
        v.compressions = s.compressions;
        v.swapins = s.swapins;
        v.swapouts = s.swapouts;
        v.faults = s.faults;
    }
    return v;
}
static void vm_print_delta(const char *label, vmcounters a, vmcounters b) {
    printf("  %-22s pagein %8llu  decompress %8llu  compress %8llu  "
           "swapin %6llu  swapout %6llu  fault %9llu\n",
           label, (unsigned long long)(b.pageins - a.pageins),
           (unsigned long long)(b.decompressions - a.decompressions),
           (unsigned long long)(b.compressions - a.compressions),
           (unsigned long long)(b.swapins - a.swapins),
           (unsigned long long)(b.swapouts - a.swapouts),
           (unsigned long long)(b.faults - a.faults));
}

/* -------- chat, split into phases -------- */
typedef struct {
    double prefill_s, decode1_s, decode2_s, decode3_s;
    double steady_tok_s;
    int tokens;
    vmcounters vm_start, vm_after_d1, vm_after_steady;
} chatphases;

static void chat_phases(qwen_session *session, h3_tokenizer *tok,
                        pthread_mutex_t *lock, chatphases *out) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    require(h3_tokenizer_encode(tok, CHAT_PROMPT, 1, &ids, &n, error,
                                sizeof(error)),
            error);
    memset(out, 0, sizeof(*out));

    pthread_mutex_lock(lock);
    require(qwen_session_rewind(session, 0, error, sizeof(error)), error);

    out->vm_start = vm_now();
    double t = now_seconds();
    h3_gpu_sched_chat_enter();
    require(qwen_session_eval(session, ids, n, error, sizeof(error)), error);
    h3_gpu_sched_chat_leave();
    out->prefill_s = now_seconds() - t;

    double phase[3];
    for (int i = 0; i < 3; i++) {
        uint32_t next = 0;
        require(qwen_session_sample(session, &next, error, sizeof(error)),
                error);
        t = now_seconds();
        h3_gpu_sched_chat_enter();
        require(qwen_session_eval(session, &next, 1, error, sizeof(error)),
                error);
        h3_gpu_sched_chat_leave();
        phase[i] = now_seconds() - t;
        if (i == 0) out->vm_after_d1 = vm_now();
    }
    out->decode1_s = phase[0];
    out->decode2_s = phase[1];
    out->decode3_s = phase[2];

    double steady_start = now_seconds();
    int steady = 0;
    for (int i = 0; i < STEADY_TOKENS; i++) {
        uint32_t next = 0;
        require(qwen_session_sample(session, &next, error, sizeof(error)),
                error);
        h3_gpu_sched_chat_enter();
        int ok = qwen_session_eval(session, &next, 1, error, sizeof(error));
        h3_gpu_sched_chat_leave();
        if (!ok) break;
        steady++;
    }
    double steady_s = now_seconds() - steady_start;
    out->vm_after_steady = vm_now();
    pthread_mutex_unlock(lock);
    h3_tokenizer_ids_free(ids);

    out->tokens = 4 + steady;
    out->steady_tok_s = steady_s > 0 ? steady / steady_s : 0.0;
}

static void warm_scratch(qwen_session *scratch, h3_tokenizer *tok,
                         pthread_mutex_t *lock) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    require(h3_tokenizer_encode(tok, WARM_PROMPT, 1, &ids, &n, error,
                                sizeof(error)),
            error);
    pthread_mutex_lock(lock);
    require(qwen_session_rewind(scratch, 0, error, sizeof(error)), error);
    h3_gpu_sched_chat_enter();
    require(qwen_session_eval(scratch, ids, n, error, sizeof(error)), error);
    uint32_t t = 0;
    require(qwen_session_sample(scratch, &t, error, sizeof(error)), error);
    require(qwen_session_eval(scratch, &t, 1, error, sizeof(error)), error);
    h3_gpu_sched_chat_leave();
    pthread_mutex_unlock(lock);
    h3_tokenizer_ids_free(ids);
}

/* -------- video -------- */
typedef struct {
    h3_generation_engine *engine;
    char output_path[1024];
    _Atomic int denoise_started, finished;
    int ok;
    char error[512];
} vidctx;

static void vid_progress(const char *phase, int c, int t, void *o) {
    (void)c;
    (void)t;
    vidctx *v = o;
    if (phase && strstr(phase, "denoise")) atomic_store(&v->denoise_started, 1);
}
static void *vid_thread(void *o) {
    vidctx *v = o;
    h3_job_request r = {H3_JOB_VIDEO, VIDEO_PROMPT, 42, 256, 256, 25};
    v->ok = h3_generation_generate_video(v->engine, &r, v->output_path, NULL,
                                         NULL, vid_progress, v, v->error,
                                         sizeof(v->error));
    atomic_store(&v->finished, 1);
    return NULL;
}

static void print_phases(const char *label, const chatphases *p) {
    printf("  %-16s prefill %5.2fs  d1 %5.2fs  d2 %5.2fs  d3 %5.2fs  "
           "steady %4.2f tok/s  (%d tok)\n",
           label, p->prefill_s, p->decode1_s, p->decode2_s, p->decode3_s,
           p->steady_tok_s, p->tokens);
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];
    char *weights = path_join(root, "FL2VA/text_encoder");
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *fl2va = path_join(root, "FL2VA");

    qwen_engine *engine = NULL;
    require(qwen_engine_open(&engine, weights, "h3_shaders.metal", error,
                            sizeof(error)),
            error);
    h3_tokenizer *tok = h3_tokenizer_load(tp, error, sizeof(error));
    require(tok != NULL, error);
    qwen_session *chat = NULL, *scratch = NULL;
    require(qwen_session_create(&chat, engine, error, sizeof(error)), error);
    require(qwen_session_create(&scratch, engine, error, sizeof(error)), error);
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    h3_generation_engine *gen = h3_generation_engine_acquire(
        engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
    require(gen != NULL, error);
    char dir[] = "/tmp/h3-ttft-XXXXXX";
    require(mkdtemp(dir) != NULL, "mkdtemp");

    /* warm everything once */
    {
        chatphases w;
        chat_phases(chat, tok, &lock, &w);
    }

    /* 1. idle reference */
    chatphases idle;
    chat_phases(chat, tok, &lock, &idle);

    /* 2. cold under load */
    chatphases cold;
    {
        vidctx v;
        memset(&v, 0, sizeof(v));
        v.engine = gen;
        snprintf(v.output_path, sizeof(v.output_path), "%s/cold.mp4", dir);
        pthread_t t;
        require(pthread_create(&t, NULL, vid_thread, &v) == 0, "spawn");
        for (int w = 0; w < 180000 && !atomic_load(&v.denoise_started) &&
                        !atomic_load(&v.finished);
             w += 50)
            usleep(50000);
        sleep(15); /* let the chat working set cool while denoise runs */
        chat_phases(chat, tok, &lock, &cold);
        pthread_join(t, NULL);
        require(v.ok, v.error);
        unlink(v.output_path);
    }

    /* 3. warm scratch 1-token right before the real chat, under load */
    chatphases warm;
    {
        vidctx v;
        memset(&v, 0, sizeof(v));
        v.engine = gen;
        snprintf(v.output_path, sizeof(v.output_path), "%s/warm.mp4", dir);
        pthread_t t;
        require(pthread_create(&t, NULL, vid_thread, &v) == 0, "spawn");
        for (int w = 0; w < 180000 && !atomic_load(&v.denoise_started) &&
                        !atomic_load(&v.finished);
             w += 50)
            usleep(50000);
        sleep(15);
        warm_scratch(scratch, tok, &lock);
        chat_phases(chat, tok, &lock, &warm);
        pthread_join(t, NULL);
        require(v.ok, v.error);
        unlink(v.output_path);
    }

    printf("\n=== P8-SCHED-01e0 (TTFT cold working-set probe) =======\n");
    printf("phase timing:\n");
    print_phases("idle", &idle);
    print_phases("under load (cold)", &cold);
    print_phases("under load + warm", &warm);

    printf("\nkernel VM deltas -- first-token window (prefill + decode-1):\n");
    vm_print_delta("idle", idle.vm_start, idle.vm_after_d1);
    vm_print_delta("cold", cold.vm_start, cold.vm_after_d1);
    vm_print_delta("warm", warm.vm_start, warm.vm_after_d1);
    printf("\nkernel VM deltas -- steady window (decode-2 .. end):\n");
    vm_print_delta("cold", cold.vm_after_d1, cold.vm_after_steady);

    double ttft_idle = idle.prefill_s + idle.decode1_s;
    double ttft_cold = cold.prefill_s + cold.decode1_s;
    double ttft_warm = warm.prefill_s + warm.decode1_s;
    printf("\nTTFT (prefill + decode-1):  idle %.2fs   cold %.2fs   warm %.2fs\n",
           ttft_idle, ttft_cold, ttft_warm);
    printf("interpretation: warm << cold  -> cold cost is in SHARED model / GPU "
           "state (scratch warm-up reaches it).\n");
    printf("                warm ~= cold  -> cold cost is in the chat session's "
           "own KV / activations.\n");
    printf("=======================================================\n");

    h3_generation_engine_release(gen);
    qwen_session_free(chat);
    qwen_session_free(scratch);
    h3_tokenizer_free(tok);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    rmdir(dir);
    free(weights);
    free(tp);
    free(fl2va);
    puts("ok: P8-SCHED-01e0 probe complete");
    return 0;
}
