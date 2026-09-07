/* P8-VID-01 (slow): one real video job through the job manager + generation
 * engine, plus a concurrency smoke test -- a chat decode runs on a separate
 * session while the video job is in flight, sharing the immutable weights but
 * not the KV / sampling state.
 *
 *   ./h3_video_job_test MiniMax-H3
 *
 * Loads the FL2VA transformer (SSD streaming) + both VAEs. Not in `make test`.
 */

#include "h3_ffmpeg.h"
#include "h3_generation.h"
#include "h3_job.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_video_job.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

/* A short greedy decode on `session`, serialised against the generation
 * engine's conditioning through `lock`. Proves chat still works while a video
 * job is running. */
static void chat_smoke(qwen_session *session, h3_tokenizer *tokenizer,
                       pthread_mutex_t *lock) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    require(h3_tokenizer_encode(tokenizer, "The capital of France is", 1, &ids,
                               &n, error, sizeof(error)),
            error);

    pthread_mutex_lock(lock);
    int ok = qwen_session_eval(session, ids, n, error, sizeof(error));
    for (int step = 0; ok && step < 8; step++) {
        uint32_t next = 0;
        ok = qwen_session_sample(session, &next, error, sizeof(error));
        require(!ok || next < 151936, "sampled token in range");
        ok = ok && qwen_session_eval(session, &next, 1, error, sizeof(error));
    }
    pthread_mutex_unlock(lock);
    h3_tokenizer_ids_free(ids);
    require(ok, error);
}

static int has_audio_stream(const char *path) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams a:0 -show_entries "
             "stream=codec_type -of csv=p=0 '%s' | grep -q audio",
             path);
    return system(cmd) == 0;
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

    /* Same lock the chat path and the conditioning forward share. */
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);

    h3_generation_engine *gen = h3_generation_engine_acquire(
        engine, fl2va, "h3_shaders.metal", &lock, error, sizeof(error));
    require(gen != NULL, error);

    char artifact_dir[] = "/tmp/h3-vidjob-XXXXXX";
    require(mkdtemp(artifact_dir) != NULL, "mkdtemp");

    h3_job_manager *manager =
        h3_job_manager_new(artifact_dir, error, sizeof(error));
    require(manager != NULL, error);
    require(h3_job_manager_start(manager, h3_generation_run_job, gen, error,
                                sizeof(error)),
            error);

    h3_job_request request = {H3_JOB_VIDEO, "A red fox walking through snow", 42,
                              256, 256, 25};
    char id[H3_JOB_ID_SIZE];
    require(h3_job_submit(manager, &request, id, sizeof(id), error,
                         sizeof(error)),
            error);
    printf("submitted video job %s\n", id);

    /* Wait until it is actually running, then hit the chat path. */
    h3_job_info info;
    int saw_running = 0;
    for (int waited = 0; waited < 120000; waited += 200) {
        require(h3_job_get(manager, id, &info), "job vanished");
        if (info.status == H3_JOB_RUNNING) {
            saw_running = 1;
            break;
        }
        if (info.status == H3_JOB_FAILED) fail(info.error);
        usleep(200000);
    }
    require(saw_running, "job never entered RUNNING");
    printf("job is running -- exercising the chat path concurrently\n");
    chat_smoke(chat, tokenizer, &lock);
    chat_smoke(chat, tokenizer, &lock);
    printf("chat decode completed while the video job was in flight\n");

    for (int waited = 0; waited < 1200000; waited += 500) {
        require(h3_job_get(manager, id, &info), "job vanished");
        if (info.status == H3_JOB_SUCCEEDED || info.status == H3_JOB_FAILED)
            break;
        usleep(500000);
    }
    require(info.status == H3_JOB_SUCCEEDED, info.error);
    printf("job succeeded -> %s\n", info.output_path);

    struct stat st;
    require(stat(info.output_path, &st) == 0 && st.st_size > 1024,
            "artifact is a non-trivial file");
    int w = 0, h = 0;
    require(h3_ffprobe_visual_size(info.output_path, &w, &h, error,
                                   sizeof(error)),
            error);
    require(w == 256 && h == 256, "video is 256x256");
    require(has_audio_stream(info.output_path), "video carries an audio track");
    printf("artifact: %lld bytes, %dx%d, with audio\n", (long long)st.st_size, w,
           h);

    h3_job_manager_free(manager);
    h3_generation_engine_release(gen);
    qwen_session_free(chat);
    h3_tokenizer_free(tokenizer);
    qwen_engine_close(engine);
    pthread_mutex_destroy(&lock);
    unlink(info.output_path);
    rmdir(artifact_dir);
    free(weights);
    free(tokenizer_path);
    free(fl2va);
    puts("ok: P8-VID-01 real video job + concurrent chat");
    return 0;
}
