/* P8-VID-02 (slow): the async video HTTP lifecycle end to end.
 *
 *   ./h3_video_http_test MiniMax-H3
 *
 * POST /v1/videos returns 202 fast; GET /v1/videos/{id} walks
 * queued -> running -> completed; /content is 409 before and a real MP4 after;
 * /v1/chat/completions still answers while the job runs; unknown ids are 404.
 *
 * Boots a full server (resident chat weights) and generates one real clip. Not
 * in `make test`.
 */

#include "h3_ffmpeg.h"
#include "qwen_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_video_http.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint8_t *http_roundtrip(uint16_t port, const char *raw, size_t *out_len) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("client socket");
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        fail("client connect");
    if (write(fd, raw, strlen(raw)) < 0) fail("client write");
    size_t cap = 8192, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) fail("client buffer");
    for (;;) {
        if (len + 1 >= cap) {
            cap *= 2;
            uint8_t *g = realloc(buf, cap);
            if (!g) fail("client grow");
            buf = g;
        }
        ssize_t n = read(fd, buf + len, cap - len - 1);
        if (n <= 0) break;
        len += (size_t)n;
    }
    buf[len] = '\0';
    close(fd);
    if (out_len) *out_len = len;
    return buf;
}

static const uint8_t *find_body(const uint8_t *response, size_t total,
                                size_t *body_len) {
    for (size_t i = 0; i + 4 <= total; i++)
        if (!memcmp(response + i, "\r\n\r\n", 4)) {
            *body_len = total - (i + 4);
            return response + i + 4;
        }
    *body_len = 0;
    return NULL;
}

/* Copy the value of a "key":"value" string field into out. */
static void json_string_field(const char *json, const char *key, char *out,
                              size_t out_size) {
    out[0] = '\0';
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *p = strstr(json, needle);
    if (!p) return;
    p += strlen(needle);
    const char *end = strchr(p, '"');
    if (!end) return;
    size_t n = (size_t)(end - p);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, p, n);
    out[n] = '\0';
}

static char *build_json_post(const char *path, const char *body) {
    size_t cap = strlen(body) + strlen(path) + 256;
    char *r = malloc(cap);
    if (!r) fail("alloc");
    snprintf(r, cap,
             "POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
             path, strlen(body), body);
    return r;
}

static char *build_get(const char *path) {
    size_t cap = strlen(path) + 96;
    char *r = malloc(cap);
    if (!r) fail("alloc");
    snprintf(r, cap, "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
             path);
    return r;
}

static int has_audio_stream(const char *path) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams a:0 -show_entries "
             "stream=codec_type -of csv=p=0 '%s' | grep -q audio",
             path);
    return system(cmd) == 0;
}

typedef struct {
    qwen_server *server;
    _Atomic uint16_t port;
} serve_state;

static void *serve_main(void *opaque) {
    serve_state *s = opaque;
    char error[512];
    if (!qwen_server_run(s->server, "127.0.0.1", 0, (uint16_t *)&s->port, error,
                         sizeof(error)))
        fprintf(stderr, "qwen_server_run: %s\n", error);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512], weights[1024], tokenizer[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    snprintf(tokenizer, sizeof(tokenizer),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);

    qwen_server *server = NULL;
    if (!qwen_server_create(&server, weights, tokenizer, "h3_shaders.metal",
                            "minimax-h3", 0, error, sizeof(error)))
        fail(error);

    serve_state state = {server, 0};
    pthread_t thread;
    if (pthread_create(&thread, NULL, serve_main, &state) != 0)
        fail("pthread_create");
    uint16_t port = 0;
    for (int waited = 0; waited < 20000 && !port; waited += 20) {
        port = atomic_load(&state.port);
        if (!port) usleep(20000);
    }
    require(port != 0, "server did not bind");

    /* 1. create -- must return 202 quickly, not after generation. */
    char *req = build_json_post(
        "/v1/videos",
        "{\"model\":\"h3-video\",\"prompt\":\"A cat walking through Osaka at "
        "night\",\"size\":\"256x256\",\"seed\":1234}");
    double t0 = now_seconds();
    size_t total = 0;
    uint8_t *response = http_roundtrip(port, req, &total);
    double create_s = now_seconds() - t0;
    free(req);
    require(strstr((char *)response, "HTTP/1.1 202") != NULL,
            "video create returns 202");
    require(create_s < 15.0, "202 comes back before generation finishes");
    require(strstr((char *)response, "\"status\":\"queued\"") != NULL,
            "new job is queued");
    char id[64];
    json_string_field((char *)response, "id", id, sizeof(id));
    require(id[0] != '\0', "response carries a job id");
    free(response);
    printf("(1) POST /v1/videos -> 202 in %.2fs, id %s\n", create_s, id);

    /* 2. status is observable; 3. content is 409 before completion. */
    char path[128];
    snprintf(path, sizeof(path), "/v1/videos/%s", id);
    req = build_get(path);
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL, "status is 200");
    require(strstr((char *)response, "\"status\":\"queued\"") ||
                strstr((char *)response, "\"status\":\"running\""),
            "status is queued or running");
    free(response);

    snprintf(path, sizeof(path), "/v1/videos/%s/content", id);
    req = build_get(path);
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 409") != NULL,
            "content before completion is 409");
    free(response);
    printf("(2) status observable; early /content -> 409\n");

    /* 4. unknown id -> 404. */
    req = build_get("/v1/videos/job-nope");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 404") != NULL,
            "unknown id -> 404");
    free(response);

    /* 5. chat still answers while the job runs. */
    req = build_json_post(
        "/v1/chat/completions",
        "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":4,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "chat answers during video generation");
    require(strstr((char *)response, "\"content\":\"\"") == NULL,
            "chat answer is non-empty");
    free(response);
    printf("(3) chat/completions answered while the job was in flight\n");

    /* 6. poll to completion. */
    int completed = 0;
    for (int waited = 0; waited < 600 && !completed; waited++) {
        snprintf(path, sizeof(path), "/v1/videos/%s", id);
        req = build_get(path);
        response = http_roundtrip(port, req, &total);
        free(req);
        if (strstr((char *)response, "\"status\":\"completed\""))
            completed = 1;
        else if (strstr((char *)response, "\"status\":\"failed\"")) {
            fprintf(stderr, "%s\n", (char *)response);
            fail("video job failed");
        }
        free(response);
        if (!completed) sleep(1);
    }
    require(completed, "video job completed within the timeout");
    printf("(4) job reached completed\n");

    /* 7. fetch the MP4. */
    snprintf(path, sizeof(path), "/v1/videos/%s/content", id);
    req = build_get(path);
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "content after completion is 200");
    require(strstr((char *)response, "video/mp4") != NULL,
            "content type is video/mp4");
    size_t body_len = 0;
    const uint8_t *body = find_body(response, total, &body_len);
    require(body && body_len > 1024, "MP4 body is non-trivial");
    require(!memcmp(body + 4, "ftyp", 4), "body looks like an MP4");
    FILE *f = fopen("/tmp/h3_p8_vid.mp4", "wb");
    require(f && fwrite(body, 1, body_len, f) == body_len, "save mp4");
    fclose(f);
    free(response);

    int w = 0, h = 0;
    require(h3_ffprobe_visual_size("/tmp/h3_p8_vid.mp4", &w, &h, error,
                                   sizeof(error)),
            error);
    require(w == 256 && h == 256, "video is 256x256");
    require(has_audio_stream("/tmp/h3_p8_vid.mp4"), "video has an audio track");
    printf("(5) /content -> %zu byte 256x256 MP4 with audio\n", body_len);

    qwen_server_stop(server);
    size_t drain = 0;
    free(http_roundtrip(port, "GET / HTTP/1.1\r\nHost: x\r\nConnection: "
                              "close\r\n\r\n",
                        &drain));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    unlink("/tmp/h3_p8_vid.mp4");
    puts("ok: P8-VID-02 async video HTTP lifecycle");
    return 0;
}
