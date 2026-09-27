/* P10-PARAMS-01 (slow): POST /v1/videos with an explicit "seconds" produces
 * a clip with meaningfully more frames than the previous unconditional
 * default (H3_IMAGE_FRAMES = 5, about 0.2s) -- and omitting "seconds"
 * still produces exactly that same short default, proving the new
 * parameter is additive, not a behavior change for existing callers.
 *
 *   ./h3_video_seconds_test MiniMax-H3
 *
 * Boots a full server (resident chat weights) and generates two real
 * clips. Not in `make test`.
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
#include <unistd.h>

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_video_seconds.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static uint8_t *http_roundtrip(uint16_t port, const char *raw, size_t raw_len,
                               size_t *out_len) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("client socket");
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        fail("client connect");
    if (write(fd, raw, raw_len) < 0) fail("client write");
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

/* POST /v1/videos with an optional "seconds" clause, poll to completion,
 * fetch /content, and return the actual decoded frame count (capped at
 * max_frames). */
static int run_and_count_frames(uint16_t port, const char *seconds_clause,
                                const char *label, int max_frames) {
    char error[512];
    char body[256];
    snprintf(body, sizeof(body),
             "{\"model\":\"h3-video\",\"prompt\":\"A calm still scene.\","
             "\"seed\":9%s}",
             seconds_clause);
    char req[512];
    snprintf(req, sizeof(req),
             "POST /v1/videos HTTP/1.1\r\nHost: x\r\nContent-Type: "
             "application/json\r\nContent-Length: %zu\r\nConnection: "
             "close\r\n\r\n%s",
             strlen(body), body);
    size_t total = 0;
    uint8_t *response = http_roundtrip(port, req, strlen(req), &total);
    require(strstr((char *)response, "HTTP/1.1 202") != NULL,
            "video create returns 202");
    char id[64];
    json_string_field((char *)response, "id", id, sizeof(id));
    require(id[0] != '\0', "response carries a job id");
    free(response);
    printf("(%s.1) POST /v1/videos%s -> 202, id %s\n", label, seconds_clause,
          id);

    char path[128];
    int completed = 0;
    for (int waited = 0; waited < 900 && !completed; waited++) {
        snprintf(path, sizeof(path), "/v1/videos/%s", id);
        char greq[160];
        snprintf(greq, sizeof(greq),
                "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                path);
        response = http_roundtrip(port, greq, strlen(greq), &total);
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

    snprintf(path, sizeof(path), "/v1/videos/%s/content", id);
    char greq[128];
    snprintf(greq, sizeof(greq),
            "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", path);
    response = http_roundtrip(port, greq, strlen(greq), &total);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "content after completion is 200");
    size_t body_len = 0;
    const uint8_t *mp4_body = find_body(response, total, &body_len);
    require(mp4_body && body_len > 1024, "MP4 body is non-trivial");
    char out_path[128];
    snprintf(out_path, sizeof(out_path), "/tmp/h3_seconds_%s.mp4", label);
    FILE *outf = fopen(out_path, "wb");
    require(outf && fwrite(mp4_body, 1, body_len, outf) == body_len,
           "save mp4");
    fclose(outf);
    free(response);

    float *pixels = NULL;
    int frames = 0;
    require(h3_ffmpeg_read_video_f32(out_path, 256, 256, max_frames, &pixels,
                                     &frames, error, sizeof(error)),
           error);
    free(pixels);
    unlink(out_path);
    printf("(%s.2) decoded %d frame(s)\n", label, frames);
    return frames;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char weights[1024], tokenizer[1024], error[512];
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

    /* Default (no "seconds"): must reproduce the exact prior 5-frame clip,
     * proving the new parameter is opt-in, not a default-behavior change. */
    int default_frames = run_and_count_frames(port, "", "default", 5);
    require(default_frames == 5,
           "omitting \"seconds\" still yields the prior 5-frame default");

    /* An explicit 1 second request aligns up to the trained cadence but
     * must be dramatically longer than the untouched default. */
    int long_frames =
        run_and_count_frames(port, ",\"seconds\":1.0", "explicit", 64);
    require(long_frames >= 20,
           "\"seconds\":1.0 did not produce a meaningfully longer clip");

    printf("ok: default=%d frames, seconds=1.0 -> %d frames\n",
          default_frames, long_frames);

    qwen_server_stop(server);
    size_t drain = 0;
    free(http_roundtrip(port,
                        "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                        strlen("GET / HTTP/1.1\r\nHost: x\r\nConnection: "
                              "close\r\n\r\n"),
                        &drain));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    puts("ok: P10-PARAMS-01 \"seconds\" controls generated clip length");
    return 0;
}
