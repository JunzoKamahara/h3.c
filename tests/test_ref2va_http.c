/* P10-REF2VA-02/03/04 (slow): the async video HTTP lifecycle with a
 * reference image, a reference video, and a reference image + audio
 * together, end to end -- POST /v1/videos with "reference_image" /
 * "reference_video" / "reference_audio" as data: URIs, the same fields
 * OpenAI's `image_url` already accepts in chat.
 *
 *   ./h3_ref2va_http_test MiniMax-H3
 *
 * P10-REF2VA-00/01/02/04 already proved (at the CLI level, then at the
 * job-manager level) that swapping a reference measurably changes the
 * output; this test's job is narrower -- prove the JSON parsing / data: URI
 * resolution / job-wiring code produces a real, non-degenerate video
 * through the actual HTTP surface, for each reference shape the surface
 * accepts. One run per shape, not a comparison.
 *
 * Boots a full server (resident chat weights) and generates three real
 * reference-conditioned clips. Not in `make test`.
 */

#include "h3_ffmpeg.h"
#include "qwen_server.h"

#include <arpa/inet.h>
#include <math.h>
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
    fprintf(stderr, "FAIL tests/test_ref2va_http.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

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

static int has_audio_stream(const char *path) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams a:0 -show_entries "
             "stream=codec_type -of csv=p=0 '%s' | grep -q audio",
             path);
    return system(cmd) == 0;
}

static char *b64_encode(const uint8_t *data, size_t len) {
    static const char A[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *out = malloc((len + 2) / 3 * 4 + 1);
    if (!out) fail("alloc");
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out[o++] = A[(v >> 18) & 63];
        out[o++] = A[(v >> 12) & 63];
        out[o++] = i + 1 < len ? A[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < len ? A[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

static uint8_t *read_whole(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    require(f != NULL, "open file");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *bytes = malloc((size_t)size);
    require(bytes != NULL, "alloc file bytes");
    require(fread(bytes, 1, (size_t)size, f) == (size_t)size, "read file");
    fclose(f);
    *out_size = (size_t)size;
    return bytes;
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

/* POST /v1/videos with one or two reference fields as data: URIs, poll to
 * completion, fetch /content, and require a real, non-degenerate video.
 * `field2_name` (and `mime2`/`media2`/`media2_len`) are optional -- NULL
 * omits the second field, for the plain image-or-video-only case. */
static void run_reference_job(uint16_t port, const char *field_name,
                              const char *mime, const uint8_t *media,
                              size_t media_len, const char *field2_name,
                              const char *mime2, const uint8_t *media2,
                              size_t media2_len, const char *label) {
    char error[512];
    char *media_b64 = b64_encode(media, media_len);
    char *media2_b64 = field2_name ? b64_encode(media2, media2_len) : NULL;

    size_t body_cap = strlen(media_b64) + strlen(field_name) + strlen(mime) +
                      (media2_b64 ? strlen(media2_b64) + strlen(field2_name) +
                                    strlen(mime2) : 0) +
                      256;
    char *body = malloc(body_cap);
    require(body != NULL, "alloc request body");
    if (media2_b64)
        snprintf(body, body_cap,
                "{\"model\":\"h3-video\",\"prompt\":\"A calm still scene.\","
                "\"%s\":\"data:%s;base64,%s\","
                "\"%s\":\"data:%s;base64,%s\"}",
                field_name, mime, media_b64, field2_name, mime2, media2_b64);
    else
        snprintf(body, body_cap,
                "{\"model\":\"h3-video\",\"prompt\":\"A calm still scene.\","
                "\"%s\":\"data:%s;base64,%s\"}",
                field_name, mime, media_b64);
    free(media_b64);
    free(media2_b64);
    size_t req_cap = strlen(body) + 256;
    char *req = malloc(req_cap);
    require(req != NULL, "alloc request");
    snprintf(req, req_cap,
             "POST /v1/videos HTTP/1.1\r\nHost: x\r\nContent-Type: "
             "application/json\r\nContent-Length: %zu\r\nConnection: "
             "close\r\n\r\n%s",
             strlen(body), body);
    free(body);

    double t0 = now_seconds();
    size_t total = 0;
    uint8_t *response = http_roundtrip(port, req, strlen(req), &total);
    double create_s = now_seconds() - t0;
    free(req);
    require(strstr((char *)response, "HTTP/1.1 202") != NULL,
            "video create with a reference returns 202");
    require(create_s < 15.0, "202 comes back before generation finishes");
    char id[64];
    json_string_field((char *)response, "id", id, sizeof(id));
    require(id[0] != '\0', "response carries a job id");
    free(response);
    printf("(%s.1) POST /v1/videos (%s) -> 202 in %.2fs, id %s\n", label,
          field_name, create_s, id);

    char path[128];
    int completed = 0;
    for (int waited = 0; waited < 900 && !completed; waited++) {
        snprintf(path, sizeof(path), "/v1/videos/%s", id);
        size_t glen = strlen(path) + 96;
        char *greq = malloc(glen);
        require(greq != NULL, "alloc get request");
        snprintf(greq, glen,
                "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                path);
        response = http_roundtrip(port, greq, strlen(greq), &total);
        free(greq);
        if (strstr((char *)response, "\"status\":\"completed\""))
            completed = 1;
        else if (strstr((char *)response, "\"status\":\"failed\"")) {
            fprintf(stderr, "%s\n", (char *)response);
            fail("reference-conditioned video job failed");
        }
        free(response);
        if (!completed) sleep(1);
    }
    require(completed, "video job completed within the timeout");
    printf("(%s.2) job reached completed\n", label);

    snprintf(path, sizeof(path), "/v1/videos/%s/content", id);
    size_t glen = strlen(path) + 96;
    char *greq = malloc(glen);
    require(greq != NULL, "alloc get request");
    snprintf(greq, glen, "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
            path);
    response = http_roundtrip(port, greq, strlen(greq), &total);
    free(greq);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "content after completion is 200");
    require(strstr((char *)response, "video/mp4") != NULL,
            "content type is video/mp4");
    size_t body_len = 0;
    const uint8_t *mp4_body = find_body(response, total, &body_len);
    require(mp4_body && body_len > 1024, "MP4 body is non-trivial");
    require(!memcmp(mp4_body + 4, "ftyp", 4), "body looks like an MP4");
    char out_path[128];
    snprintf(out_path, sizeof(out_path), "/tmp/h3_ref2va_http_%s.mp4", label);
    FILE *outf = fopen(out_path, "wb");
    require(outf && fwrite(mp4_body, 1, body_len, outf) == body_len,
           "save mp4");
    fclose(outf);
    free(response);

    int w = 0, h = 0;
    require(h3_ffprobe_visual_size(out_path, &w, &h, error, sizeof(error)),
           error);
    require(w == 256 && h == 256, "video is 256x256");
    require(has_audio_stream(out_path), "video carries an audio track");

    float *pixels = NULL;
    int frames = 0;
    require(h3_ffmpeg_read_video_f32(out_path, 256, 256, 22, &pixels, &frames,
                                     error, sizeof(error)),
           error);
    size_t n = (size_t)3 * frames * 256 * 256;
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        require(isfinite(pixels[i]), "generated output has a non-finite pixel");
        mean += pixels[i];
    }
    mean /= (double)n;
    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)pixels[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;
    free(pixels);
    require(variance > 1e-4, "generated output is degenerate (near-constant)");
    printf("(%s.3) /content -> %zu byte 256x256 MP4 with audio, pixel "
          "variance %.6f\n",
          label, body_len, variance);
    unlink(out_path);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char weights[1024], tokenizer[1024], ref2va_probe[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    snprintf(tokenizer, sizeof(tokenizer),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);
    snprintf(ref2va_probe, sizeof(ref2va_probe), "%s/Ref2VA/transformer",
             model_root);
    struct stat st;
    if (stat(ref2va_probe, &st) != 0) {
        fprintf(stderr,
                "skip: Ref2VA transformer checkpoint is not installed\n");
        return 0;
    }

    /* Synthetic reference PNG (solid red). */
    char png_path[] = "/tmp/h3-ref2va-http-ref-XXXXXX.png";
    int fd = mkstemps(png_path, 4);
    require(fd >= 0, "mkstemps for reference PNG");
    close(fd);
    char error[512];
    {
        int w = 64, h = 64;
        uint8_t *pixels = malloc((size_t)w * h * 3);
        require(pixels != NULL, "alloc reference pixels");
        for (int i = 0; i < w * h; i++) {
            pixels[i * 3 + 0] = 220;
            pixels[i * 3 + 1] = 40;
            pixels[i * 3 + 2] = 40;
        }
        require(h3_ffmpeg_write_png_rgb24(png_path, pixels, w, h, error,
                                          sizeof(error)),
               error);
        free(pixels);
    }
    size_t png_size = 0;
    uint8_t *png_bytes = read_whole(png_path, &png_size);
    unlink(png_path);

    /* Synthetic reference clip (solid blue, 39 frames -- 2 Qwen vision
     * blocks, matching the job-level P10-REF2VA-02 gate). */
    char mp4_path[] = "/tmp/h3-ref2va-http-ref-XXXXXX.mp4";
    fd = mkstemps(mp4_path, 4);
    require(fd >= 0, "mkstemps for reference MP4");
    close(fd);
    {
        int w = 64, h = 64, frames = 39;
        size_t frame_bytes = (size_t)w * h * 3;
        uint8_t *pixels = malloc(frame_bytes * (size_t)frames);
        require(pixels != NULL, "alloc reference video pixels");
        for (size_t f = 0; f < (size_t)frames; f++)
            for (int i = 0; i < w * h; i++) {
                uint8_t *px = pixels + f * frame_bytes + (size_t)i * 3;
                px[0] = 40; px[1] = 60; px[2] = 220;
            }
        require(h3_ffmpeg_write_rgb24(mp4_path, pixels, frames, w, h, 24,
                                      error, sizeof(error)),
               error);
        free(pixels);
    }
    size_t mp4_size = 0;
    uint8_t *mp4_bytes = read_whole(mp4_path, &mp4_size);
    unlink(mp4_path);

    /* Synthetic reference audio (a 3 s, 220 Hz stereo tone). */
    char wav_path[] = "/tmp/h3-ref2va-http-ref-XXXXXX.wav";
    fd = mkstemps(wav_path, 4);
    require(fd >= 0, "mkstemps for reference WAV");
    close(fd);
    {
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                "ffmpeg -y -v error -f lavfi -i "
                "\"sine=frequency=220:duration=3:sample_rate=32000\" -ac 2 "
                "'%s'",
                wav_path);
        require(system(cmd) == 0,
               "ffmpeg could not synthesize a reference audio clip");
    }
    size_t wav_size = 0;
    uint8_t *wav_bytes = read_whole(wav_path, &wav_size);
    unlink(wav_path);

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

    run_reference_job(port, "reference_image", "image/png", png_bytes,
                      png_size, NULL, NULL, NULL, 0, "image");
    run_reference_job(port, "reference_video", "video/mp4", mp4_bytes,
                      mp4_size, NULL, NULL, NULL, 0, "video");
    run_reference_job(port, "reference_image", "image/png", png_bytes,
                      png_size, "reference_audio", "audio/wav", wav_bytes,
                      wav_size, "image+audio");
    free(png_bytes);
    free(mp4_bytes);
    free(wav_bytes);

    qwen_server_stop(server);
    size_t drain = 0;
    free(http_roundtrip(port,
                        "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                        strlen("GET / HTTP/1.1\r\nHost: x\r\nConnection: "
                              "close\r\n\r\n"),
                        &drain));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    puts("ok: P10-REF2VA-02/03/04 reference-conditioned video HTTP lifecycle "
        "(image + video + image+audio)");
    return 0;
}
