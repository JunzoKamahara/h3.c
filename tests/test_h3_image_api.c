/* P8-IMG-01: POST /v1/images/generations end to end.
 *
 * Boots the server, requests one 256x256 image, fetches the PNG the server
 * saved, and checks it is a real, non-degenerate 256x256 image. Also checks
 * that an unsupported size is rejected before any generation work starts.
 *
 * Slow: this loads the FL2VA transformer (SSD streaming) and both VAEs, so it
 * is a standalone target, not part of `make test`.
 *
 *   ./h3_p8_img_test MiniMax-H3
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
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_image_api.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

/* Binary-safe HTTP round trip: returns the whole response and its length. */
static uint8_t *http_roundtrip(uint16_t port, const char *raw_request,
                               size_t *out_len) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("client socket");
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        fail("client connect");
    if (write(fd, raw_request, strlen(raw_request)) < 0) fail("client write");

    size_t capacity = 8192, length = 0;
    uint8_t *buffer = malloc(capacity);
    if (!buffer) fail("client buffer");
    for (;;) {
        if (length + 1 >= capacity) {
            capacity *= 2;
            uint8_t *grown = realloc(buffer, capacity);
            if (!grown) fail("client buffer grow");
            buffer = grown;
        }
        ssize_t got = read(fd, buffer + length, capacity - length - 1);
        if (got <= 0) break;
        length += (size_t)got;
    }
    buffer[length] = '\0';
    close(fd);
    *out_len = length;
    return buffer;
}

static const uint8_t *find_body(const uint8_t *response, size_t total,
                                size_t *body_len) {
    const char *marker = "\r\n\r\n";
    for (size_t i = 0; i + 4 <= total; i++)
        if (memcmp(response + i, marker, 4) == 0) {
            *body_len = total - (i + 4);
            return response + i + 4;
        }
    *body_len = 0;
    return NULL;
}

typedef struct {
    qwen_server *server;
    _Atomic uint16_t port;
} serve_state;

static void *serve_main(void *opaque) {
    serve_state *state = opaque;
    char error[512];
    if (!qwen_server_run(state->server, "127.0.0.1", 0,
                         (uint16_t *)&state->port, error, sizeof(error)))
        fprintf(stderr, "qwen_server_run: %s\n", error);
    return NULL;
}

static char *build_request(const char *method, const char *path,
                           const char *body) {
    size_t cap = (body ? strlen(body) : 0) + strlen(path) + 256;
    char *request = malloc(cap);
    if (!request) fail("request alloc");
    if (body)
        snprintf(request, cap,
                 "%s %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
                 method, path, strlen(body), body);
    else
        snprintf(request, cap,
                 "%s %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", method,
                 path);
    return request;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];
    char weights[1024], tokenizer[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    snprintf(tokenizer, sizeof(tokenizer), "%s/FL2VA/tokenizer/tokenizer.json",
             model_root);

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

    /* 1. an unsupported size is rejected up front. */
    {
        char *request = build_request(
            "POST", "/v1/images/generations",
            "{\"model\":\"h3-image\",\"prompt\":\"x\",\"size\":\"512x512\"}");
        size_t total = 0;
        uint8_t *response = http_roundtrip(port, request, &total);
        require(strstr((char *)response, "HTTP/1.1 4") != NULL,
                "512x512 must be rejected");
        require(strstr((char *)response, "256x256") != NULL,
                "rejection names the supported size");
        free(response);
        free(request);
        printf("(1) unsupported size rejected ok\n");
    }

    /* 2. one real 256x256 generation. */
    char image_path[256] = {0};
    {
        char *request = build_request(
            "POST", "/v1/images/generations",
            "{\"model\":\"h3-image\",\"prompt\":\"A red fox walking through "
            "snow\",\"size\":\"256x256\",\"n\":1,\"seed\":42}");
        size_t total = 0;
        uint8_t *response = http_roundtrip(port, request, &total);
        free(request);
        require(strstr((char *)response, "HTTP/1.1 200") != NULL,
                "image generation status");
        const char *tag = strstr((char *)response,
                                 "\"url\":\"/v1/generated/images/img-");
        require(tag != NULL, "response carries a generated-image url");
        const char *start = strchr(tag, '/');
        const char *end = strchr(start, '"');
        require(start && end && end > start, "url is quoted");
        size_t url_len = (size_t)(end - start);
        require(url_len < sizeof(image_path), "url fits");
        memcpy(image_path, start, url_len);
        image_path[url_len] = '\0';
        free(response);
        printf("(2) POST /v1/images/generations 200, url %s\n", image_path);
    }

    /* 3. fetch the PNG and check it is a real 256x256 image. */
    {
        char *request = build_request("GET", image_path, NULL);
        size_t total = 0;
        uint8_t *response = http_roundtrip(port, request, &total);
        free(request);
        require(strstr((char *)response, "HTTP/1.1 200") != NULL,
                "image fetch status");
        require(strstr((char *)response, "image/png") != NULL,
                "image fetch content-type");
        size_t body_len = 0;
        const uint8_t *body = find_body(response, total, &body_len);
        require(body && body_len > 8, "image fetch has a body");
        static const uint8_t png_magic[8] = {0x89, 'P', 'N', 'G',
                                             0x0d, 0x0a, 0x1a, 0x0a};
        require(memcmp(body, png_magic, 8) == 0, "body is a PNG");

        FILE *f = fopen("/tmp/h3_p8_img.png", "wb");
        require(f && fwrite(body, 1, body_len, f) == body_len, "save png");
        fclose(f);
        free(response);

        int w = 0, h = 0;
        require(h3_ffprobe_visual_size("/tmp/h3_p8_img.png", &w, &h, error,
                                       sizeof(error)),
                error);
        require(w == 256 && h == 256, "png is 256x256");

        float *pixels = NULL;
        require(h3_ffmpeg_read_image_f32("/tmp/h3_p8_img.png", 256, 256,
                                         H3_IMAGE_FIT_STRETCH, &pixels, error,
                                         sizeof(error)),
                error);
        double mean = 0.0;
        size_t count = (size_t)256 * 256 * 3;
        for (size_t i = 0; i < count; i++) mean += pixels[i];
        mean /= (double)count;
        double var = 0.0;
        for (size_t i = 0; i < count; i++) {
            double d = pixels[i] - mean;
            var += d * d;
        }
        var /= (double)count;
        free(pixels);
        require(var > 1e-4, "image is not a flat field");
        printf("(3) PNG is 256x256, non-degenerate (pixel variance %.4f)\n",
               var);
    }

    qwen_server_stop(server);
    size_t drain = 0;
    free(http_roundtrip(port, "GET / HTTP/1.1\r\nHost: x\r\nConnection: "
                              "close\r\n\r\n",
                        &drain));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    puts("ok: P8-IMG-01 /v1/images/generations end to end");
    return 0;
}
