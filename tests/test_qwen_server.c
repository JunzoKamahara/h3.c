/* Phase 4 test -- JSON parser, the HTTP server in isolation, and the
 * OpenAI-compatible endpoints end to end.
 *
 *   1. h3_json parse / accessors / escape;
 *   2. h3_http loopback: a stub handler answers a GET over a real socket;
 *   3. GET /v1/models returns the model id;
 *   4. POST /v1/chat/completions (buffered) returns a chat.completion with
 *      non-empty content and a finish_reason;
 *   5. POST /v1/chat/completions (stream) emits SSE data: chunks and [DONE].
 *
 *   ./h3_qwen_server_test MiniMax-H3
 */

#include "h3_http.h"
#include "h3_json.h"
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

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_server.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

/* -------- 1. JSON -------- */

static void test_json(void) {
    char error[256];
    const char *doc =
        "{ \"model\": \"m\", \"n\": -2.5, \"stream\": true, \"skip\": null,"
        " \"messages\": [ {\"role\":\"user\",\"content\":\"hi\\n\\\"x\\\"\"} ] }";
    h3_json *root = h3_json_parse(doc, strlen(doc), error, sizeof(error));
    require(root && h3_json_is(root, H3_JSON_OBJECT), "object parse failed");
    require(!strcmp(h3_json_string_value(h3_json_object_get(root, "model")),
                    "m"),
            "string field");
    require(h3_json_number_or(h3_json_object_get(root, "n"), 0) == -2.5,
            "number field");
    require(h3_json_bool_or(h3_json_object_get(root, "stream"), 0) == 1,
            "bool field");
    require(h3_json_is(h3_json_object_get(root, "skip"), H3_JSON_NULL),
            "null field");
    const h3_json *messages = h3_json_object_get(root, "messages");
    require(h3_json_array_size(messages) == 1, "array size");
    const h3_json *first = h3_json_array_at(messages, 0);
    require(!strcmp(h3_json_string_value(h3_json_object_get(first, "content")),
                    "hi\n\"x\""),
            "string escape decode");
    h3_json_free(root);

    require(h3_json_parse("{bad", 4, error, sizeof(error)) == NULL,
            "malformed JSON must fail");

    char *escaped = h3_json_escape("a\"b\nc\\d");
    require(escaped && !strcmp(escaped, "a\\\"b\\nc\\\\d"), "json escape");
    free(escaped);
    printf("(1) h3_json parse / accessors / escape ok\n");
}

/* -------- 2. HTTP loopback with a stub handler -------- */

typedef struct {
    h3_http_server *server;
    _Atomic uint16_t port;
    h3_http_handler handler;
} serve_thread;

static void *serve_main(void *opaque) {
    serve_thread *state = opaque;
    char error[256];
    state->server = h3_http_listen("127.0.0.1", 0, error, sizeof(error));
    if (!state->server) {
        fprintf(stderr, "listen: %s\n", error);
        return NULL;
    }
    atomic_store(&state->port, h3_http_server_port(state->server));
    h3_http_run(state->server, state->handler, NULL);
    return NULL;
}

static void stub_handler(const h3_http_request *request,
                         h3_http_responder *responder, void *user) {
    (void)user;
    if (!strcmp(request->path, "/ping"))
        h3_http_send(responder, 200, "text/plain", "pong", 4);
    else
        h3_http_send(responder, 404, "text/plain", "no", 2);
}

/* Send one request, return the full raw response (caller frees). */
static char *http_roundtrip(uint16_t port, const char *raw_request) {
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

    size_t capacity = 4096, length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) fail("client buffer");
    for (;;) {
        if (length + 1 >= capacity) {
            capacity *= 2;
            char *grown = realloc(buffer, capacity);
            if (!grown) fail("client buffer grow");
            buffer = grown;
        }
        ssize_t got = read(fd, buffer + length, capacity - length - 1);
        if (got <= 0) break;
        length += (size_t)got;
    }
    buffer[length] = '\0';
    close(fd);
    return buffer;
}

static char *b64enc(const uint8_t *data, size_t n) {
    static const char A[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *out = malloc((n + 2) / 3 * 4 + 1);
    if (!out) fail("b64 alloc");
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t x = (uint32_t)data[i] << 16;
        if (i + 1 < n) x |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < n) x |= data[i + 2];
        out[o++] = A[(x >> 18) & 63];
        out[o++] = A[(x >> 12) & 63];
        out[o++] = (i + 1 < n) ? A[(x >> 6) & 63] : '=';
        out[o++] = (i + 2 < n) ? A[x & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    uint8_t *b = malloc((size_t)sz);
    if (b && fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        b = NULL;
    }
    fclose(f);
    if (b) *n = (size_t)sz;
    return b;
}

static uint16_t start_stub(pthread_t *thread, serve_thread *state) {
    state->handler = stub_handler;
    atomic_store(&state->port, 0);
    if (pthread_create(thread, NULL, serve_main, state) != 0)
        fail("pthread_create");
    for (int wait_ms = 0; wait_ms < 5000; wait_ms += 20) {
        if (atomic_load(&state->port)) break;
        usleep(20000);
    }
    uint16_t port = atomic_load(&state->port);
    require(port != 0, "stub server did not bind");
    return port;
}

static void test_http(void) {
    pthread_t thread;
    serve_thread state = {0};
    uint16_t port = start_stub(&thread, &state);

    char *response = http_roundtrip(
        port, "GET /ping HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    require(strstr(response, "HTTP/1.1 200") != NULL, "stub 200 status");
    require(strstr(response, "\r\n\r\npong") != NULL, "stub body");
    free(response);

    response = http_roundtrip(
        port, "GET /nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    require(strstr(response, "HTTP/1.1 404") != NULL, "stub 404 status");
    free(response);

    h3_http_stop(state.server);
    /* nudge the accept loop out of blocking */
    free(http_roundtrip(port,
                        "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"));
    pthread_join(thread, NULL);
    h3_http_close(state.server);
    printf("(2) h3_http loopback ok (port %u)\n", port);
}

/* -------- 3-5. OpenAI endpoints -------- */

typedef struct {
    qwen_server *server;
    _Atomic uint16_t port;
} qwen_thread;

static void *qwen_serve_main(void *opaque) {
    qwen_thread *state = opaque;
    char error[512];
    /* qwen_server_run writes the bound port before entering the accept loop;
     * on arm64 the aligned 16-bit store is seen atomically by the test. */
    if (!qwen_server_run(state->server, "127.0.0.1", 0,
                         (uint16_t *)&state->port, error, sizeof(error)))
        fprintf(stderr, "qwen_server_run: %s\n", error);
    return NULL;
}

static void test_openai(const char *model_root) {
    char error[512];
    char weights[1024], tokenizer[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    snprintf(tokenizer, sizeof(tokenizer),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);

    qwen_server *server = NULL;
    if (!qwen_server_create(&server, weights, tokenizer, "h3_shaders.metal",
                            "minimax-h3", 0, error, sizeof(error)))
        fail(error);

    qwen_thread state = {server, 0};
    pthread_t thread;
    if (pthread_create(&thread, NULL, qwen_serve_main, &state) != 0)
        fail("pthread_create");
    uint16_t port = 0;
    for (int wait_ms = 0; wait_ms < 20000 && !port; wait_ms += 20) {
        port = atomic_load(&state.port);
        if (!port) usleep(20000);
    }
    require(port != 0, "qwen server did not bind");

    /* 3. models */
    char *response = http_roundtrip(
        port, "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    require(strstr(response, "HTTP/1.1 200") != NULL, "/v1/models status");
    require(strstr(response, "\"minimax-h3\"") != NULL, "/v1/models id");
    free(response);
    printf("(3) GET /v1/models ok\n");

    /* 4. buffered chat completion */
    const char *body =
        "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":2,"
        "\"messages\":[{\"role\":\"system\",\"content\":\"One word.\"},"
        "{\"role\":\"user\",\"content\":\"Capital of France?\"}]}";
    char request[1024];
    snprintf(request, sizeof(request),
             "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
             "Content-Type: application/json\r\nContent-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(body), body);
    response = http_roundtrip(port, request);
    require(strstr(response, "HTTP/1.1 200") != NULL, "chat status");
    require(strstr(response, "\"object\":\"chat.completion\"") != NULL,
            "chat object");
    require(strstr(response, "\"finish_reason\"") != NULL, "chat finish_reason");
    require(strstr(response, "\"content\":\"") != NULL, "chat has content");
    require(strstr(response, "\"content\":\"\"") == NULL,
            "chat content is non-empty");
    printf("(4) POST /v1/chat/completions (buffered) ok\n");
    free(response);

    /* 5. streaming */
    const char *sbody =
        "{\"stream\":true,\"max_tokens\":2,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}]}";
    snprintf(request, sizeof(request),
             "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
             "Content-Type: application/json\r\nContent-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(sbody), sbody);
    response = http_roundtrip(port, request);
    require(strstr(response, "text/event-stream") != NULL, "sse content-type");
    require(strstr(response, "data: ") != NULL, "sse data lines");
    require(strstr(response, "chat.completion.chunk") != NULL, "sse chunk obj");
    require(strstr(response, "data: [DONE]") != NULL, "sse [DONE]");
    printf("(5) POST /v1/chat/completions (stream) ok\n");
    free(response);

    /* 6. image input (P7-004): a data: URI image_url content part. */
    if (system("ffmpeg -y -v error -f lavfi -i "
               "smptebars=size=128x128:rate=1 -frames:v 1 "
               "/tmp/h3_srv_vlm.png") == 0) {
        size_t png_n = 0;
        uint8_t *png = slurp("/tmp/h3_srv_vlm.png", &png_n);
        require(png && png_n > 0, "read test png");
        char *b64 = b64enc(png, png_n);
        free(png);
        size_t body_cap = strlen(b64) + 512;
        char *mmbody = malloc(body_cap);
        char *mmreq = malloc(body_cap + 256);
        require(mmbody && mmreq, "mm alloc");
        snprintf(mmbody, body_cap,
                 "{\"model\":\"minimax-h3\",\"max_tokens\":24,\"messages\":[{"
                 "\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":"
                 "\"What is in this image?\"},{\"type\":\"image_url\","
                 "\"image_url\":{\"url\":\"data:image/png;base64,%s\"}}]}]}",
                 b64);
        free(b64);
        snprintf(mmreq, body_cap + 256,
                 "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                 "Connection: close\r\n\r\n%s",
                 strlen(mmbody), mmbody);
        response = http_roundtrip(port, mmreq);
        require(strstr(response, "HTTP/1.1 200") != NULL, "vlm status");
        require(strstr(response, "\"object\":\"chat.completion\"") != NULL,
                "vlm object");
        require(strstr(response, "\"content\":\"\"") == NULL,
                "vlm answer non-empty");
        free(response);
        free(mmbody);
        free(mmreq);

        /* a remote URL must be rejected, not fetched. */
        const char *hbody =
            "{\"model\":\"minimax-h3\",\"messages\":[{\"role\":\"user\","
            "\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":"
            "\"http://example.com/x.png\"}}]}]}";
        char hreq[512];
        snprintf(hreq, sizeof(hreq),
                 "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                 "Connection: close\r\n\r\n%s",
                 strlen(hbody), hbody);
        response = http_roundtrip(port, hreq);
        require(strstr(response, "HTTP/1.1 4") != NULL, "remote url rejected");
        require(strstr(response, "--allow-remote-images") != NULL,
                "remote url error mentions the flag");
        free(response);
        printf("(6) POST /v1/chat/completions (image_url data URI) ok\n");

        /* 7. /v1/responses input_image + detail. */
        b64 = b64enc((png = slurp("/tmp/h3_srv_vlm.png", &png_n)), png_n);
        free(png);
        body_cap = strlen(b64) + 512;
        mmbody = malloc(body_cap);
        mmreq = malloc(body_cap + 256);
        require(mmbody && mmreq, "resp mm alloc");
        snprintf(mmbody, body_cap,
                 "{\"model\":\"minimax-h3\",\"max_output_tokens\":24,\"input\":"
                 "[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\","
                 "\"text\":\"Describe it.\"},{\"type\":\"input_image\","
                 "\"image_url\":\"data:image/png;base64,%s\",\"detail\":"
                 "\"low\"}]}]}",
                 b64);
        free(b64);
        snprintf(mmreq, body_cap + 256,
                 "POST /v1/responses HTTP/1.1\r\nHost: x\r\n"
                 "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                 "Connection: close\r\n\r\n%s",
                 strlen(mmbody), mmbody);
        response = http_roundtrip(port, mmreq);
        require(strstr(response, "HTTP/1.1 200") != NULL, "resp vlm status");
        require(strstr(response, "\"output_text\":\"\"") == NULL,
                "resp vlm answer non-empty");
        free(response);
        free(mmbody);
        free(mmreq);
        printf("(7) POST /v1/responses (input_image + detail) ok\n");
    } else {
        printf("(6/7) skipped -- ffmpeg not available\n");
    }

    qwen_server_stop(server);
    free(http_roundtrip(port,
                        "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"));
    pthread_join(thread, NULL);
    qwen_server_free(server);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    test_json();
    test_http();
    test_openai(model_root);
    puts("ok: qwen Phase 4 server (json + http + /v1/models + "
         "/v1/chat/completions buffered & stream)");
    return 0;
}
