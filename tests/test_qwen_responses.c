/* Phase 6 test -- the OpenAI Responses API (POST /v1/responses).
 *
 *   1. buffered text: `instructions` + string `input` -> a `response` object
 *      with an assistant message item and `output_text`;
 *   2. array `input` (message objects) is accepted;
 *   3. tools: a function_call item with finish via `status: completed`;
 *   4. streaming: response.created / response.output_text.delta /
 *      response.completed events.
 *
 *   ./h3_qwen_responses_test MiniMax-H3
 */

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
    fprintf(stderr, "FAIL tests/test_qwen_responses.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
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

static char *http_post(uint16_t port, const char *body) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("socket");
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        fail("connect");
    char header[256];
    int n = snprintf(header, sizeof(header),
                     "POST /v1/responses HTTP/1.1\r\nHost: x\r\n"
                     "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                     "Connection: close\r\n\r\n",
                     strlen(body));
    if (write(fd, header, (size_t)n) < 0 || write(fd, body, strlen(body)) < 0)
        fail("write");
    size_t capacity = 8192, length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) fail("buffer");
    for (;;) {
        if (length + 1 >= capacity) {
            capacity *= 2;
            char *grown = realloc(buffer, capacity);
            if (!grown) fail("grow");
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

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512], weights[1024], tokenizer[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    snprintf(tokenizer, sizeof(tokenizer),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);

    qwen_server *server = NULL;
    if (!qwen_server_create(&server, weights, tokenizer, "h3_shaders.metal",
                            "minimax-h3", 1, error, sizeof(error)))
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

    /* 1. buffered text with instructions + string input */
    char *r = http_post(
        port,
        "{\"model\":\"minimax-h3\",\"max_output_tokens\":24,"
        "\"instructions\":\"Answer with a single short word.\","
        "\"input\":\"What is the capital of France?\"}");
    require(strstr(r, "HTTP/1.1 200") != NULL, "responses text: status");
    require(strstr(r, "\"object\":\"response\"") != NULL, "responses: object");
    require(strstr(r, "\"status\":\"completed\"") != NULL, "responses: status");
    require(strstr(r, "\"type\":\"message\"") != NULL, "responses: message item");
    require(strstr(r, "\"type\":\"output_text\"") != NULL,
            "responses: output_text part");
    require(strstr(r, "\"output_text\":\"\"") == NULL,
            "responses: output_text non-empty");
    require(strstr(r, "\"input_tokens\":") != NULL, "responses: usage");
    printf("(1) POST /v1/responses (buffered text) ok\n");
    free(r);

    /* 2. array input */
    r = http_post(port,
                  "{\"max_output_tokens\":16,\"input\":[{\"role\":\"user\","
                  "\"content\":\"Say hi in one word.\"}]}");
    require(strstr(r, "HTTP/1.1 200") && strstr(r, "\"object\":\"response\""),
            "responses array input: ok");
    printf("(2) POST /v1/responses (array input) ok\n");
    free(r);

    /* 3. tools -> function_call item */
    r = http_post(
        port,
        "{\"max_output_tokens\":64,"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_current_weather\",\"description\":\"weather\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],"
        "\"input\":\"Get the current weather in Tokyo with the "
        "get_current_weather function.\"}");
    require(strstr(r, "\"type\":\"function_call\"") != NULL,
            "responses tools: function_call item");
    require(strstr(r, "\"name\":\"get_current_weather\"") != NULL,
            "responses tools: name");
    require(strstr(r, "Tokyo") != NULL, "responses tools: arguments carry city");
    require(strstr(r, "\"id\":\"call_0001\"") != NULL,
            "responses tools: function_call id");
    require(strstr(r, "fc_\"") == NULL && strstr(r, "\"fc_") == NULL,
            "responses tools: no broken fc_ id token");
    /* the output array must be valid JSON: no "" run from a bad concat */
    require(strstr(r, "\"\"call_") == NULL,
            "responses tools: output JSON is well formed");
    printf("(3) POST /v1/responses (tools -> function_call) ok\n");
    free(r);

    /* 4. streaming events */
    r = http_post(port,
                  "{\"stream\":true,\"max_output_tokens\":8,"
                  "\"input\":\"Name a color.\"}");
    require(strstr(r, "text/event-stream") != NULL, "responses stream: ctype");
    require(strstr(r, "event: response.created") != NULL,
            "responses stream: created event");
    require(strstr(r, "event: response.output_text.delta") != NULL,
            "responses stream: delta event");
    require(strstr(r, "\"type\":\"response.output_text.delta\"") != NULL,
            "responses stream: delta payload type");
    require(strstr(r, "event: response.completed") != NULL,
            "responses stream: completed event");
    printf("(4) POST /v1/responses (streaming events) ok\n");
    free(r);

    qwen_server_stop(server);
    free(http_post(port, "{}"));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    puts("ok: qwen Phase 6 Responses API (/v1/responses buffered, array, "
         "tools, streaming)");
    return 0;
}
