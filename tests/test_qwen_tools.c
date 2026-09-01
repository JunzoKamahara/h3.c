/* Phase 5 test -- tool calling.
 *
 *   1. h3_json_stringify round-trip;
 *   2. qwen_tool_calls_parse: leading text + one/many <tool_call> blocks,
 *      object and string arguments, no-markup case;
 *   3. qwen_chat_render_tools: the `tools` system block and assistant
 *      tool_calls markup; no-tools render still matches Phase 3;
 *   4. end to end -- POST /v1/chat/completions with `tools` yields a
 *      tool_calls response with finish_reason "tool_calls".
 *
 *   ./h3_qwen_tools_test MiniMax-H3
 */

#include "h3_json.h"
#include "qwen_engine.h"
#include "qwen_server.h"
#include "qwen_tools.h"

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
    fprintf(stderr, "FAIL tests/test_qwen_tools.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void test_stringify(void) {
    char error[256];
    const char *doc = "{\"a\":1,\"b\":[true,\"x\",null],\"c\":{\"d\":\"e\"}}";
    h3_json *root = h3_json_parse(doc, strlen(doc), error, sizeof(error));
    require(root != NULL, "stringify: parse");
    char *out = h3_json_stringify(root);
    require(out && !strcmp(out, doc), "stringify: compact round-trip");
    free(out);
    h3_json_free(root);
    printf("(1) h3_json_stringify ok\n");
}

static void test_parse(void) {
    char error[256];
    h3_tool_call *calls = NULL;
    size_t count = 0;
    char *content = NULL;

    const char *one =
        "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", "
        "\"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>";
    require(qwen_tool_calls_parse(one, &calls, &count, &content, error,
                                  sizeof(error)),
            "parse one: ok");
    require(count == 1, "parse one: count");
    require(!strcmp(content, "Let me check."), "parse one: leading content");
    require(!strcmp(calls[0].name, "get_weather"), "parse one: name");
    require(!strcmp(calls[0].arguments, "{\"city\":\"Tokyo\"}"),
            "parse one: arguments compacted");
    require(!strcmp(calls[0].id, "call_0001"), "parse one: id");
    h3_tool_calls_free(calls, count);
    free(content);

    const char *two =
        "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"b\", \"arguments\": \"{\\\"x\\\":1}\"}\n"
        "</tool_call>";
    require(qwen_tool_calls_parse(two, &calls, &count, &content, error,
                                  sizeof(error)),
            "parse two: ok");
    require(count == 2, "parse two: count");
    require(!strcmp(content, ""), "parse two: empty content");
    require(!strcmp(calls[1].name, "b"), "parse two: second name");
    require(!strcmp(calls[1].arguments, "{\"x\":1}"),
            "parse two: string arguments passed through");
    require(!strcmp(calls[1].id, "call_0002"), "parse two: second id");
    h3_tool_calls_free(calls, count);
    free(content);

    require(qwen_tool_calls_parse("just a normal answer.", &calls, &count,
                                  &content, error, sizeof(error)),
            "parse none: ok");
    require(count == 0 && calls == NULL, "parse none: no calls");
    require(!strcmp(content, "just a normal answer."), "parse none: content");
    free(content);
    printf("(2) qwen_tool_calls_parse ok\n");
}

static void test_render(void) {
    char error[256], *text = NULL;

    qwen_chat_message plain[] = {{QWEN_ROLE_USER, "hi", NULL}};
    require(qwen_chat_render(plain, 1, 0, &text, error, sizeof(error)),
            "render plain");
    require(!strcmp(text, "<|im_start|>user\nhi<|im_end|>\n"),
            "no-tools render unchanged from Phase 3");
    free(text);

    const char *tool =
        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\"}}}";
    const char *tools[] = {tool};
    qwen_chat_message convo[] = {
        {QWEN_ROLE_SYSTEM, "You are helpful.", NULL},
        {QWEN_ROLE_USER, "Weather in Tokyo?", NULL},
    };
    require(qwen_chat_render_tools(convo, 2, tools, 1, 1, &text, error,
                                  sizeof(error)),
            "render tools");
    require(strstr(text, "<|im_start|>system\nYou are helpful.\n\n# Tools") !=
                NULL,
            "tools: system block prefix");
    require(strstr(text, "<tools>\n") && strstr(text, tool) &&
                strstr(text, "\n</tools>"),
            "tools: <tools> wraps the signature");
    require(strstr(text, "<tool_call></tool_call> XML tags") != NULL,
            "tools: instructions present");
    require(strstr(text, "</tool_call><|im_end|>\n<|im_start|>user\n") != NULL,
            "tools: system turn closes before the user turn");
    require(strstr(text, "<|im_start|>assistant\n") ==
                text + strlen(text) - strlen("<|im_start|>assistant\n"),
            "tools: ends with the generation prompt");
    free(text);

    qwen_chat_message with_calls[] = {
        {QWEN_ROLE_USER, "Weather?", NULL},
        {QWEN_ROLE_ASSISTANT, "",
         "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}]"},
        {QWEN_ROLE_TOOL, "{\"temp\":12}", NULL},
    };
    require(qwen_chat_render(with_calls, 3, 1, &text, error, sizeof(error)),
            "render assistant tool_calls");
    require(strstr(text,
                   "<|im_start|>assistant\n<tool_call>\n{\"name\": "
                   "\"get_weather\", \"arguments\": {\"city\":\"Tokyo\"}}\n"
                   "</tool_call><|im_end|>\n") != NULL,
            "assistant tool_calls markup");
    require(strstr(text, "<tool_response>\n{\"temp\":12}\n</tool_response>") !=
                NULL,
            "tool response still rendered");
    free(text);
    printf("(3) qwen_chat_render_tools ok\n");
}

/* -- minimal HTTP client + server thread (shared with other server tests) -- */

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

static char *http_roundtrip(uint16_t port, const char *request) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("client socket");
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        fail("client connect");
    if (write(fd, request, strlen(request)) < 0) fail("client write");
    size_t capacity = 8192, length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) fail("client buffer");
    for (;;) {
        if (length + 1 >= capacity) {
            capacity *= 2;
            char *grown = realloc(buffer, capacity);
            if (!grown) fail("client grow");
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

static void test_endpoint(const char *model_root) {
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

    const char *body =
        "{\"stream\":false,\"max_tokens\":64,"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_current_weather\","
        "\"description\":\"Get the current weather for a city\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"What is the current "
        "weather in Tokyo? Call the get_current_weather function.\"}]}";
    char request[2048];
    snprintf(request, sizeof(request),
             "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
             "Content-Type: application/json\r\nContent-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(body), body);
    char *response = http_roundtrip(port, request);
    require(strstr(response, "HTTP/1.1 200") != NULL, "tool call: status");
    require(strstr(response, "\"finish_reason\":\"tool_calls\"") != NULL,
            "tool call: finish_reason is tool_calls");
    require(strstr(response, "\"name\":\"get_current_weather\"") != NULL,
            "tool call: function name echoed");
    require(strstr(response, "\"tool_calls\":[{") != NULL,
            "tool call: tool_calls array present");
    printf("(4) POST /v1/chat/completions with tools -> tool_calls\n");
    const char *tc = strstr(response, "\"tool_calls\"");
    if (tc) {
        const char *end = strchr(tc, '}');
        printf("    %.*s...\n", (int)(end ? end - tc + 1 : 40), tc);
    }
    free(response);

    qwen_server_stop(server);
    free(http_roundtrip(port,
                        "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"));
    pthread_join(thread, NULL);
    qwen_server_free(server);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    test_stringify();
    test_parse();
    test_render();
    test_endpoint(model_root);
    puts("ok: qwen Phase 5 tool calling (parse + render + endpoint)");
    return 0;
}
