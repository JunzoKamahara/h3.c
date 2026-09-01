/* Unit test for the incremental tool-call streamer (no model). */

#include "qwen_stream.h"
#include "qwen_tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_stream.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

#define MAX_CALLS 8

typedef struct {
    char text[4096];
    char names[MAX_CALLS][128];
    char args[MAX_CALLS][2048];
    int begun[MAX_CALLS];
    int ended[MAX_CALLS];
    size_t call_count;
} recorder;

static void on_text(void *ctx, const char *delta) {
    strncat(((recorder *)ctx)->text, delta,
            sizeof(((recorder *)ctx)->text) - strlen(((recorder *)ctx)->text) - 1);
}
static void on_begin(void *ctx, size_t index, const char *id,
                     const char *name) {
    recorder *r = ctx;
    require(index < MAX_CALLS, "index");
    require(!r->begun[index], "begin fired twice");
    require(id && strlen(id) > 0, "begin id");
    r->begun[index] = 1;
    snprintf(r->names[index], sizeof(r->names[index]), "%s", name);
    if (index + 1 > r->call_count) r->call_count = index + 1;
}
static void on_args(void *ctx, size_t index, const char *id, const char *delta) {
    recorder *r = ctx;
    require(index < MAX_CALLS && r->begun[index], "args before begin");
    require(id && strlen(id) > 0, "args id");
    strncat(r->args[index], delta,
            sizeof(r->args[index]) - strlen(r->args[index]) - 1);
}
static void on_end(void *ctx, size_t index, const char *id) {
    recorder *r = ctx;
    require(index < MAX_CALLS && r->begun[index] && !r->ended[index], "end");
    require(id && strlen(id) > 0, "end id");
    r->ended[index] = 1;
}

/* Feed `target` one byte at a time. */
static void drip(const char *target, recorder *out) {
    memset(out, 0, sizeof(*out));
    qwen_stream_sink sink = {out, on_text, on_begin, on_args, on_end};
    qwen_stream *stream = qwen_stream_new(&sink);
    require(stream != NULL, "stream alloc");
    size_t total = strlen(target);
    char *buffer = calloc(total + 1, 1);
    require(buffer != NULL, "buffer");
    for (size_t i = 0; i < total; i++) {
        buffer[i] = target[i];
        require(qwen_stream_feed(stream, buffer), "feed");
    }
    qwen_stream_finish(stream);

    size_t final_count = 0;
    const h3_tool_call *calls = qwen_stream_calls(stream, &final_count);
    require(final_count == out->call_count, "calls() count vs begin count");
    for (size_t i = 0; i < final_count; i++) {
        require(!strcmp(calls[i].name, out->names[i]), "calls() name");
        require(!strcmp(calls[i].arguments, out->args[i]),
                "calls() arguments == concatenated arg deltas");
        require(out->ended[i], "every call ended");
    }
    free(buffer);
    qwen_stream_free(stream);
}

int main(void) {
    recorder r;

    /* 1. leading text + one call; whitespace before <tool_call> is dropped */
    drip("Let me check.\n<tool_call>\n{\"name\": \"get_weather\", "
         "\"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>",
         &r);
    require(!strcmp(r.text, "Let me check."), "1: leading text trimmed");
    require(r.call_count == 1, "1: one call");
    require(!strcmp(r.names[0], "get_weather"), "1: name");
    require(!strcmp(r.args[0], "{\"city\": \"Tokyo\"}"),
            "1: raw arguments value");
    printf("(1) text + single call ok\n");

    /* 2. parallel calls, no leading text */
    drip("<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n"
         "</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": "
         "{\"y\": 2}}\n</tool_call>",
         &r);
    require(!strcmp(r.text, ""), "2: no leading text");
    require(r.call_count == 2, "2: two calls");
    require(!strcmp(r.names[0], "a") && !strcmp(r.names[1], "b"), "2: names");
    require(!strcmp(r.args[0], "{\"x\": 1}"), "2: first args");
    require(!strcmp(r.args[1], "{\"y\": 2}"), "2: second args");
    printf("(2) parallel calls ok\n");

    /* 3. plain answer, no tool call */
    drip("The answer is 42.", &r);
    require(!strcmp(r.text, "The answer is 42."), "3: full text");
    require(r.call_count == 0, "3: no calls");
    printf("(3) plain text ok\n");

    /* 4. string-form arguments */
    drip("<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"n\\\":1}\"}\n"
         "</tool_call>",
         &r);
    require(r.call_count == 1 && !strcmp(r.names[0], "f"), "4: name");
    require(!strcmp(r.args[0], "\"{\\\"n\\\":1}\""),
            "4: raw string-form arguments streamed verbatim");
    printf("(4) string arguments ok\n");

    puts("ok: qwen_stream incremental tool-call splitter");
    return 0;
}
