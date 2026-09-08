/* P8-TOOL-01 (slow): the chat model calls the built-in `generate_video` tool,
 * the server submits an async job and feeds the job id back so the model
 * keeps talking, and the job is pollable through /v1/generations/{id} to a
 * real MP4. A second chat request still answers while the job runs.
 *
 *   ./h3_media_tools_test MiniMax-H3
 *
 * Boots a full server and generates one real clip. Not in `make test`
 * (phase4-check covers the routing / regression without generating).
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
#include <unistd.h>

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_h3_media_tools.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

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

static const uint8_t *find_body(const uint8_t *r, size_t total, size_t *bl) {
    for (size_t i = 0; i + 4 <= total; i++)
        if (!memcmp(r + i, "\r\n\r\n", 4)) {
            *bl = total - (i + 4);
            return r + i + 4;
        }
    *bl = 0;
    return NULL;
}

static char *post_json(const char *path, const char *body) {
    size_t cap = strlen(body) + strlen(path) + 200;
    char *r = malloc(cap);
    if (!r) fail("alloc");
    snprintf(r, cap,
             "POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
             path, strlen(body), body);
    return r;
}
static char *get_req(const char *path) {
    size_t cap = strlen(path) + 96;
    char *r = malloc(cap);
    if (!r) fail("alloc");
    snprintf(r, cap, "GET %s HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
             path);
    return r;
}

typedef struct {
    qwen_server *server;
    _Atomic uint16_t port;
} serve_state;
static void *serve_main(void *o) {
    serve_state *s = o;
    char e[512];
    if (!qwen_server_run(s->server, "127.0.0.1", 0, (uint16_t *)&s->port, e,
                         sizeof(e)))
        fprintf(stderr, "qwen_server_run: %s\n", e);
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

    /* 1. a chat turn that should call the built-in generate_video tool. */
    const char *body =
        "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":80,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Call the generate_video "
        "tool now to make a video of a red fox walking through snow.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
        "\"generate_video\"}}]}";
    char *req = post_json("/v1/chat/completions", body);
    size_t total = 0;
    uint8_t *response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL, "chat 200");
    /* The server executed the tool, so the reply is normal assistant text --
     * not an unresolved tool_calls finish. */
    require(strstr((char *)response, "\"finish_reason\":\"tool_calls\"") == NULL,
            "built-in tool call was resolved server-side");
    free(response);
    printf("(1) chat turn returned after the built-in tool ran\n");

    /* 2. the tool submitted job-00000001; it is pollable. */
    req = get_req("/v1/generations/job-00000001");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "generate_video submitted a job (job-00000001 exists)");
    require(strstr((char *)response, "\"object\":\"video\"") != NULL,
            "job is a video job");
    free(response);
    printf("(2) /v1/generations/job-00000001 is live\n");

    /* 3. chat still answers while the job runs. */
    req = post_json("/v1/chat/completions",
                    "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":4,"
                    "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL,
            "chat answers during generation");
    require(strstr((char *)response, "\"content\":\"\"") == NULL,
            "answer is non-empty");
    free(response);
    printf("(3) a second chat turn answered while the job was in flight\n");

    /* 3b. the model answers "is it ready?" via the get_generation_status tool
     * while the job is still running. */
    req = post_json(
        "/v1/chat/completions",
        "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":80,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Is generation job "
        "job-00000001 finished yet? Use the get_generation_status tool.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
        "\"get_generation_status\"}}]}");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL, "status chat 200");
    require(strstr((char *)response, "\"finish_reason\":\"tool_calls\"") == NULL,
            "get_generation_status resolved server-side");
    require(strstr((char *)response, "\"content\":\"\"") == NULL,
            "status answer non-empty");
    free(response);
    printf("(3b) get_generation_status answered mid-job\n");

    /* 4. poll to completion and fetch the MP4. */
    int completed = 0;
    for (int waited = 0; waited < 600 && !completed; waited++) {
        req = get_req("/v1/generations/job-00000001");
        response = http_roundtrip(port, req, &total);
        free(req);
        if (strstr((char *)response, "\"status\":\"completed\"")) completed = 1;
        else if (strstr((char *)response, "\"status\":\"failed\"")) {
            fprintf(stderr, "%s\n", (char *)response);
            fail("video job failed");
        }
        free(response);
        if (!completed) sleep(1);
    }
    require(completed, "job completed within the timeout");

    req = get_req("/v1/generations/job-00000001/content");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL, "content 200");
    require(strstr((char *)response, "video/mp4") != NULL, "content-type mp4");
    size_t bl = 0;
    const uint8_t *b = find_body(response, total, &bl);
    require(b && bl > 1024 && !memcmp(b + 4, "ftyp", 4), "body is an MP4");
    FILE *f = fopen("/tmp/h3_tool_vid.mp4", "wb");
    require(f && fwrite(b, 1, bl, f) == bl, "save mp4");
    fclose(f);
    free(response);
    int w = 0, h = 0;
    require(h3_ffprobe_visual_size("/tmp/h3_tool_vid.mp4", &w, &h, error,
                                   sizeof(error)),
            error);
    require(w == 256 && h == 256, "video is 256x256");
    printf("(4) job completed -> 256x256 MP4 via /v1/generations/{id}/content\n");

    /* 5. after completion, get_generation_status reports it and the model can
     * hand the user a content link. */
    req = post_json(
        "/v1/chat/completions",
        "{\"model\":\"minimax-h3\",\"stream\":false,\"max_tokens\":80,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Is job-00000001 done "
        "now? Use get_generation_status and tell me where to get it.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
        "\"get_generation_status\"}}]}");
    response = http_roundtrip(port, req, &total);
    free(req);
    require(strstr((char *)response, "HTTP/1.1 200") != NULL, "final status 200");
    require(strstr((char *)response, "\"content\":\"\"") == NULL,
            "final status answer non-empty");
    free(response);
    printf("(5) get_generation_status reports completion after the job\n");

    /* 6. MCP facade: generate_video with the tasks opt-in -> an MCP task,
     * pollable with tasks/get to completed + a result carrying content_url. */
    {
        char *m = post_json(
            "/mcp",
            "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":"
            "{\"name\":\"generate_video\",\"arguments\":{\"prompt\":\"a lantern "
            "drifting on a river\",\"seed\":7},\"_meta\":{"
            "\"io.modelcontextprotocol/tasks\":{}}}}");
        response = http_roundtrip(port, m, &total);
        free(m);
        require(strstr((char *)response, "\"result\"") != NULL, "mcp result");
        require(strstr((char *)response, "\"status\":\"working\"") != NULL,
                "generate_video returned a working task");
        char *tp = strstr((char *)response, "\"taskId\":\"");
        require(tp != NULL, "task carries a taskId");
        tp += strlen("\"taskId\":\"");
        char task_id[64];
        size_t k = 0;
        while (tp[k] && tp[k] != '"' && k < sizeof(task_id) - 1) {
            task_id[k] = tp[k];
            k++;
        }
        task_id[k] = '\0';
        require(k == 32, "taskId is 128-bit hex (not the internal job id)");
        free(response);
        printf("(6) MCP generate_video -> working task %s\n", task_id);

        int mcp_done = 0;
        for (int waited = 0; waited < 600 && !mcp_done; waited++) {
            char gbody[160];
            snprintf(gbody, sizeof(gbody),
                     "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tasks/get\","
                     "\"params\":{\"taskId\":\"%s\"}}",
                     task_id);
            m = post_json("/mcp", gbody);
            response = http_roundtrip(port, m, &total);
            free(m);
            if (strstr((char *)response, "\"status\":\"completed\""))
                mcp_done = 1;
            free(response);
            if (!mcp_done) sleep(1);
        }
        require(mcp_done, "MCP task reached completed");

        char gbody[160];
        snprintf(gbody, sizeof(gbody),
                 "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tasks/get\","
                 "\"params\":{\"taskId\":\"%s\"}}",
                 task_id);
        m = post_json("/mcp", gbody);
        response = http_roundtrip(port, m, &total);
        free(m);
        require(strstr((char *)response, "content_url") != NULL,
                "completed MCP task result carries a content_url");
        require(strstr((char *)response, "\"isError\":true") == NULL,
                "a successful generation is not an error");
        free(response);
        printf("(7) MCP tasks/get -> completed with a content_url\n");
    }

    qwen_server_stop(server);
    size_t drain = 0;
    free(http_roundtrip(port, "GET / HTTP/1.1\r\nHost: x\r\nConnection: "
                              "close\r\n\r\n",
                        &drain));
    pthread_join(thread, NULL);
    qwen_server_free(server);
    unlink("/tmp/h3_tool_vid.mp4");
    puts("ok: P8-TOOL-01 built-in generate_video tool");
    return 0;
}
