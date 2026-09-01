#include "qwen_server.h"

#include "h3_http.h"
#include "h3_json.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ strings */

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} strbuf;

static void strbuf_free(strbuf *sb) {
    free(sb->data);
    memset(sb, 0, sizeof(*sb));
}

static void strbuf_append(strbuf *sb, const char *text) {
    if (sb->failed || !text) return;
    size_t add = strlen(text);
    if (sb->length + add + 1 > sb->capacity) {
        size_t capacity = sb->capacity ? sb->capacity : 256;
        while (capacity < sb->length + add + 1) capacity *= 2;
        char *grown = realloc(sb->data, capacity);
        if (!grown) {
            sb->failed = 1;
            return;
        }
        sb->data = grown;
        sb->capacity = capacity;
    }
    memcpy(sb->data + sb->length, text, add + 1);
    sb->length += add;
}

static void strbuf_append_json_string(strbuf *sb, const char *raw) {
    char *escaped = h3_json_escape(raw ? raw : "");
    if (!escaped) {
        sb->failed = 1;
        return;
    }
    strbuf_append(sb, "\"");
    strbuf_append(sb, escaped);
    strbuf_append(sb, "\"");
    free(escaped);
}

static void strbuf_appendf(strbuf *sb, const char *format, ...) {
    char scratch[128];
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(scratch, sizeof(scratch), format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= sizeof(scratch)) {
        sb->failed = 1;
        return;
    }
    strbuf_append(sb, scratch);
}

/* ------------------------------------------------------------------- server */

struct qwen_server {
    qwen_engine *engine;
    h3_tokenizer *tokenizer;
    char *model_id;
    h3_http_server *http;
    pthread_mutex_t lock;
    unsigned long completion_counter;
};

static void send_json_error(h3_http_responder *responder, int status,
                            const char *message) {
    strbuf body = {0};
    strbuf_append(&body, "{\"error\":{\"message\":");
    strbuf_append_json_string(&body, message);
    strbuf_append(&body, ",\"type\":\"invalid_request_error\"}}");
    if (body.failed || !body.data) {
        h3_http_send(responder, status, "application/json",
                     "{\"error\":{\"message\":\"error\"}}", 29);
    } else {
        h3_http_send(responder, status, "application/json", body.data,
                     body.length);
    }
    strbuf_free(&body);
}

static void handle_models(qwen_server *server, h3_http_responder *responder) {
    strbuf body = {0};
    strbuf_append(&body, "{\"object\":\"list\",\"data\":[{\"id\":");
    strbuf_append_json_string(&body, server->model_id);
    strbuf_append(&body,
                  ",\"object\":\"model\",\"created\":0,\"owned_by\":"
                  "\"h3-runtime\"}]}");
    if (body.failed || !body.data)
        send_json_error(responder, 500, "out of memory");
    else
        h3_http_send(responder, 200, "application/json", body.data,
                     body.length);
    strbuf_free(&body);
}

/* Collect the text of one OpenAI message `content` (string, or an array of
 * {type:"text",text:...} parts). Returns a malloc'd string or NULL. */
static char *message_text(const h3_json *content) {
    const char *direct = h3_json_string_value(content);
    if (direct) {
        char *copy = malloc(strlen(direct) + 1);
        if (copy) strcpy(copy, direct);
        return copy;
    }
    if (!h3_json_is(content, H3_JSON_ARRAY)) return NULL;
    strbuf joined = {0};
    for (size_t index = 0; index < h3_json_array_size(content); index++) {
        const h3_json *part = h3_json_array_at(content, index);
        const char *text = h3_json_string_value(h3_json_object_get(part,
                                                                  "text"));
        if (text) strbuf_append(&joined, text);
    }
    if (joined.failed) {
        strbuf_free(&joined);
        return NULL;
    }
    if (!joined.data) {
        joined.data = calloc(1, 1);
    }
    return joined.data;
}

static int role_from_string(const char *name, qwen_role *out) {
    if (!name) return 0;
    if (!strcmp(name, "system")) *out = QWEN_ROLE_SYSTEM;
    else if (!strcmp(name, "user")) *out = QWEN_ROLE_USER;
    else if (!strcmp(name, "assistant")) *out = QWEN_ROLE_ASSISTANT;
    else if (!strcmp(name, "tool")) *out = QWEN_ROLE_TOOL;
    else return 0;
    return 1;
}

typedef struct {
    char *id;
    char *model;
    long created;
} completion_meta;

static void completion_meta_free(completion_meta *meta) {
    free(meta->id);
    free(meta->model);
}

static void emit_chunk(strbuf *sb, const completion_meta *meta,
                       const char *role_or_null, const char *content_or_null,
                       const char *finish_or_null) {
    strbuf_append(sb, "{\"id\":");
    strbuf_append_json_string(sb, meta->id);
    strbuf_append(sb, ",\"object\":\"chat.completion.chunk\",\"created\":");
    strbuf_appendf(sb, "%ld", meta->created);
    strbuf_append(sb, ",\"model\":");
    strbuf_append_json_string(sb, meta->model);
    strbuf_append(sb, ",\"choices\":[{\"index\":0,\"delta\":{");
    int wrote = 0;
    if (role_or_null) {
        strbuf_append(sb, "\"role\":");
        strbuf_append_json_string(sb, role_or_null);
        wrote = 1;
    }
    if (content_or_null) {
        if (wrote) strbuf_append(sb, ",");
        strbuf_append(sb, "\"content\":");
        strbuf_append_json_string(sb, content_or_null);
    }
    strbuf_append(sb, "},\"finish_reason\":");
    if (finish_or_null) strbuf_append_json_string(sb, finish_or_null);
    else strbuf_append(sb, "null");
    strbuf_append(sb, "}]}");
}

static int stream_line(h3_http_responder *responder, const char *payload) {
    if (!h3_http_write(responder, "data: ", 6)) return 0;
    if (!h3_http_write(responder, payload, strlen(payload))) return 0;
    return h3_http_write(responder, "\n\n", 2);
}

static void handle_chat_completion(qwen_server *server,
                                   const h3_http_request *request,
                                   h3_http_responder *responder) {
    char error[512];
    h3_json *root = h3_json_parse(request->body, request->body_length, error,
                                  sizeof(error));
    if (!root || !h3_json_is(root, H3_JSON_OBJECT)) {
        send_json_error(responder, 400,
                        root ? "request body must be a JSON object" : error);
        h3_json_free(root);
        return;
    }

    const h3_json *messages = h3_json_object_get(root, "messages");
    size_t message_count = h3_json_array_size(messages);
    if (!message_count) {
        send_json_error(responder, 400, "\"messages\" must be a non-empty "
                        "array");
        h3_json_free(root);
        return;
    }

    int stream = h3_json_bool_or(h3_json_object_get(root, "stream"), 0);
    double max_tokens_raw =
        h3_json_number_or(h3_json_object_get(root, "max_tokens"), 256.0);
    int max_tokens = (int)max_tokens_raw;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 4096) max_tokens = 4096;
    const char *requested_model =
        h3_json_string_value(h3_json_object_get(root, "model"));

    qwen_chat_message *chat = calloc(message_count, sizeof(*chat));
    char **owned = calloc(message_count, sizeof(*owned));
    if (!chat || !owned) {
        free(chat);
        free(owned);
        h3_json_free(root);
        send_json_error(responder, 500, "out of memory");
        return;
    }
    int bad = 0;
    for (size_t index = 0; index < message_count && !bad; index++) {
        const h3_json *message = h3_json_array_at(messages, index);
        qwen_role role;
        if (!role_from_string(
                h3_json_string_value(h3_json_object_get(message, "role")),
                &role)) {
            bad = 1;
            break;
        }
        char *text = message_text(h3_json_object_get(message, "content"));
        if (!text) {
            bad = 1;
            break;
        }
        owned[index] = text;
        chat[index].role = role;
        chat[index].content = text;
    }

    uint32_t *ids = NULL;
    size_t prompt_len = 0;
    if (!bad && !qwen_chat_tokenize(server->tokenizer, chat, message_count, 1,
                                    &ids, &prompt_len, error, sizeof(error)))
        bad = 1;

    for (size_t index = 0; index < message_count; index++) free(owned[index]);
    free(owned);
    free(chat);
    h3_json_free(root);

    if (bad) {
        send_json_error(responder, 400,
                        "invalid messages (role/content/tokenization)");
        free(ids);
        return;
    }

    pthread_mutex_lock(&server->lock);

    completion_meta meta = {0};
    meta.created = (long)time(NULL);
    meta.model = strdup(requested_model ? requested_model : server->model_id);
    char id_buffer[48];
    snprintf(id_buffer, sizeof(id_buffer), "chatcmpl-%08lx",
             ++server->completion_counter);
    meta.id = strdup(id_buffer);

    qwen_session *session = NULL;
    int ok = meta.model && meta.id &&
             qwen_session_create(&session, server->engine, error,
                                 sizeof(error)) &&
             qwen_session_eval(session, ids, prompt_len, error, sizeof(error));

    uint32_t *generated = NULL;
    size_t generated_count = 0;
    size_t sent_prefix = 0;   /* bytes of decoded text already streamed */
    const char *finish_reason = "length";
    int streaming_started = 0;

    if (ok) generated = malloc((size_t)max_tokens * sizeof(*generated));
    if (ok && !generated) {
        ok = 0;
        snprintf(error, sizeof(error), "out of memory");
    }

    if (ok && stream) {
        h3_http_begin_stream(responder, 200, "text/event-stream");
        streaming_started = 1;
        strbuf chunk = {0};
        emit_chunk(&chunk, &meta, "assistant", NULL, NULL);
        if (!chunk.failed) stream_line(responder, chunk.data);
        strbuf_free(&chunk);
    }

    for (int step = 0; ok && step < max_tokens; step++) {
        uint32_t next = 0;
        if (!qwen_session_sample(session, &next, error, sizeof(error))) {
            ok = 0;
            break;
        }
        if (next == QWEN_TOKEN_IM_END || next == QWEN_TOKEN_ENDOFTEXT) {
            finish_reason = "stop";
            break;
        }
        generated[generated_count++] = next;

        char *decoded = h3_tokenizer_decode(server->tokenizer, generated,
                                            generated_count, error,
                                            sizeof(error));
        if (!decoded) {
            ok = 0;
            break;
        }
        size_t decoded_length = strlen(decoded);
        if (stream && decoded_length > sent_prefix) {
            strbuf chunk = {0};
            emit_chunk(&chunk, &meta, NULL, decoded + sent_prefix, NULL);
            if (!chunk.failed) stream_line(responder, chunk.data);
            strbuf_free(&chunk);
        }
        if (decoded_length >= sent_prefix) sent_prefix = decoded_length;
        free(decoded);

        if (step + 1 < max_tokens &&
            !qwen_session_eval(session, &next, 1, error, sizeof(error))) {
            ok = 0;
            break;
        }
    }

    if (ok && streaming_started) {
        strbuf chunk = {0};
        emit_chunk(&chunk, &meta, NULL, NULL, finish_reason);
        if (!chunk.failed) stream_line(responder, chunk.data);
        strbuf_free(&chunk);
        stream_line(responder, "[DONE]");
        h3_http_finish(responder);
    } else if (ok) {
        char *final_text = h3_tokenizer_decode(server->tokenizer, generated,
                                               generated_count, error,
                                               sizeof(error));
        strbuf body = {0};
        strbuf_append(&body, "{\"id\":");
        strbuf_append_json_string(&body, meta.id);
        strbuf_append(&body, ",\"object\":\"chat.completion\",\"created\":");
        strbuf_appendf(&body, "%ld", meta.created);
        strbuf_append(&body, ",\"model\":");
        strbuf_append_json_string(&body, meta.model);
        strbuf_append(&body,
                      ",\"choices\":[{\"index\":0,\"message\":{\"role\":"
                      "\"assistant\",\"content\":");
        strbuf_append_json_string(&body, final_text ? final_text : "");
        strbuf_append(&body, "},\"finish_reason\":");
        strbuf_append_json_string(&body, finish_reason);
        strbuf_append(&body, "}],\"usage\":{\"prompt_tokens\":");
        strbuf_appendf(&body, "%zu", prompt_len);
        strbuf_append(&body, ",\"completion_tokens\":");
        strbuf_appendf(&body, "%zu", generated_count);
        strbuf_append(&body, ",\"total_tokens\":");
        strbuf_appendf(&body, "%zu", prompt_len + generated_count);
        strbuf_append(&body, "}}");
        free(final_text);
        if (body.failed || !body.data)
            send_json_error(responder, 500, "out of memory");
        else
            h3_http_send(responder, 200, "application/json", body.data,
                         body.length);
        strbuf_free(&body);
    } else if (streaming_started) {
        strbuf chunk = {0};
        emit_chunk(&chunk, &meta, NULL, NULL, "error");
        if (!chunk.failed) stream_line(responder, chunk.data);
        strbuf_free(&chunk);
        stream_line(responder, "[DONE]");
    } else {
        send_json_error(responder, 500, error);
    }

    free(generated);
    qwen_session_free(session);
    free(ids);
    completion_meta_free(&meta);
    pthread_mutex_unlock(&server->lock);
}

static void dispatch(const h3_http_request *request,
                     h3_http_responder *responder, void *user) {
    qwen_server *server = user;
    if (!strcmp(request->method, "OPTIONS")) {
        h3_http_send(responder, 200, "text/plain", "", 0);
        return;
    }
    if (!strcmp(request->method, "GET") &&
        !strcmp(request->path, "/v1/models")) {
        handle_models(server, responder);
        return;
    }
    if (!strcmp(request->method, "POST") &&
        !strcmp(request->path, "/v1/chat/completions")) {
        handle_chat_completion(server, request, responder);
        return;
    }
    if (!strcmp(request->method, "GET") && !strcmp(request->path, "/")) {
        h3_http_send(responder, 200, "text/plain",
                     "h3-runtime OpenAI-compatible server\n", 35);
        return;
    }
    send_json_error(responder, 404, "unknown route");
}

/* -------------------------------------------------------------- lifecycle */

int qwen_server_create(qwen_server **out, const char *weight_directory,
                       const char *tokenizer_path,
                       const char *shader_source_path, const char *model_id,
                       char *error, size_t error_size) {
    if (out) *out = NULL;
    if (!out || !weight_directory || !tokenizer_path || !shader_source_path) {
        if (error && error_size)
            snprintf(error, error_size, "qwen_server_create missing arguments");
        return 0;
    }
    qwen_server *server = calloc(1, sizeof(*server));
    if (!server) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    server->model_id = strdup(model_id && *model_id ? model_id : "minimax-h3");
    if (!server->model_id) {
        free(server);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    pthread_mutex_init(&server->lock, NULL);
    server->tokenizer = h3_tokenizer_load(tokenizer_path, error, error_size);
    if (!server->tokenizer) {
        qwen_server_free(server);
        return 0;
    }
    if (!qwen_engine_open(&server->engine, weight_directory, shader_source_path,
                          error, error_size)) {
        qwen_server_free(server);
        return 0;
    }
    *out = server;
    return 1;
}

void qwen_server_free(qwen_server *server) {
    if (!server) return;
    if (server->http) h3_http_close(server->http);
    if (server->engine) qwen_engine_close(server->engine);
    if (server->tokenizer) h3_tokenizer_free(server->tokenizer);
    pthread_mutex_destroy(&server->lock);
    free(server->model_id);
    free(server);
}

int qwen_server_run(qwen_server *server, const char *host, uint16_t port,
                    uint16_t *bound_port, char *error, size_t error_size) {
    if (!server) return 0;
    server->http = h3_http_listen(host, port, error, error_size);
    if (!server->http) return 0;
    if (bound_port) *bound_port = h3_http_server_port(server->http);
    int result = h3_http_run(server->http, dispatch, server);
    h3_http_close(server->http);
    server->http = NULL;
    return result == 0;
}

void qwen_server_stop(qwen_server *server) {
    if (server && server->http) h3_http_stop(server->http);
}
