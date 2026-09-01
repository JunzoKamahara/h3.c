#include "qwen_server.h"

#include "h3_http.h"
#include "h3_json.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"
#include "qwen_tools.h"

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
    qwen_session *session; /* persistent: rewound to empty per request */
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

/* [{"id":..,"type":"function","function":{"name":..,"arguments":".."}}, ...]
 * `arguments` is emitted as a JSON string per the OpenAI schema. With
 * `with_index` each entry also carries a streaming "index". */
static void append_tool_calls_array(strbuf *sb, const h3_tool_call *calls,
                                    size_t count, int with_index) {
    strbuf_append(sb, "[");
    for (size_t index = 0; index < count; index++) {
        if (index) strbuf_append(sb, ",");
        strbuf_append(sb, "{");
        if (with_index) strbuf_appendf(sb, "\"index\":%zu,", index);
        strbuf_append(sb, "\"id\":");
        strbuf_append_json_string(sb, calls[index].id);
        strbuf_append(sb, ",\"type\":\"function\",\"function\":{\"name\":");
        strbuf_append_json_string(sb, calls[index].name);
        strbuf_append(sb, ",\"arguments\":");
        strbuf_append_json_string(sb, calls[index].arguments);
        strbuf_append(sb, "}}");
    }
    strbuf_append(sb, "]");
}

static void emit_tool_calls_chunk(strbuf *sb, const completion_meta *meta,
                                  const h3_tool_call *calls, size_t count) {
    strbuf_append(sb, "{\"id\":");
    strbuf_append_json_string(sb, meta->id);
    strbuf_append(sb, ",\"object\":\"chat.completion.chunk\",\"created\":");
    strbuf_appendf(sb, "%ld", meta->created);
    strbuf_append(sb, ",\"model\":");
    strbuf_append_json_string(sb, meta->model);
    strbuf_append(sb, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
    append_tool_calls_array(sb, calls, count, 1);
    strbuf_append(sb, "},\"finish_reason\":null}]}");
}

/* ------------------------------------------------------- generation core */

typedef struct {
    int ok;
    char *text;             /* full assistant text (NULL on failure) */
    h3_tool_call *calls;
    size_t call_count;
    char *content;          /* text before the first <tool_call>, or NULL */
    size_t prompt_tokens;
    size_t completion_tokens;
    const char *finish;     /* "stop" | "length" | "tool_calls" */
} gen_result;

static void gen_result_free(gen_result *result) {
    free(result->text);
    free(result->content);
    h3_tool_calls_free(result->calls, result->call_count);
    memset(result, 0, sizeof(*result));
}

/* Tokenize `chat`, prefill the (persistent) session, greedily decode up to
 * `max_tokens`, and lift any <tool_call> markup. `on_text_delta`, when
 * non-NULL, receives each fresh run of assistant text -- never tool markup.
 * The caller holds server->lock. */
static void run_chat(qwen_server *server, const qwen_chat_message *chat,
                     size_t msg_count, const char *const *tool_jsons,
                     size_t tool_count, int max_tokens,
                     void (*on_text_delta)(void *, const char *), void *ctx,
                     gen_result *out, char *error, size_t error_size) {
    memset(out, 0, sizeof(*out));
    out->finish = "length";
    int has_tools = tool_count > 0;

    uint32_t *ids = NULL;
    size_t prompt_len = 0;
    if (!qwen_chat_tokenize_tools(server->tokenizer, chat, msg_count,
                                  tool_jsons, tool_count, 1, &ids, &prompt_len,
                                  error, error_size))
        return;
    out->prompt_tokens = prompt_len;

    int ok = qwen_session_rewind(server->session, 0, error, error_size) &&
             qwen_session_eval(server->session, ids, prompt_len, error,
                               error_size);
    uint32_t *generated =
        ok ? malloc((size_t)max_tokens * sizeof(*generated)) : NULL;
    if (ok && !generated) {
        ok = 0;
        snprintf(error, error_size, "out of memory");
    }
    size_t generated_count = 0;
    size_t sent_prefix = 0;
    int saw_tool_markup = 0;

    for (int step = 0; ok && step < max_tokens; step++) {
        uint32_t next = 0;
        if (!qwen_session_sample(server->session, &next, error, error_size)) {
            ok = 0;
            break;
        }
        if (next == QWEN_TOKEN_IM_END || next == QWEN_TOKEN_ENDOFTEXT) {
            out->finish = "stop";
            break;
        }
        generated[generated_count++] = next;
        char *decoded = h3_tokenizer_decode(server->tokenizer, generated,
                                            generated_count, error,
                                            error_size);
        if (!decoded) {
            ok = 0;
            break;
        }
        size_t decoded_length = strlen(decoded);
        if (has_tools && strstr(decoded, "<tool_call>")) saw_tool_markup = 1;
        if (on_text_delta && !saw_tool_markup && decoded_length > sent_prefix)
            on_text_delta(ctx, decoded + sent_prefix);
        if (decoded_length >= sent_prefix) sent_prefix = decoded_length;
        free(decoded);
        if (step + 1 < max_tokens &&
            !qwen_session_eval(server->session, &next, 1, error, error_size)) {
            ok = 0;
            break;
        }
    }

    out->completion_tokens = generated_count;
    if (ok) {
        out->text = h3_tokenizer_decode(server->tokenizer, generated,
                                        generated_count, error, error_size);
        if (!out->text) ok = 0;
    }
    if (ok && has_tools &&
        qwen_tool_calls_parse(out->text, &out->calls, &out->call_count,
                              &out->content, error, error_size) &&
        out->call_count > 0)
        out->finish = "tool_calls";
    out->ok = ok;
    free(generated);
    free(ids);
}

/* --------------------------------------------------- /v1/chat/completions */

typedef struct {
    h3_http_responder *responder;
    const completion_meta *meta;
} chat_delta_ctx;

static void chat_text_delta(void *opaque, const char *delta) {
    chat_delta_ctx *context = opaque;
    strbuf chunk = {0};
    emit_chunk(&chunk, context->meta, NULL, delta, NULL);
    if (!chunk.failed) stream_line(context->responder, chunk.data);
    strbuf_free(&chunk);
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

    /* Optional tool definitions -> one compact JSON string per tool. */
    const h3_json *tools = h3_json_object_get(root, "tools");
    size_t tool_count = h3_json_array_size(tools);
    char **tool_jsons = tool_count ? calloc(tool_count, sizeof(*tool_jsons))
                                   : NULL;
    int has_tools = 0;
    if (tool_count) {
        has_tools = tool_jsons != NULL;
        for (size_t index = 0; index < tool_count && has_tools; index++) {
            tool_jsons[index] =
                h3_json_stringify(h3_json_array_at(tools, index));
            if (!tool_jsons[index]) has_tools = 0;
        }
    }

    qwen_chat_message *chat = calloc(message_count, sizeof(*chat));
    char **owned = calloc(message_count, sizeof(*owned));
    char **owned_calls = calloc(message_count, sizeof(*owned_calls));
    int bad = !chat || !owned || !owned_calls ||
              (tool_count && (!tool_jsons || !has_tools));
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
        const h3_json *prior = h3_json_object_get(message, "tool_calls");
        if (role == QWEN_ROLE_ASSISTANT && h3_json_is(prior, H3_JSON_ARRAY) &&
            h3_json_array_size(prior)) {
            owned_calls[index] = h3_json_stringify(prior);
            chat[index].tool_calls_json = owned_calls[index];
        }
    }

    h3_json_free(root);
    if (bad) {
        for (size_t index = 0; index < message_count; index++) {
            if (owned) free(owned[index]);
            if (owned_calls) free(owned_calls[index]);
        }
        for (size_t index = 0; index < tool_count; index++)
            if (tool_jsons) free(tool_jsons[index]);
        free(tool_jsons);
        free(owned);
        free(owned_calls);
        free(chat);
        send_json_error(responder, 400,
                        "invalid request (messages / tools / tokenization)");
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

    int streaming_started = 0;
    if (meta.model && meta.id && stream) {
        h3_http_begin_stream(responder, 200, "text/event-stream");
        streaming_started = 1;
        strbuf chunk = {0};
        emit_chunk(&chunk, &meta, "assistant", NULL, NULL);
        if (!chunk.failed) stream_line(responder, chunk.data);
        strbuf_free(&chunk);
    }

    chat_delta_ctx delta_ctx = {responder, &meta};
    gen_result result;
    memset(&result, 0, sizeof(result));
    if (meta.model && meta.id)
        run_chat(server, chat, message_count,
                 (const char *const *)tool_jsons, tool_count, max_tokens,
                 streaming_started ? chat_text_delta : NULL, &delta_ctx,
                 &result, error, sizeof(error));

    if (result.ok && streaming_started) {
        if (result.call_count > 0) {
            strbuf chunk = {0};
            emit_tool_calls_chunk(&chunk, &meta, result.calls,
                                  result.call_count);
            if (!chunk.failed) stream_line(responder, chunk.data);
            strbuf_free(&chunk);
        }
        strbuf chunk = {0};
        emit_chunk(&chunk, &meta, NULL, NULL, result.finish);
        if (!chunk.failed) stream_line(responder, chunk.data);
        strbuf_free(&chunk);
        stream_line(responder, "[DONE]");
        h3_http_finish(responder);
    } else if (result.ok) {
        const char *content =
            result.call_count > 0 ? result.content : result.text;
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
        if (result.call_count > 0 && (!content || !content[0]))
            strbuf_append(&body, "null");
        else
            strbuf_append_json_string(&body, content ? content : "");
        if (result.call_count > 0) {
            strbuf_append(&body, ",\"tool_calls\":");
            append_tool_calls_array(&body, result.calls, result.call_count, 0);
        }
        strbuf_append(&body, "},\"finish_reason\":");
        strbuf_append_json_string(&body, result.finish);
        strbuf_append(&body, "}],\"usage\":{\"prompt_tokens\":");
        strbuf_appendf(&body, "%zu", result.prompt_tokens);
        strbuf_append(&body, ",\"completion_tokens\":");
        strbuf_appendf(&body, "%zu", result.completion_tokens);
        strbuf_append(&body, ",\"total_tokens\":");
        strbuf_appendf(&body, "%zu",
                       result.prompt_tokens + result.completion_tokens);
        strbuf_append(&body, "}}");
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

    gen_result_free(&result);
    for (size_t index = 0; index < message_count; index++) {
        free(owned[index]);
        free(owned_calls[index]);
    }
    for (size_t index = 0; index < tool_count; index++) free(tool_jsons[index]);
    free(tool_jsons);
    free(owned);
    free(owned_calls);
    free(chat);
    completion_meta_free(&meta);
    pthread_mutex_unlock(&server->lock);
}

/* ---------------------------------------------------------- /v1/responses */

static int sse_event(h3_http_responder *responder, const char *type,
                     const char *payload) {
    if (!h3_http_write(responder, "event: ", 7)) return 0;
    if (!h3_http_write(responder, type, strlen(type))) return 0;
    if (!h3_http_write(responder, "\ndata: ", 7)) return 0;
    if (!h3_http_write(responder, payload, strlen(payload))) return 0;
    return h3_http_write(responder, "\n\n", 2);
}

/* The `output` array: one assistant message item for the leading text, then a
 * function_call item per tool call. */
static void append_response_output(strbuf *sb, const completion_meta *meta,
                                   const gen_result *result, int completed) {
    const char *item_status = completed ? "completed" : "in_progress";
    const char *content =
        result->call_count > 0 ? result->content : result->text;
    strbuf_append(sb, "[");
    int wrote = 0;
    if (result->call_count == 0 || (content && content[0])) {
        strbuf_append(sb, "{\"id\":\"msg_");
        strbuf_append(sb, meta->id);
        strbuf_append(sb, "\",\"type\":\"message\",\"status\":\"");
        strbuf_append(sb, item_status);
        strbuf_append(sb, "\",\"role\":\"assistant\",\"content\":[{\"type\":"
                          "\"output_text\",\"text\":");
        strbuf_append_json_string(sb, content ? content : "");
        strbuf_append(sb, ",\"annotations\":[]}]}");
        wrote = 1;
    }
    for (size_t index = 0; index < result->call_count; index++) {
        if (wrote) strbuf_append(sb, ",");
        wrote = 1;
        strbuf_append(sb, "{\"id\":\"fc_");
        strbuf_append_json_string(sb, result->calls[index].id);
        strbuf_append(sb, ",\"type\":\"function_call\",\"status\":\"");
        strbuf_append(sb, item_status);
        strbuf_append(sb, "\",\"call_id\":");
        strbuf_append_json_string(sb, result->calls[index].id);
        strbuf_append(sb, ",\"name\":");
        strbuf_append_json_string(sb, result->calls[index].name);
        strbuf_append(sb, ",\"arguments\":");
        strbuf_append_json_string(sb, result->calls[index].arguments);
        strbuf_append(sb, "}");
    }
    strbuf_append(sb, "]");
}

static void append_response_object(strbuf *sb, const completion_meta *meta,
                                   const char *status,
                                   const gen_result *result) {
    strbuf_append(sb, "{\"id\":");
    strbuf_append_json_string(sb, meta->id);
    strbuf_append(sb, ",\"object\":\"response\",\"created_at\":");
    strbuf_appendf(sb, "%ld", meta->created);
    strbuf_append(sb, ",\"model\":");
    strbuf_append_json_string(sb, meta->model);
    strbuf_append(sb, ",\"status\":");
    strbuf_append_json_string(sb, status);
    strbuf_append(sb, ",\"output\":");
    if (result) {
        append_response_output(sb, meta, result, 1);
        const char *text =
            result->call_count > 0 ? result->content : result->text;
        strbuf_append(sb, ",\"output_text\":");
        strbuf_append_json_string(sb, text ? text : "");
        strbuf_append(sb, ",\"usage\":{\"input_tokens\":");
        strbuf_appendf(sb, "%zu", result->prompt_tokens);
        strbuf_append(sb, ",\"output_tokens\":");
        strbuf_appendf(sb, "%zu", result->completion_tokens);
        strbuf_append(sb, ",\"total_tokens\":");
        strbuf_appendf(sb, "%zu",
                       result->prompt_tokens + result->completion_tokens);
        strbuf_append(sb, "}}");
    } else {
        strbuf_append(sb, "[],\"output_text\":\"\",\"usage\":null}");
    }
}

typedef struct {
    h3_http_responder *responder;
    const completion_meta *meta;
} responses_delta_ctx;

static void responses_text_delta(void *opaque, const char *delta) {
    responses_delta_ctx *context = opaque;
    strbuf payload = {0};
    strbuf_append(&payload,
                  "{\"type\":\"response.output_text.delta\",\"item_id\":"
                  "\"msg_");
    strbuf_append(&payload, context->meta->id);
    strbuf_append(&payload,
                  "\",\"output_index\":0,\"content_index\":0,\"delta\":");
    strbuf_append_json_string(&payload, delta);
    strbuf_append(&payload, "}");
    if (!payload.failed)
        sse_event(context->responder, "response.output_text.delta",
                  payload.data);
    strbuf_free(&payload);
}

static void handle_responses(qwen_server *server,
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

    int stream = h3_json_bool_or(h3_json_object_get(root, "stream"), 0);
    double max_raw = h3_json_number_or(
        h3_json_object_get(root, "max_output_tokens"), 256.0);
    int max_tokens = (int)max_raw;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 4096) max_tokens = 4096;
    const char *requested_model =
        h3_json_string_value(h3_json_object_get(root, "model"));
    const char *instructions =
        h3_json_string_value(h3_json_object_get(root, "instructions"));
    const h3_json *input = h3_json_object_get(root, "input");
    const h3_json *tools = h3_json_object_get(root, "tools");
    size_t tool_count = h3_json_array_size(tools);

    /* input: a bare string, or an array of message / function_call_output
     * items. */
    const char *input_string = h3_json_string_value(input);
    size_t input_count = input_string ? 1 : h3_json_array_size(input);
    if (!input_string && !input_count) {
        send_json_error(responder, 400, "\"input\" must be a string or a "
                        "non-empty array");
        h3_json_free(root);
        return;
    }

    size_t max_messages = input_count + (instructions ? 1 : 0);
    qwen_chat_message *chat = calloc(max_messages, sizeof(*chat));
    char **owned = calloc(max_messages, sizeof(*owned));
    char **owned_calls = calloc(max_messages, sizeof(*owned_calls));
    char **tool_jsons = tool_count ? calloc(tool_count, sizeof(*tool_jsons))
                                   : NULL;
    int bad = !chat || !owned || !owned_calls ||
              (tool_count && !tool_jsons);
    for (size_t index = 0; index < tool_count && !bad; index++) {
        tool_jsons[index] = h3_json_stringify(h3_json_array_at(tools, index));
        if (!tool_jsons[index]) bad = 1;
    }

    size_t message_count = 0;
    if (!bad && instructions) {
        owned[message_count] = message_text(
            h3_json_object_get(root, "instructions"));
        if (!owned[message_count]) bad = 1;
        else {
            chat[message_count].role = QWEN_ROLE_SYSTEM;
            chat[message_count].content = owned[message_count];
            message_count++;
        }
    }
    if (!bad && input_string) {
        owned[message_count] = message_text(input);
        if (!owned[message_count]) bad = 1;
        else {
            chat[message_count].role = QWEN_ROLE_USER;
            chat[message_count].content = owned[message_count];
            message_count++;
        }
    }
    for (size_t index = 0; index < input_count && !bad && !input_string;
         index++) {
        const h3_json *item = h3_json_array_at(input, index);
        const char *type =
            h3_json_string_value(h3_json_object_get(item, "type"));
        qwen_role role = QWEN_ROLE_USER;
        char *text = NULL;
        if (type && !strcmp(type, "function_call_output")) {
            role = QWEN_ROLE_TOOL;
            text = message_text(h3_json_object_get(item, "output"));
        } else if (type && !strcmp(type, "function_call")) {
            role = QWEN_ROLE_ASSISTANT;
            text = calloc(1, 1);
            const char *name =
                h3_json_string_value(h3_json_object_get(item, "name"));
            const h3_json *arguments = h3_json_object_get(item, "arguments");
            char *args = arguments ? h3_json_stringify(arguments) : NULL;
            strbuf calls = {0};
            strbuf_append(&calls, "[{\"name\":");
            strbuf_append_json_string(&calls, name ? name : "");
            strbuf_append(&calls, ",\"arguments\":");
            strbuf_append(&calls, args ? args : "{}");
            strbuf_append(&calls, "}]");
            free(args);
            owned_calls[message_count] = calls.data;
            chat[message_count].tool_calls_json = calls.data;
        } else {
            const char *role_name =
                h3_json_string_value(h3_json_object_get(item, "role"));
            if (role_name && !role_from_string(role_name, &role)) {
                bad = 1;
                break;
            }
            text = message_text(h3_json_object_get(item, "content"));
        }
        if (!text) {
            bad = 1;
            break;
        }
        owned[message_count] = text;
        chat[message_count].role = role;
        chat[message_count].content = text;
        message_count++;
    }

    h3_json_free(root);
    if (bad || !message_count) {
        for (size_t index = 0; index < max_messages; index++) {
            if (owned) free(owned[index]);
            if (owned_calls) free(owned_calls[index]);
        }
        for (size_t index = 0; index < tool_count; index++)
            if (tool_jsons) free(tool_jsons[index]);
        free(tool_jsons);
        free(owned);
        free(owned_calls);
        free(chat);
        send_json_error(responder, 400, "invalid \"input\" for /v1/responses");
        return;
    }

    pthread_mutex_lock(&server->lock);

    completion_meta meta = {0};
    meta.created = (long)time(NULL);
    meta.model = strdup(requested_model ? requested_model : server->model_id);
    char id_buffer[48];
    snprintf(id_buffer, sizeof(id_buffer), "resp_%08lx",
             ++server->completion_counter);
    meta.id = strdup(id_buffer);

    int streaming_started = 0;
    if (meta.model && meta.id && stream) {
        h3_http_begin_stream(responder, 200, "text/event-stream");
        streaming_started = 1;
        strbuf payload = {0};
        strbuf_append(&payload, "{\"type\":\"response.created\",\"response\":");
        append_response_object(&payload, &meta, "in_progress", NULL);
        strbuf_append(&payload, "}");
        if (!payload.failed)
            sse_event(responder, "response.created", payload.data);
        strbuf_free(&payload);

        payload = (strbuf){0};
        strbuf_append(&payload,
                      "{\"type\":\"response.output_item.added\",\"output_index"
                      "\":0,\"item\":{\"id\":\"msg_");
        strbuf_append(&payload, meta.id);
        strbuf_append(&payload,
                      "\",\"type\":\"message\",\"status\":\"in_progress\","
                      "\"role\":\"assistant\",\"content\":[]}}");
        if (!payload.failed)
            sse_event(responder, "response.output_item.added", payload.data);
        strbuf_free(&payload);
    }

    responses_delta_ctx delta_ctx = {responder, &meta};
    gen_result result;
    memset(&result, 0, sizeof(result));
    if (meta.model && meta.id)
        run_chat(server, chat, message_count,
                 (const char *const *)tool_jsons, tool_count, max_tokens,
                 streaming_started ? responses_text_delta : NULL, &delta_ctx,
                 &result, error, sizeof(error));

    if (result.ok && streaming_started) {
        strbuf payload = {0};
        strbuf_append(&payload,
                      "{\"type\":\"response.output_text.done\",\"item_id\":"
                      "\"msg_");
        strbuf_append(&payload, meta.id);
        strbuf_append(&payload,
                      "\",\"output_index\":0,\"content_index\":0,\"text\":");
        strbuf_append_json_string(
            &payload, result.call_count > 0
                          ? (result.content ? result.content : "")
                          : (result.text ? result.text : ""));
        strbuf_append(&payload, "}");
        if (!payload.failed)
            sse_event(responder, "response.output_text.done", payload.data);
        strbuf_free(&payload);

        /* response.output_item.done and .completed both carry the finished
         * output array; clients rebuild the response from either. */
        payload = (strbuf){0};
        strbuf_append(&payload,
                      "{\"type\":\"response.completed\",\"response\":");
        append_response_object(&payload, &meta, "completed", &result);
        strbuf_append(&payload, "}");
        if (!payload.failed)
            sse_event(responder, "response.completed", payload.data);
        strbuf_free(&payload);
        h3_http_finish(responder);
    } else if (result.ok) {
        strbuf body = {0};
        append_response_object(&body, &meta, "completed", &result);
        if (body.failed || !body.data)
            send_json_error(responder, 500, "out of memory");
        else
            h3_http_send(responder, 200, "application/json", body.data,
                         body.length);
        strbuf_free(&body);
    } else if (streaming_started) {
        strbuf payload = {0};
        strbuf_append(&payload,
                      "{\"type\":\"response.failed\",\"response\":");
        append_response_object(&payload, &meta, "failed", NULL);
        strbuf_append(&payload, "}");
        if (!payload.failed)
            sse_event(responder, "response.failed", payload.data);
        strbuf_free(&payload);
        h3_http_finish(responder);
    } else {
        send_json_error(responder, 500, error);
    }

    gen_result_free(&result);
    for (size_t index = 0; index < max_messages; index++) {
        free(owned[index]);
        free(owned_calls[index]);
    }
    for (size_t index = 0; index < tool_count; index++) free(tool_jsons[index]);
    free(tool_jsons);
    free(owned);
    free(owned_calls);
    free(chat);
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
    if (!strcmp(request->method, "POST") &&
        !strcmp(request->path, "/v1/responses")) {
        handle_responses(server, request, responder);
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
                       int resident, char *error, size_t error_size) {
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
    if (!qwen_session_create(&server->session, server->engine, error,
                             error_size)) {
        qwen_server_free(server);
        return 0;
    }
    if (resident) {
        /* Force the ~62 GB resident load now with a throwaway warm-up eval so
         * the server is only "ready" once weights are pinned. */
        uint32_t warm = QWEN_TOKEN_ENDOFTEXT;
        if (!qwen_session_set_resident(server->session, 1, error, error_size) ||
            !qwen_session_eval(server->session, &warm, 1, error, error_size) ||
            !qwen_session_rewind(server->session, 0, error, error_size)) {
            qwen_server_free(server);
            return 0;
        }
    }
    *out = server;
    return 1;
}

void qwen_server_free(qwen_server *server) {
    if (!server) return;
    if (server->http) h3_http_close(server->http);
    if (server->session) qwen_session_free(server->session);
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
