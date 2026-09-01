#include "qwen_tools.h"

#include "h3_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOOL_OPEN "<tool_call>"
#define TOOL_CLOSE "</tool_call>"

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static char *dup_range(const char *begin, const char *end) {
    size_t length = (size_t)(end - begin);
    char *copy = malloc(length + 1);
    if (!copy) return NULL;
    memcpy(copy, begin, length);
    copy[length] = '\0';
    return copy;
}

static char *dup_string(const char *text) {
    return dup_range(text, text + strlen(text));
}

void h3_tool_calls_free(h3_tool_call *calls, size_t count) {
    if (!calls) return;
    for (size_t index = 0; index < count; index++) {
        free(calls[index].id);
        free(calls[index].name);
        free(calls[index].arguments);
    }
    free(calls);
}

/* Pull "name" and "arguments" out of one <tool_call> body. */
static int parse_one(const char *body_begin, const char *body_end,
                     h3_tool_call *call, size_t ordinal, char *error,
                     size_t error_size) {
    while (body_begin < body_end &&
           (*body_begin == '\n' || *body_begin == '\r' || *body_begin == ' ' ||
            *body_begin == '\t'))
        body_begin++;
    while (body_end > body_begin &&
           (body_end[-1] == '\n' || body_end[-1] == '\r' ||
            body_end[-1] == ' ' || body_end[-1] == '\t'))
        body_end--;

    h3_json *object = h3_json_parse(body_begin, (size_t)(body_end - body_begin),
                                    error, error_size);
    if (!object || !h3_json_is(object, H3_JSON_OBJECT)) {
        h3_json_free(object);
        set_error(error, error_size, "tool_call body is not a JSON object");
        return 0;
    }
    const char *name = h3_json_string_value(h3_json_object_get(object, "name"));
    if (!name) {
        h3_json_free(object);
        set_error(error, error_size, "tool_call has no string \"name\"");
        return 0;
    }
    const h3_json *arguments = h3_json_object_get(object, "arguments");
    char *arguments_text;
    const char *as_string = h3_json_string_value(arguments);
    if (as_string) {
        arguments_text = dup_string(as_string);
    } else if (arguments) {
        arguments_text = h3_json_stringify(arguments);
    } else {
        arguments_text = dup_string("{}");
    }

    char id_buffer[24];
    snprintf(id_buffer, sizeof(id_buffer), "call_%04zu", ordinal + 1);
    call->id = dup_string(id_buffer);
    call->name = dup_string(name);
    call->arguments = arguments_text;
    h3_json_free(object);
    if (!call->id || !call->name || !call->arguments) {
        free(call->id);
        free(call->name);
        free(call->arguments);
        memset(call, 0, sizeof(*call));
        set_error(error, error_size, "out of memory parsing tool_call");
        return 0;
    }
    return 1;
}

int qwen_tool_calls_parse(const char *assistant_text, h3_tool_call **out_calls,
                          size_t *out_count, char **out_content, char *error,
                          size_t error_size) {
    if (out_calls) *out_calls = NULL;
    if (out_count) *out_count = 0;
    if (out_content) *out_content = NULL;
    if (!assistant_text || !out_calls || !out_count || !out_content) {
        set_error(error, error_size, "qwen_tool_calls_parse missing arguments");
        return 0;
    }

    const char *first = strstr(assistant_text, TOOL_OPEN);
    const char *content_end = first ? first : assistant_text + strlen(assistant_text);
    while (content_end > assistant_text &&
           (content_end[-1] == '\n' || content_end[-1] == '\r' ||
            content_end[-1] == ' ' || content_end[-1] == '\t'))
        content_end--;
    *out_content = dup_range(assistant_text, content_end);
    if (!*out_content) {
        set_error(error, error_size, "out of memory");
        return 0;
    }
    if (!first) return 1;

    size_t capacity = 4;
    h3_tool_call *calls = calloc(capacity, sizeof(*calls));
    if (!calls) {
        free(*out_content);
        *out_content = NULL;
        set_error(error, error_size, "out of memory");
        return 0;
    }
    size_t count = 0;
    const char *cursor = first;
    while ((cursor = strstr(cursor, TOOL_OPEN)) != NULL) {
        const char *body = cursor + strlen(TOOL_OPEN);
        const char *close = strstr(body, TOOL_CLOSE);
        if (!close) {
            set_error(error, error_size, "unterminated <tool_call>");
            h3_tool_calls_free(calls, count);
            free(*out_content);
            *out_content = NULL;
            return 0;
        }
        if (count == capacity) {
            capacity *= 2;
            h3_tool_call *grown = realloc(calls, capacity * sizeof(*calls));
            if (!grown) {
                set_error(error, error_size, "out of memory");
                h3_tool_calls_free(calls, count);
                free(*out_content);
                *out_content = NULL;
                return 0;
            }
            calls = grown;
        }
        if (!parse_one(body, close, &calls[count], count, error, error_size)) {
            h3_tool_calls_free(calls, count);
            free(*out_content);
            *out_content = NULL;
            return 0;
        }
        count++;
        cursor = close + strlen(TOOL_CLOSE);
    }

    *out_calls = calls;
    *out_count = count;
    return 1;
}
