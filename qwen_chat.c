#include "qwen_engine.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Phase 3 -- MiniMax-H3 / Qwen3-VL ChatML rendering.
 *
 * Mirrors the non-tools path of the released chat_template.json:
 *
 *   [system]     <|im_start|>system\n{content}<|im_end|>\n   (only messages[0])
 *   user         <|im_start|>user\n{content}<|im_end|>\n
 *   assistant    <|im_start|>assistant\n{content}<|im_end|>\n
 *   tool (run)   <|im_start|>user\n<tool_response>\n{c}\n</tool_response>
 *                ... more tool turns ...
 *                <|im_end|>\n
 *   [gen prompt] <|im_start|>assistant\n
 *
 * Assistant tool_calls and the `tools` system block are Phase 5. */

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} string_builder;

static void sb_free(string_builder *sb) {
    free(sb->data);
    sb->data = NULL;
    sb->length = sb->capacity = 0;
}

static void sb_append(string_builder *sb, const char *text) {
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

static void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const char *role_name(qwen_role role) {
    switch (role) {
        case QWEN_ROLE_SYSTEM: return "system";
        case QWEN_ROLE_USER: return "user";
        case QWEN_ROLE_ASSISTANT: return "assistant";
        case QWEN_ROLE_TOOL: return "tool";
    }
    return NULL;
}

int qwen_chat_render(const qwen_chat_message *messages, size_t count,
                     int add_generation_prompt, char **text_out,
                     char *error, size_t error_size) {
    if (text_out) *text_out = NULL;
    if (!text_out || (count && !messages)) {
        set_error(error, error_size, "qwen_chat_render requires messages and "
                  "text_out");
        return 0;
    }
    for (size_t index = 0; index < count; index++) {
        if (!role_name(messages[index].role)) {
            set_error(error, error_size, "message %zu has an unknown role",
                      index);
            return 0;
        }
        if (!messages[index].content) {
            set_error(error, error_size, "message %zu has NULL content", index);
            return 0;
        }
        if (messages[index].role == QWEN_ROLE_SYSTEM && index != 0) {
            set_error(error, error_size,
                      "system message must be first (message %zu)", index);
            return 0;
        }
    }

    string_builder sb = {0};
    size_t start = 0;
    if (count && messages[0].role == QWEN_ROLE_SYSTEM) {
        sb_append(&sb, "<|im_start|>system\n");
        sb_append(&sb, messages[0].content);
        sb_append(&sb, "<|im_end|>\n");
        start = 1;
    }

    for (size_t index = start; index < count; index++) {
        const qwen_chat_message *message = &messages[index];
        if (message->role == QWEN_ROLE_USER) {
            sb_append(&sb, "<|im_start|>user\n");
            sb_append(&sb, message->content);
            sb_append(&sb, "<|im_end|>\n");
        } else if (message->role == QWEN_ROLE_ASSISTANT) {
            sb_append(&sb, "<|im_start|>assistant\n");
            sb_append(&sb, message->content);
            sb_append(&sb, "<|im_end|>\n");
        } else if (message->role == QWEN_ROLE_TOOL) {
            int first_in_run =
                index == start || messages[index - 1].role != QWEN_ROLE_TOOL;
            int last_in_run =
                index + 1 == count || messages[index + 1].role != QWEN_ROLE_TOOL;
            if (first_in_run) sb_append(&sb, "<|im_start|>user");
            sb_append(&sb, "\n<tool_response>\n");
            sb_append(&sb, message->content);
            sb_append(&sb, "\n</tool_response>");
            if (last_in_run) sb_append(&sb, "<|im_end|>\n");
        }
    }

    if (add_generation_prompt) sb_append(&sb, "<|im_start|>assistant\n");

    if (sb.failed) {
        sb_free(&sb);
        set_error(error, error_size, "out of memory rendering chat template");
        return 0;
    }
    if (!sb.data) {
        sb.data = calloc(1, 1);
        if (!sb.data) {
            set_error(error, error_size, "out of memory rendering chat "
                      "template");
            return 0;
        }
    }
    *text_out = sb.data;
    return 1;
}

int qwen_chat_tokenize(const h3_tokenizer *tokenizer,
                       const qwen_chat_message *messages, size_t count,
                       int add_generation_prompt, uint32_t **ids,
                       size_t *id_count, char *error, size_t error_size) {
    if (ids) *ids = NULL;
    if (id_count) *id_count = 0;
    if (!tokenizer || !ids || !id_count) {
        set_error(error, error_size, "qwen_chat_tokenize requires tokenizer, "
                  "ids and id_count");
        return 0;
    }
    char *text = NULL;
    if (!qwen_chat_render(messages, count, add_generation_prompt, &text, error,
                          error_size))
        return 0;
    int ok = h3_tokenizer_encode(tokenizer, text, 0, ids, id_count, error,
                                 error_size);
    free(text);
    return ok;
}
