#include "qwen_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TC_OPEN "<tool_call>"
#define TC_CLOSE "</tool_call>"
#define TC_OPEN_LEN 11
#define TC_CLOSE_LEN 12
/* longest tail we must withhold from an open arguments value: "}\n</tool_call>" */
#define TC_ARGS_HOLDBACK 14

typedef struct {
    char *id;
    char *name;
    char *arguments;   /* accumulated stable prefix */
    size_t args_emitted;
    int begun;
    int ended;
} call_state;

struct qwen_stream {
    qwen_stream_sink sink;
    char *text;            /* leading text region, NUL-terminated */
    size_t text_len;
    size_t text_emitted;
    int in_markup;
    call_state *state;
    size_t state_count;
    h3_tool_call *calls;   /* finalized */
    size_t calls_count;
    int failed;
};

qwen_stream *qwen_stream_new(const qwen_stream_sink *sink) {
    qwen_stream *stream = calloc(1, sizeof(*stream));
    if (!stream) return NULL;
    if (sink) stream->sink = *sink;
    stream->text = calloc(1, 1);
    if (!stream->text) {
        free(stream);
        return NULL;
    }
    return stream;
}

void qwen_stream_free(qwen_stream *stream) {
    if (!stream) return;
    free(stream->text);
    for (size_t index = 0; index < stream->state_count; index++) {
        free(stream->state[index].id);
        free(stream->state[index].name);
        free(stream->state[index].arguments);
    }
    free(stream->state);
    h3_tool_calls_free(stream->calls, stream->calls_count);
    free(stream);
}

/* Longest suffix of s[0:len] that is a proper prefix of TC_OPEN. */
static size_t open_tag_tail(const char *s, size_t len) {
    size_t most = len < (size_t)(TC_OPEN_LEN - 1) ? len
                                                  : (size_t)(TC_OPEN_LEN - 1);
    for (size_t k = most; k > 0; k--) {
        if (!memcmp(s + len - k, TC_OPEN, k)) return k;
    }
    return 0;
}

static void emit_text(qwen_stream *stream, size_t up_to) {
    if (up_to <= stream->text_emitted || up_to > stream->text_len) return;
    if (stream->sink.on_text) {
        char *piece = malloc(up_to - stream->text_emitted + 1);
        if (!piece) {
            stream->failed = 1;
            return;
        }
        memcpy(piece, stream->text + stream->text_emitted,
               up_to - stream->text_emitted);
        piece[up_to - stream->text_emitted] = '\0';
        stream->sink.on_text(stream->sink.ctx, piece);
        free(piece);
    }
    stream->text_emitted = up_to;
}

/* Find `"key"` then `:` in body; return the offset of the first value byte
 * (skipping whitespace after the colon), or (size_t)-1. */
static size_t value_start(const char *body, size_t body_len, const char *key) {
    size_t key_len = strlen(key);
    for (size_t i = 0; i + key_len + 1 < body_len; i++) {
        if (body[i] != '"' || memcmp(body + i + 1, key, key_len) ||
            body[i + 1 + key_len] != '"')
            continue;
        size_t j = i + 1 + key_len + 1;
        while (j < body_len && (body[j] == ' ' || body[j] == '\t')) j++;
        if (j >= body_len || body[j] != ':') return (size_t)-1;
        j++;
        while (j < body_len && (body[j] == ' ' || body[j] == '\t' ||
                                body[j] == '\n' || body[j] == '\r'))
            j++;
        return j;
    }
    return (size_t)-1;
}

/* Extract a complete "string" starting at body[off] == '"'. Returns 1 with
 * *out / *out_len set when the closing quote is present. */
static int json_string_at(const char *body, size_t body_len, size_t off,
                          const char **out, size_t *out_len) {
    if (off >= body_len || body[off] != '"') return 0;
    for (size_t j = off + 1; j < body_len; j++) {
        if (body[j] == '\\') {
            j++;
            continue;
        }
        if (body[j] == '"') {
            *out = body + off + 1;
            *out_len = j - off - 1;
            return 1;
        }
    }
    return 0;
}

static call_state *ensure_state(qwen_stream *stream, size_t index) {
    if (index < stream->state_count) return &stream->state[index];
    call_state *grown =
        realloc(stream->state, (index + 1) * sizeof(*grown));
    if (!grown) {
        stream->failed = 1;
        return NULL;
    }
    stream->state = grown;
    for (size_t k = stream->state_count; k <= index; k++)
        memset(&stream->state[k], 0, sizeof(stream->state[k]));
    stream->state_count = index + 1;
    return &stream->state[index];
}

static void emit_args(qwen_stream *stream, size_t index, const char *stable,
                      size_t stable_len) {
    call_state *call = &stream->state[index];
    if (stable_len <= call->args_emitted) return;
    size_t delta_len = stable_len - call->args_emitted;
    char *piece = malloc(delta_len + 1);
    char *merged = realloc(call->arguments, stable_len + 1);
    if (!piece || !merged) {
        free(piece);
        if (merged) call->arguments = merged;
        stream->failed = 1;
        return;
    }
    call->arguments = merged;
    memcpy(piece, stable + call->args_emitted, delta_len);
    piece[delta_len] = '\0';
    memcpy(call->arguments + call->args_emitted, stable + call->args_emitted,
           delta_len);
    call->arguments[stable_len] = '\0';
    call->args_emitted = stable_len;
    if (stream->sink.on_call_arguments)
        stream->sink.on_call_arguments(stream->sink.ctx, index, call->id, piece);
    free(piece);
}

static void finalize_call(qwen_stream *stream, size_t index, const char *name,
                          const char *args) {
    h3_tool_call *grown = realloc(
        stream->calls, (stream->calls_count + 1) * sizeof(*grown));
    if (!grown) {
        stream->failed = 1;
        return;
    }
    stream->calls = grown;
    h3_tool_call *slot = &stream->calls[stream->calls_count];
    char id_buffer[24];
    snprintf(id_buffer, sizeof(id_buffer), "call_%04zu", index + 1);
    slot->id = strdup(id_buffer);
    slot->name = strdup(name ? name : "");
    slot->arguments = strdup(args ? args : "{}");
    if (!slot->id || !slot->name || !slot->arguments) {
        free(slot->id);
        free(slot->name);
        free(slot->arguments);
        stream->failed = 1;
        return;
    }
    stream->calls_count++;
}

static void process_call(qwen_stream *stream, size_t index, const char *body,
                         size_t body_len, int closed, int finishing) {
    call_state *call = ensure_state(stream, index);
    if (!call || call->ended) return;

    /* name */
    size_t name_off = value_start(body, body_len, "name");
    const char *name = NULL;
    size_t name_len = 0;
    if (name_off != (size_t)-1)
        json_string_at(body, body_len, name_off, &name, &name_len);
    char name_buffer[128];
    if (name && !call->begun) {
        size_t copy = name_len < sizeof(name_buffer) - 1 ? name_len
                                                         : sizeof(name_buffer) - 1;
        memcpy(name_buffer, name, copy);
        name_buffer[copy] = '\0';
        char id_buffer[24];
        snprintf(id_buffer, sizeof(id_buffer), "call_%04zu", index + 1);
        call->id = strdup(id_buffer);
        call->name = strdup(name_buffer);
        if (!call->id || !call->name) {
            stream->failed = 1;
            return;
        }
        call->begun = 1;
        if (stream->sink.on_call_begin)
            stream->sink.on_call_begin(stream->sink.ctx, index, call->id,
                                       call->name);
    }
    if (!call->begun) return; /* wait for the name */

    /* arguments */
    size_t args_off = value_start(body, body_len, "arguments");
    if (args_off == (size_t)-1) {
        if (closed || finishing) {
            emit_args(stream, index, "", 0);
            call->ended = 1;
            finalize_call(stream, index, call->name,
                          call->arguments ? call->arguments : "{}");
            if (stream->sink.on_call_end)
                stream->sink.on_call_end(stream->sink.ctx, index,
                                         stream->state[index].id);
        }
        return;
    }
    const char *args = body + args_off;
    size_t region_len = body_len - args_off;

    if (closed || finishing) {
        size_t end = region_len;
        while (end > 0 && (args[end - 1] == ' ' || args[end - 1] == '\t' ||
                           args[end - 1] == '\n' || args[end - 1] == '\r'))
            end--;
        if (end > 0 && args[end - 1] == '}') end--; /* wrapper brace */
        while (end > 0 && (args[end - 1] == ' ' || args[end - 1] == '\t' ||
                           args[end - 1] == '\n' || args[end - 1] == '\r'))
            end--;
        emit_args(stream, index, args, end);
        call->ended = 1;
        finalize_call(stream, index, call->name,
                      call->arguments ? call->arguments : "{}");
        if (stream->sink.on_call_end)
            stream->sink.on_call_end(stream->sink.ctx, index,
                                     stream->state[index].id);
    } else {
        size_t stable = region_len > TC_ARGS_HOLDBACK
                            ? region_len - TC_ARGS_HOLDBACK
                            : 0;
        emit_args(stream, index, args, stable);
    }
}

static void scan_calls(qwen_stream *stream, const char *from, int finishing) {
    const char *cursor = from;
    size_t index = 0;
    while ((cursor = strstr(cursor, TC_OPEN)) != NULL) {
        const char *body = cursor + TC_OPEN_LEN;
        const char *close = strstr(body, TC_CLOSE);
        size_t body_len = close ? (size_t)(close - body) : strlen(body);
        process_call(stream, index, body, body_len, close != NULL, finishing);
        index++;
        if (!close) break;
        cursor = close + TC_CLOSE_LEN;
    }
}

int qwen_stream_feed(qwen_stream *stream, const char *full_text) {
    if (!stream || !full_text) return 0;
    if (stream->failed) return 0;
    size_t total = strlen(full_text);

    const char *first = strstr(full_text, TC_OPEN);
    size_t text_region = first ? (size_t)(first - full_text) : total;

    if (text_region > stream->text_len) {
        char *grown = realloc(stream->text, text_region + 1);
        if (!grown) {
            stream->failed = 1;
            return 0;
        }
        stream->text = grown;
        memcpy(stream->text, full_text, text_region);
        stream->text[text_region] = '\0';
        stream->text_len = text_region;
    }

    size_t safe = text_region;
    if (!first) safe -= open_tag_tail(full_text, text_region);
    /* Hold back trailing whitespace: it is dropped if a <tool_call> follows
     * (matching qwen_tool_calls_parse) and flushed by qwen_stream_finish
     * otherwise. */
    while (safe > stream->text_emitted &&
           (stream->text[safe - 1] == ' ' || stream->text[safe - 1] == '\t' ||
            stream->text[safe - 1] == '\n' || stream->text[safe - 1] == '\r'))
        safe--;
    emit_text(stream, safe);

    if (first) {
        stream->in_markup = 1;
        scan_calls(stream, first, 0);
    }
    return !stream->failed;
}

void qwen_stream_finish(qwen_stream *stream) {
    if (!stream || stream->failed) return;
    if (!stream->in_markup) {
        emit_text(stream, stream->text_len);
        return;
    }
    /* Re-scan with `finishing` so a still-open last call is closed out. We need
     * the full text again; reconstruct is not stored, so rely on state: the
     * only open call is the highest index not ended. Nothing to feed here --
     * process_call already ran on the last feed with closed possibly false.
     * Force-close any unfinished call from its accumulated prefix. */
    for (size_t index = 0; index < stream->state_count; index++) {
        call_state *call = &stream->state[index];
        if (!call->begun || call->ended) continue;
        call->ended = 1;
        finalize_call(stream, index, call->name,
                      call->arguments ? call->arguments : "{}");
        if (stream->sink.on_call_end)
            stream->sink.on_call_end(stream->sink.ctx, index,
                                     stream->state[index].id);
    }
}

const h3_tool_call *qwen_stream_calls(const qwen_stream *stream,
                                      size_t *count) {
    if (count) *count = stream ? stream->calls_count : 0;
    return stream ? stream->calls : NULL;
}

const char *qwen_stream_text(const qwen_stream *stream) {
    return stream && stream->text ? stream->text : "";
}
