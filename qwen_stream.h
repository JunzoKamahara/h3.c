#ifndef QWEN_STREAM_H
#define QWEN_STREAM_H

/* Phase 5/6 -- incremental splitter for a streaming assistant turn.
 *
 * Fed the cumulative decoded text after every token, it separates leading
 * assistant text from <tool_call>{...}</tool_call> blocks and reports both as
 * deltas, so a server can stream OpenAI-style tool_calls: a "begin" with the
 * function name, then `arguments` string fragments, then an "end" -- one such
 * sequence per call, with an incrementing index for parallel calls.
 *
 * Invariant: concatenating every on_call_arguments delta for a given index
 * yields exactly qwen_stream_calls()[index].arguments (the raw value text the
 * model wrote), and concatenating every on_text delta yields the leading text.
 */

#include "qwen_tools.h"

#include <stddef.h>

typedef struct {
    void *ctx;
    void (*on_text)(void *ctx, const char *delta);
    void (*on_call_begin)(void *ctx, size_t index, const char *id,
                          const char *name);
    void (*on_call_arguments)(void *ctx, size_t index, const char *id,
                              const char *delta);
    void (*on_call_end)(void *ctx, size_t index, const char *id);
} qwen_stream_sink;

typedef struct qwen_stream qwen_stream;

qwen_stream *qwen_stream_new(const qwen_stream_sink *sink);
void qwen_stream_free(qwen_stream *stream);

/* Feed the full decoded assistant text so far (must be a growing prefix of the
 * previous call's argument). Emits any new deltas through the sink. Returns 0
 * on allocation failure. */
int qwen_stream_feed(qwen_stream *stream, const char *full_text);

/* Flush held-back bytes and close any still-open call. Call once when
 * generation stops. */
void qwen_stream_finish(qwen_stream *stream);

/* The complete tool calls seen so far (raw argument text, synthesised ids). */
const h3_tool_call *qwen_stream_calls(const qwen_stream *stream, size_t *count);

/* The leading assistant text (before the first <tool_call>), NUL-terminated. */
const char *qwen_stream_text(const qwen_stream *stream);

#endif
