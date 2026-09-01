#ifndef QWEN_TOOLS_H
#define QWEN_TOOLS_H

/* Phase 5 -- tool calling.
 *
 * The intermediate representation (spec section 20) plus a parser that lifts
 * Qwen <tool_call>...</tool_call> markup out of an assistant turn. Rendering
 * of the `tools` system block and assistant tool_calls lives in qwen_chat.c;
 * OpenAI JSON serialization lives in qwen_server.c. */

#include <stddef.h>

typedef struct {
    char *id;        /* synthesised, e.g. "call_0001" */
    char *name;      /* function name */
    char *arguments; /* JSON text (object) or a raw string, as the model emitted */
} h3_tool_call;

void h3_tool_calls_free(h3_tool_call *calls, size_t count);

/* Split an assistant turn into leading text and tool calls.
 *
 * `*out_content` receives everything before the first <tool_call> block,
 * trimmed of trailing whitespace (malloc'd, never NULL, may be "").
 * `*out_calls` / `*out_count` receive the parsed calls (count 0 and calls
 * NULL when there are none). Every id is synthesised. The caller frees
 * `*out_content` and `h3_tool_calls_free(*out_calls, *out_count)`. */
int qwen_tool_calls_parse(const char *assistant_text, h3_tool_call **out_calls,
                          size_t *out_count, char **out_content, char *error,
                          size_t error_size);

#endif
