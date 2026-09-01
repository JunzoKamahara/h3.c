#ifndef H3_JSON_H
#define H3_JSON_H

/* Small read-only JSON parser for the OpenAI-compatible server (Phase 4).
 * Not a general document model: no number/string round-tripping, no mutation.
 * Serialization is done by hand in qwen_server.c with h3_json_escape(). */

#include <stddef.h>

typedef enum {
    H3_JSON_NULL,
    H3_JSON_BOOL,
    H3_JSON_NUMBER,
    H3_JSON_STRING,
    H3_JSON_ARRAY,
    H3_JSON_OBJECT
} h3_json_type;

typedef struct h3_json h3_json;

/* Parse `length` bytes of UTF-8 JSON. Returns NULL and fills `error` on
 * failure. Free the result with h3_json_free(). */
h3_json *h3_json_parse(const char *text, size_t length, char *error,
                       size_t error_size);
void h3_json_free(h3_json *value);

h3_json_type h3_json_type_of(const h3_json *value);
int h3_json_is(const h3_json *value, h3_json_type type);

/* Scalars. The *_or forms return `fallback` when `value` is NULL or the wrong
 * type. h3_json_string_value() returns a NUL-terminated pointer owned by the
 * tree, or NULL. */
int h3_json_bool_or(const h3_json *value, int fallback);
double h3_json_number_or(const h3_json *value, double fallback);
const char *h3_json_string_value(const h3_json *value);

/* Arrays. */
size_t h3_json_array_size(const h3_json *value);
const h3_json *h3_json_array_at(const h3_json *value, size_t index);

/* Objects. h3_json_object_get() returns NULL when absent. */
const h3_json *h3_json_object_get(const h3_json *value, const char *key);

/* Escape `input` for embedding between JSON string quotes (control chars,
 * quote, backslash; UTF-8 passes through). Returns a malloc'd NUL-terminated
 * string the caller frees, or NULL on allocation failure. */
char *h3_json_escape(const char *input);

/* Serialize `value` back to compact JSON (`,` / `:` separators, no spaces).
 * Numbers are printed with %g. Returns a malloc'd string the caller frees, or
 * NULL on failure. */
char *h3_json_stringify(const h3_json *value);

#endif
