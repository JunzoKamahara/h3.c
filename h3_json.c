#include "h3_json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_json {
    h3_json_type type;
    union {
        int boolean;
        double number;
        char *string; /* STRING: value; OBJECT entries store keys separately */
        struct {
            h3_json **items;
            char **keys; /* OBJECT only, parallel to items; NULL for ARRAY */
            size_t count;
        } compound;
    } as;
};

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
    int depth;
} json_cursor;

#define JSON_MAX_DEPTH 64

static void json_fail(json_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size && !cursor->error[0])
        snprintf(cursor->error, cursor->error_size, "%s", message);
}

static void json_skip_ws(json_cursor *cursor) {
    while (cursor->at < cursor->end) {
        char c = *cursor->at;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') cursor->at++;
        else break;
    }
}

static h3_json *json_new(h3_json_type type) {
    h3_json *value = calloc(1, sizeof(*value));
    if (value) value->type = type;
    return value;
}

void h3_json_free(h3_json *value) {
    if (!value) return;
    if (value->type == H3_JSON_STRING) {
        free(value->as.string);
    } else if (value->type == H3_JSON_ARRAY || value->type == H3_JSON_OBJECT) {
        for (size_t index = 0; index < value->as.compound.count; index++) {
            h3_json_free(value->as.compound.items[index]);
            if (value->as.compound.keys) free(value->as.compound.keys[index]);
        }
        free(value->as.compound.items);
        free(value->as.compound.keys);
    }
    free(value);
}

static int json_hex4(const char *p, unsigned *out) {
    unsigned value = 0;
    for (int index = 0; index < 4; index++) {
        char c = p[index];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') value |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = value;
    return 1;
}

static void utf8_encode(unsigned codepoint, char **out) {
    char *o = *out;
    if (codepoint < 0x80) {
        *o++ = (char)codepoint;
    } else if (codepoint < 0x800) {
        *o++ = (char)(0xC0 | (codepoint >> 6));
        *o++ = (char)(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        *o++ = (char)(0xE0 | (codepoint >> 12));
        *o++ = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        *o++ = (char)(0x80 | (codepoint & 0x3F));
    } else {
        *o++ = (char)(0xF0 | (codepoint >> 18));
        *o++ = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        *o++ = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        *o++ = (char)(0x80 | (codepoint & 0x3F));
    }
    *out = o;
}

/* Parse a JSON string starting at the opening quote. Returns a malloc'd
 * NUL-terminated decoded string, or NULL on error. */
static char *json_parse_string_raw(json_cursor *cursor) {
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        json_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t capacity = 16;
    size_t length = 0;
    char *out = malloc(capacity);
    if (!out) {
        json_fail(cursor, "out of memory parsing JSON string");
        return NULL;
    }
#define PUSH(byte) do {                                                        \
    if (length + 5 > capacity) {                                               \
        size_t grown = capacity * 2;                                           \
        char *bigger = realloc(out, grown);                                    \
        if (!bigger) { free(out); json_fail(cursor, "out of memory");          \
            return NULL; }                                                     \
        out = bigger; capacity = grown;                                        \
    }                                                                         \
    out[length++] = (char)(byte);                                              \
} while (0)
    while (cursor->at < cursor->end) {
        char c = *cursor->at++;
        if (c == '"') {
            out[length] = '\0';
            return out;
        }
        if (c == '\\') {
            if (cursor->at >= cursor->end) break;
            char escape = *cursor->at++;
            switch (escape) {
                case '"': PUSH('"'); break;
                case '\\': PUSH('\\'); break;
                case '/': PUSH('/'); break;
                case 'b': PUSH('\b'); break;
                case 'f': PUSH('\f'); break;
                case 'n': PUSH('\n'); break;
                case 'r': PUSH('\r'); break;
                case 't': PUSH('\t'); break;
                case 'u': {
                    unsigned codepoint = 0;
                    if (cursor->end - cursor->at < 4 ||
                        !json_hex4(cursor->at, &codepoint)) {
                        free(out);
                        json_fail(cursor, "bad \\u escape in JSON string");
                        return NULL;
                    }
                    cursor->at += 4;
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF &&
                        cursor->end - cursor->at >= 6 &&
                        cursor->at[0] == '\\' && cursor->at[1] == 'u') {
                        unsigned low = 0;
                        if (json_hex4(cursor->at + 2, &low) && low >= 0xDC00 &&
                            low <= 0xDFFF) {
                            cursor->at += 6;
                            codepoint = 0x10000 +
                                        ((codepoint - 0xD800) << 10) +
                                        (low - 0xDC00);
                        }
                    }
                    char buffer[4];
                    char *tail = buffer;
                    utf8_encode(codepoint, &tail);
                    for (char *scan = buffer; scan < tail; scan++) PUSH(*scan);
                    break;
                }
                default:
                    free(out);
                    json_fail(cursor, "unknown escape in JSON string");
                    return NULL;
            }
        } else {
            PUSH(c);
        }
    }
#undef PUSH
    free(out);
    json_fail(cursor, "unterminated JSON string");
    return NULL;
}

static h3_json *json_parse_value(json_cursor *cursor);

static h3_json *json_parse_array(json_cursor *cursor) {
    cursor->at++; /* '[' */
    h3_json *array = json_new(H3_JSON_ARRAY);
    if (!array) {
        json_fail(cursor, "out of memory");
        return NULL;
    }
    json_skip_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == ']') {
        cursor->at++;
        return array;
    }
    for (;;) {
        h3_json *item = json_parse_value(cursor);
        if (!item) {
            h3_json_free(array);
            return NULL;
        }
        h3_json **grown = realloc(
            array->as.compound.items,
            (array->as.compound.count + 1) * sizeof(*grown));
        if (!grown) {
            h3_json_free(item);
            h3_json_free(array);
            json_fail(cursor, "out of memory");
            return NULL;
        }
        array->as.compound.items = grown;
        array->as.compound.items[array->as.compound.count++] = item;
        json_skip_ws(cursor);
        if (cursor->at >= cursor->end) break;
        if (*cursor->at == ',') {
            cursor->at++;
            continue;
        }
        if (*cursor->at == ']') {
            cursor->at++;
            return array;
        }
        break;
    }
    h3_json_free(array);
    json_fail(cursor, "malformed JSON array");
    return NULL;
}

static h3_json *json_parse_object(json_cursor *cursor) {
    cursor->at++; /* '{' */
    h3_json *object = json_new(H3_JSON_OBJECT);
    if (!object) {
        json_fail(cursor, "out of memory");
        return NULL;
    }
    json_skip_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == '}') {
        cursor->at++;
        return object;
    }
    for (;;) {
        json_skip_ws(cursor);
        char *key = json_parse_string_raw(cursor);
        if (!key) {
            h3_json_free(object);
            return NULL;
        }
        json_skip_ws(cursor);
        if (cursor->at >= cursor->end || *cursor->at != ':') {
            free(key);
            h3_json_free(object);
            json_fail(cursor, "expected ':' in JSON object");
            return NULL;
        }
        cursor->at++;
        h3_json *item = json_parse_value(cursor);
        if (!item) {
            free(key);
            h3_json_free(object);
            return NULL;
        }
        size_t count = object->as.compound.count;
        h3_json **items =
            realloc(object->as.compound.items, (count + 1) * sizeof(*items));
        char **keys =
            realloc(object->as.compound.keys, (count + 1) * sizeof(*keys));
        if (items) object->as.compound.items = items;
        if (keys) object->as.compound.keys = keys;
        if (!items || !keys) {
            free(key);
            h3_json_free(item);
            h3_json_free(object);
            json_fail(cursor, "out of memory");
            return NULL;
        }
        object->as.compound.items[count] = item;
        object->as.compound.keys[count] = key;
        object->as.compound.count = count + 1;
        json_skip_ws(cursor);
        if (cursor->at >= cursor->end) break;
        if (*cursor->at == ',') {
            cursor->at++;
            continue;
        }
        if (*cursor->at == '}') {
            cursor->at++;
            return object;
        }
        break;
    }
    h3_json_free(object);
    json_fail(cursor, "malformed JSON object");
    return NULL;
}

static h3_json *json_parse_value(json_cursor *cursor) {
    json_skip_ws(cursor);
    if (cursor->at >= cursor->end) {
        json_fail(cursor, "unexpected end of JSON");
        return NULL;
    }
    if (++cursor->depth > JSON_MAX_DEPTH) {
        json_fail(cursor, "JSON nested too deeply");
        cursor->depth--;
        return NULL;
    }
    h3_json *result = NULL;
    char c = *cursor->at;
    if (c == '{') {
        result = json_parse_object(cursor);
    } else if (c == '[') {
        result = json_parse_array(cursor);
    } else if (c == '"') {
        char *string = json_parse_string_raw(cursor);
        if (string) {
            result = json_new(H3_JSON_STRING);
            if (result) result->as.string = string;
            else free(string);
        }
    } else if (c == 't' || c == 'f') {
        int is_true = c == 't';
        const char *word = is_true ? "true" : "false";
        size_t word_length = is_true ? 4 : 5;
        if ((size_t)(cursor->end - cursor->at) >= word_length &&
            !strncmp(cursor->at, word, word_length)) {
            cursor->at += word_length;
            result = json_new(H3_JSON_BOOL);
            if (result) result->as.boolean = is_true;
        } else {
            json_fail(cursor, "malformed JSON literal");
        }
    } else if (c == 'n') {
        if ((size_t)(cursor->end - cursor->at) >= 4 &&
            !strncmp(cursor->at, "null", 4)) {
            cursor->at += 4;
            result = json_new(H3_JSON_NULL);
        } else {
            json_fail(cursor, "malformed JSON literal");
        }
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char *tail = NULL;
        double number = strtod(cursor->at, &tail);
        if (tail && tail != cursor->at) {
            cursor->at = tail;
            result = json_new(H3_JSON_NUMBER);
            if (result) result->as.number = number;
        } else {
            json_fail(cursor, "malformed JSON number");
        }
    } else {
        json_fail(cursor, "unexpected character in JSON");
    }
    cursor->depth--;
    return result;
}

h3_json *h3_json_parse(const char *text, size_t length, char *error,
                       size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!text) {
        if (error && error_size) snprintf(error, error_size, "no JSON input");
        return NULL;
    }
    /* Work on a NUL-terminated copy so strtod() can never scan past the end. */
    char *buffer = malloc(length + 1);
    if (!buffer) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    memcpy(buffer, text, length);
    buffer[length] = '\0';
    json_cursor cursor = {buffer, buffer + length, error, error_size, 0};
    h3_json *value = json_parse_value(&cursor);
    if (value) {
        json_skip_ws(&cursor);
        if (cursor.at != cursor.end) {
            h3_json_free(value);
            value = NULL;
            if (error && error_size && !error[0])
                snprintf(error, error_size, "trailing bytes after JSON value");
        }
    }
    free(buffer);
    return value;
}

h3_json_type h3_json_type_of(const h3_json *value) {
    return value ? value->type : H3_JSON_NULL;
}

int h3_json_is(const h3_json *value, h3_json_type type) {
    return value && value->type == type;
}

int h3_json_bool_or(const h3_json *value, int fallback) {
    if (!value) return fallback;
    if (value->type == H3_JSON_BOOL) return value->as.boolean;
    return fallback;
}

double h3_json_number_or(const h3_json *value, double fallback) {
    if (value && value->type == H3_JSON_NUMBER) return value->as.number;
    return fallback;
}

const char *h3_json_string_value(const h3_json *value) {
    if (value && value->type == H3_JSON_STRING) return value->as.string;
    return NULL;
}

size_t h3_json_array_size(const h3_json *value) {
    if (value && value->type == H3_JSON_ARRAY) return value->as.compound.count;
    return 0;
}

const h3_json *h3_json_array_at(const h3_json *value, size_t index) {
    if (!value || value->type != H3_JSON_ARRAY ||
        index >= value->as.compound.count)
        return NULL;
    return value->as.compound.items[index];
}

const h3_json *h3_json_object_get(const h3_json *value, const char *key) {
    if (!value || value->type != H3_JSON_OBJECT || !key) return NULL;
    for (size_t index = 0; index < value->as.compound.count; index++) {
        if (!strcmp(value->as.compound.keys[index], key))
            return value->as.compound.items[index];
    }
    return NULL;
}

char *h3_json_escape(const char *input) {
    if (!input) input = "";
    size_t worst = 6 * strlen(input) + 1;
    char *out = malloc(worst);
    if (!out) return NULL;
    char *write = out;
    for (const unsigned char *scan = (const unsigned char *)input; *scan;
         scan++) {
        unsigned char c = *scan;
        switch (c) {
            case '"': *write++ = '\\'; *write++ = '"'; break;
            case '\\': *write++ = '\\'; *write++ = '\\'; break;
            case '\b': *write++ = '\\'; *write++ = 'b'; break;
            case '\f': *write++ = '\\'; *write++ = 'f'; break;
            case '\n': *write++ = '\\'; *write++ = 'n'; break;
            case '\r': *write++ = '\\'; *write++ = 'r'; break;
            case '\t': *write++ = '\\'; *write++ = 't'; break;
            default:
                if (c < 0x20) {
                    write += sprintf(write, "\\u%04x", c);
                } else {
                    *write++ = (char)c;
                }
        }
    }
    *write = '\0';
    return out;
}
