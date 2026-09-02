/* QINT-011: does Mixed-W4/BF16 decode degrade tool calling vs BF16?
 *
 *   H3_QWEN_Q4=0     ./h3_qwen_tool_parity --emit bf16.txt
 *   H3_QWEN_Q4=mixed ./h3_qwen_tool_parity --emit mixed.txt
 *   ./h3_qwen_tool_parity --compare bf16.txt mixed.txt
 *
 * `make qint-011` runs all three. For each tool-use prompt it greedily
 * generates the assistant turn, then compares:
 *   - emitted a <tool_call> (or not) -- both agree?
 *   - selected tool name -- parity?
 *   - arguments parse as a JSON object -- valid-JSON rate
 *   - arguments byte-identical after canonicalisation -- exact-match rate
 */

#include "h3_json.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"
#include "qwen_tools.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEN_CAP 160

#define T_WEATHER \
"{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"," \
"\"description\":\"Get the current weather for a location\"," \
"\"parameters\":{\"type\":\"object\",\"properties\":{" \
"\"location\":{\"type\":\"string\"}," \
"\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}}," \
"\"required\":[\"location\"]}}}"
#define T_CALC \
"{\"type\":\"function\",\"function\":{\"name\":\"calculate\"," \
"\"description\":\"Evaluate a math expression\"," \
"\"parameters\":{\"type\":\"object\",\"properties\":{" \
"\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}"
#define T_SEARCH \
"{\"type\":\"function\",\"function\":{\"name\":\"search_web\"," \
"\"description\":\"Search the web for information\"," \
"\"parameters\":{\"type\":\"object\",\"properties\":{" \
"\"query\":{\"type\":\"string\"}," \
"\"num_results\":{\"type\":\"integer\"}},\"required\":[\"query\"]}}}"
#define T_EMAIL \
"{\"type\":\"function\",\"function\":{\"name\":\"send_email\"," \
"\"description\":\"Send an email\"," \
"\"parameters\":{\"type\":\"object\",\"properties\":{" \
"\"to\":{\"type\":\"string\"},\"subject\":{\"type\":\"string\"}," \
"\"body\":{\"type\":\"string\"}},\"required\":[\"to\",\"subject\",\"body\"]}}}"

static const struct {
    const char *user;
    const char *tools[4];
    int tool_count;
    const char *want; /* expected tool name, or NULL for "no tool" */
} CASES[] = {
    {"What's the weather in Tokyo?", {T_WEATHER}, 1, "get_weather"},
    {"What's the weather in Paris, in fahrenheit?", {T_WEATHER}, 1,
     "get_weather"},
    {"What is 47 multiplied by 89?", {T_CALC}, 1, "calculate"},
    {"Search for recent news about Mars rovers.", {T_SEARCH}, 1, "search_web"},
    {"What's the weather in Berlin right now?",
     {T_WEATHER, T_CALC, T_SEARCH}, 3, "get_weather"},
    {"Calculate the square root of 144.",
     {T_WEATHER, T_CALC, T_SEARCH}, 3, "calculate"},
    {"Find 3 articles about quantum computing.",
     {T_WEATHER, T_CALC, T_SEARCH}, 3, "search_web"},
    {"Email bob@example.com with subject 'Meeting' about tomorrow at 3pm.",
     {T_EMAIL, T_WEATHER}, 2, "send_email"},
    {"東京の天気を教えてください。", {T_WEATHER}, 1, "get_weather"},
    {"Hi there, how are you doing today?", {T_WEATHER, T_CALC}, 2, NULL},
};
#define NCASES ((int)(sizeof(CASES) / sizeof(CASES[0])))

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_tool_parity.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

/* Greedy assistant turn for one case; returns malloc'd UTF-8. */
static char *run_case(qwen_session *session, const h3_tokenizer *tok, int c) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    qwen_chat_message msg = {QWEN_ROLE_USER, CASES[c].user, NULL};
    if (!qwen_chat_tokenize_tools(tok, &msg, 1, CASES[c].tools,
                                  (size_t)CASES[c].tool_count, 1, &ids, &n,
                                  error, sizeof(error)))
        fail(error);

    require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
    if (!qwen_session_eval(session, ids, n, error, sizeof(error))) fail(error);
    h3_tokenizer_ids_free(ids);

    uint32_t gen[GEN_CAP];
    int g = 0;
    for (; g < GEN_CAP; g++) {
        uint32_t next = 0;
        if (!qwen_session_sample(session, &next, error, sizeof(error)))
            fail(error);
        if (next == QWEN_TOKEN_IM_END || next == QWEN_TOKEN_ENDOFTEXT) break;
        gen[g] = next;
        if (!qwen_session_eval(session, &next, 1, error, sizeof(error)))
            fail(error);
    }
    char *text = h3_tokenizer_decode(tok, gen, (size_t)g, error, sizeof(error));
    if (!text) fail(error);
    return text;
}

static void emit(const char *root, const char *out_path) {
    char error[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *wp = path_join(root, "FL2VA/text_encoder");
    h3_tokenizer *tok = h3_tokenizer_load(tp, error, sizeof(error));
    if (!tok) fail(error);
    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, wp, "h3_shaders.metal", error, sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 1, error, sizeof(error)))
        fail(error);

    FILE *f = fopen(out_path, "wb");
    require(f != NULL, "cannot create emit file");
    for (int c = 0; c < NCASES; c++) {
        char *text = run_case(session, tok, c);
        uint32_t len = (uint32_t)strlen(text);
        require(fwrite(&len, sizeof(len), 1, f) == 1 &&
                    fwrite(text, 1, len, f) == len,
                "emit write");
        fprintf(stderr, "  case %d: %.80s%s\n", c, text,
                strlen(text) > 80 ? "..." : "");
        free(text);
    }
    fclose(f);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(tp);
    free(wp);
    printf("emit: %d tool-use turns -> %s\n", NCASES, out_path);
}

static char **read_emit(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    require(f != NULL, "cannot open emit file");
    char **rows = calloc(NCASES, sizeof(*rows));
    if (!rows) fail("alloc");
    for (int c = 0; c < NCASES; c++) {
        uint32_t len = 0;
        require(fread(&len, sizeof(len), 1, f) == 1, "emit truncated");
        rows[c] = malloc(len + 1);
        require(rows[c] && fread(rows[c], 1, len, f) == len, "emit body");
        rows[c][len] = '\0';
    }
    fclose(f);
    *n_out = NCASES;
    return rows;
}

/* Parsed view of one assistant turn. */
typedef struct {
    int has_call;
    char *name;     /* tool name or NULL */
    char *args_raw; /* arguments string or NULL */
    int args_valid; /* arguments parse to a JSON object */
    char *args_canon; /* canonical (stringify) form, or NULL */
} turn;

static void parse_turn(const char *text, turn *t) {
    char error[256];
    memset(t, 0, sizeof(*t));
    h3_tool_call *calls = NULL;
    size_t count = 0;
    char *content = NULL;
    if (!qwen_tool_calls_parse(text, &calls, &count, &content, error,
                               sizeof(error))) {
        free(content);
        return;
    }
    free(content);
    if (count == 0) {
        h3_tool_calls_free(calls, count);
        return;
    }
    t->has_call = 1;
    t->name = calls[0].name ? strdup(calls[0].name) : NULL;
    t->args_raw = calls[0].arguments ? strdup(calls[0].arguments) : NULL;
    if (t->args_raw) {
        h3_json *j = h3_json_parse(t->args_raw, strlen(t->args_raw), error,
                                   sizeof(error));
        if (j && h3_json_is(j, H3_JSON_OBJECT)) {
            t->args_valid = 1;
            t->args_canon = h3_json_stringify(j);
        }
        if (j) h3_json_free(j);
    }
    h3_tool_calls_free(calls, count);
}

static void free_turn(turn *t) {
    free(t->name);
    free(t->args_raw);
    free(t->args_canon);
}

static void compare(const char *bf16_path, const char *mixed_path) {
    int nb = 0, nm = 0;
    char **b = read_emit(bf16_path, &nb);
    char **m = read_emit(mixed_path, &nm);
    require(nb == nm, "case count mismatch");

    int call_agree = 0, want_call_ok_b = 0, want_call_ok_m = 0;
    int name_parity = 0, name_want_b = 0, name_want_m = 0;
    int json_b = 0, json_m = 0, args_exact = 0, both_call = 0;

    printf("%-3s %-16s %-16s %-6s %-6s %-6s\n", "#", "bf16 tool", "mixed tool",
           "call=", "json", "arg=");
    for (int c = 0; c < nb; c++) {
        turn tb, tm;
        parse_turn(b[c], &tb);
        parse_turn(m[c], &tm);
        int want_call = CASES[c].want != NULL;

        if (tb.has_call == tm.has_call) call_agree++;
        if (tb.has_call == want_call) want_call_ok_b++;
        if (tm.has_call == want_call) want_call_ok_m++;
        if (want_call && tb.name && !strcmp(tb.name, CASES[c].want))
            name_want_b++;
        if (want_call && tm.name && !strcmp(tm.name, CASES[c].want))
            name_want_m++;

        const char *sel = "-";
        if (tb.has_call && tm.has_call) {
            both_call++;
            if (tb.name && tm.name && !strcmp(tb.name, tm.name)) name_parity++;
            if (tb.args_valid) json_b++;
            if (tm.args_valid) json_m++;
            if (tb.args_canon && tm.args_canon &&
                !strcmp(tb.args_canon, tm.args_canon))
                args_exact++;
            sel = (tb.name && tm.name && !strcmp(tb.name, tm.name)) ? "same"
                                                                   : "DIFF";
        } else if (!tb.has_call && !tm.has_call) {
            sel = "(text)";
        } else {
            sel = "MISMATCH";
        }
        printf("%-3d %-16s %-16s %-6s %-6s %-6s\n", c,
               tb.name ? tb.name : (tb.has_call ? "?" : "(none)"),
               tm.name ? tm.name : (tm.has_call ? "?" : "(none)"), sel,
               (tb.has_call && tm.has_call)
                   ? (tb.args_valid && tm.args_valid ? "both"
                      : tb.args_valid || tm.args_valid ? "one" : "none")
                   : "-",
               (tb.has_call && tm.has_call)
                   ? (tb.args_canon && tm.args_canon &&
                              !strcmp(tb.args_canon, tm.args_canon)
                          ? "yes"
                          : "no")
                   : "-");
        free_turn(&tb);
        free_turn(&tm);
    }

    printf("\nQINT-011 tool-calling parity (%d cases):\n", nb);
    printf("  tool-call emitted, bf16 vs mixed agree : %d/%d\n", call_agree,
           nb);
    printf("  correct call/no-call vs expectation    : bf16 %d/%d  mixed %d/%d\n",
           want_call_ok_b, nb, want_call_ok_m, nb);
    printf("  correct tool name vs expectation       : bf16 %d/%d  mixed %d/%d\n",
           name_want_b, nb - 1, name_want_m, nb - 1); /* one case wants none */
    printf("  tool-selection parity (both called)    : %d/%d\n", name_parity,
           both_call);
    printf("  valid-JSON args                        : bf16 %d/%d  mixed %d/%d\n",
           json_b, both_call, json_m, both_call);
    printf("  argument byte-exact (canonical)        : %d/%d\n", args_exact,
           both_call);
    puts("ok: QINT-011 tool parity");

    for (int c = 0; c < nb; c++) { free(b[c]); free(m[c]); }
    free(b);
    free(m);
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    if (argc >= 3 && !strcmp(argv[1], "--emit")) {
        emit(root, argv[2]);
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "--compare")) {
        compare(argv[2], argv[3]);
        return 0;
    }
    fail("usage: --emit FILE | --compare BF16_FILE MIXED_FILE");
}
