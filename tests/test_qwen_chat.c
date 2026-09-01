/* Phase 3 test -- ChatML rendering + one end-to-end templated turn.
 *
 *   1. exact-string render for system / user / assistant / tool roles and the
 *      generation prompt;
 *   2. consecutive tool messages fold into one user turn;
 *   3. tokenization puts <|im_start|> / <|im_end|> at the turn boundaries and
 *      round-trips through decode;
 *   4. a rendered [system, user] prompt run through the KV session decodes a
 *      short answer that stops at <|im_end|> within budget.
 *
 *   ./h3_qwen_chat_test MiniMax-H3
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_qwen_chat.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (!result) fail("path allocation failed");
    snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static char *render(const qwen_chat_message *messages, size_t count, int gen) {
    char error[256];
    char *text = NULL;
    if (!qwen_chat_render(messages, count, gen, &text, error, sizeof(error)))
        fail(error);
    return text;
}

static void expect_string(const char *got, const char *want, const char *what) {
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "  %s\n  --- got ---\n%s\n  --- want ---\n%s\n", what,
                got, want);
        fail("rendered ChatML does not match the expected string");
    }
}

static size_t count_token(const uint32_t *ids, size_t n, uint32_t token) {
    size_t total = 0;
    for (size_t index = 0; index < n; index++)
        if (ids[index] == token) total++;
    return total;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";

    /* (1) exact render */
    qwen_chat_message convo[] = {
        {QWEN_ROLE_SYSTEM, "You are terse."},
        {QWEN_ROLE_USER, "Hi"},
        {QWEN_ROLE_ASSISTANT, "Hello."},
        {QWEN_ROLE_USER, "Bye"},
    };
    char *text = render(convo, 4, 1);
    expect_string(text,
                  "<|im_start|>system\nYou are terse.<|im_end|>\n"
                  "<|im_start|>user\nHi<|im_end|>\n"
                  "<|im_start|>assistant\nHello.<|im_end|>\n"
                  "<|im_start|>user\nBye<|im_end|>\n"
                  "<|im_start|>assistant\n",
                  "system/user/assistant + generation prompt");
    free(text);

    /* no system, no generation prompt */
    qwen_chat_message plain[] = {{QWEN_ROLE_USER, "ping"}};
    text = render(plain, 1, 0);
    expect_string(text, "<|im_start|>user\nping<|im_end|>\n", "bare user turn");
    free(text);

    /* (2) consecutive tool messages fold into one user turn */
    qwen_chat_message tools[] = {
        {QWEN_ROLE_USER, "weather?"},
        {QWEN_ROLE_ASSISTANT, "checking"},
        {QWEN_ROLE_TOOL, "{\"temp\": 12}"},
        {QWEN_ROLE_TOOL, "{\"wind\": 5}"},
        {QWEN_ROLE_USER, "thanks"},
    };
    text = render(tools, 5, 1);
    expect_string(text,
                  "<|im_start|>user\nweather?<|im_end|>\n"
                  "<|im_start|>assistant\nchecking<|im_end|>\n"
                  "<|im_start|>user\n<tool_response>\n{\"temp\": 12}\n"
                  "</tool_response>\n<tool_response>\n{\"wind\": 5}\n"
                  "</tool_response><|im_end|>\n"
                  "<|im_start|>user\nthanks<|im_end|>\n"
                  "<|im_start|>assistant\n",
                  "tool-response folding");
    free(text);

    /* rejects a misplaced system message */
    qwen_chat_message misplaced[] = {{QWEN_ROLE_USER, "a"},
                                     {QWEN_ROLE_SYSTEM, "b"}};
    char error[256];
    char *bad = NULL;
    require(!qwen_chat_render(misplaced, 2, 0, &bad, error, sizeof(error)),
            "a non-leading system message must be rejected");

    /* (3) tokenization boundaries + round trip */
    char *tokenizer_path =
        path_join(model_root, "FL2VA/tokenizer/tokenizer.json");
    h3_tokenizer *tokenizer =
        h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);

    uint32_t *ids = NULL;
    size_t id_count = 0;
    if (!qwen_chat_tokenize(tokenizer, convo, 4, 1, &ids, &id_count, error,
                            sizeof(error)))
        fail(error);
    require(ids[0] == QWEN_TOKEN_IM_START, "rendered prompt must open with "
            "<|im_start|>");
    require(ids[id_count - 1] != QWEN_TOKEN_IM_END,
            "generation prompt must not end with <|im_end|>");
    /* system + user + assistant + user each open and close a turn, and the
     * generation prompt opens a fifth: 5 <|im_start|>, 4 <|im_end|>. */
    require(count_token(ids, id_count, QWEN_TOKEN_IM_START) == 5,
            "expected five <|im_start|> tokens");
    require(count_token(ids, id_count, QWEN_TOKEN_IM_END) == 4,
            "expected four <|im_end|> tokens");
    char *decoded =
        h3_tokenizer_decode(tokenizer, ids, id_count, error, sizeof(error));
    if (!decoded) fail(error);
    text = render(convo, 4, 1);
    expect_string(decoded, text, "tokenize/decode round trip");
    free(decoded);
    free(text);
    h3_tokenizer_ids_free(ids);
    printf("(1-3) ChatML render + tokenization checks passed\n");

    /* (4) end-to-end: one templated turn through the KV session */
    char *weights_path = path_join(model_root, "FL2VA/text_encoder");
    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, weights_path, "h3_shaders.metal", error,
                          sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);

    qwen_chat_message ask[] = {
        {QWEN_ROLE_SYSTEM, "You answer with a single word."},
        {QWEN_ROLE_USER, "What is the capital of France?"},
    };
    uint32_t *prompt_ids = NULL;
    size_t prompt_len = 0;
    if (!qwen_chat_tokenize(tokenizer, ask, 2, 1, &prompt_ids, &prompt_len,
                            error, sizeof(error)))
        fail(error);
    printf("prompt tokens: %zu\n", prompt_len);
    if (!qwen_session_eval(session, prompt_ids, prompt_len, error,
                           sizeof(error)))
        fail(error);

    enum { BUDGET = 12 };
    uint32_t reply[BUDGET];
    int reply_len = 0;
    int stopped = 0;
    for (int step = 0; step < BUDGET; step++) {
        uint32_t next = 0;
        if (!qwen_session_sample(session, &next, error, sizeof(error)))
            fail(error);
        if (next == QWEN_TOKEN_IM_END || next == QWEN_TOKEN_ENDOFTEXT) {
            stopped = 1;
            break;
        }
        reply[reply_len++] = next;
        if (!qwen_session_eval(session, &next, 1, error, sizeof(error)))
            fail(error);
    }
    require(stopped, "assistant turn did not stop at <|im_end|> within budget");
    char *answer =
        h3_tokenizer_decode(tokenizer, reply, (size_t)reply_len, error,
                            sizeof(error));
    if (!answer) fail(error);
    printf("assistant: \"%s\" (%d tokens, stopped on <|im_end|>)\n", answer,
           reply_len);
    int has_paris = strstr(answer, "Paris") || strstr(answer, "paris");
    require(has_paris, "templated chat turn did not answer \"Paris\"");
    free(answer);

    free(prompt_ids);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    puts("ok: qwen Phase 3 chat template (roles + tokenization + one turn)");
    return 0;
}
