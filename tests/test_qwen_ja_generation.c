/* QINT-009: does Mixed-W4/BF16 decode degrade plain Japanese chat vs BF16?
 *
 *   H3_QWEN_Q4=0     ./h3_qwen_ja_generation --emit bf16.txt
 *   H3_QWEN_Q4=mixed ./h3_qwen_ja_generation --emit mixed.txt
 *   ./h3_qwen_ja_generation --compare bf16.txt mixed.txt
 *
 * `make qint-009` runs all three. For each Japanese user turn it greedily
 * generates the assistant reply, then the compare reports the mechanical
 * gates -- non-empty, valid UTF-8, no runaway repetition, sane length --
 * plus BF16-vs-mixed leading-character agreement, and prints every pair so
 * a human can close the "meaning matches / Japanese still fluent" judgement.
 * This is the light task-level check, not a full eval.
 */

#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEN_CAP 384

static const char *PROMPTS[] = {
    "日本の首都はどこですか。一文で答えてください。",
    "りんごが3個あります。2個もらうと全部で何個ですか。考え方も一言そえてください。",
    "次の文を丁寧語に直してください。「明日きて」",
    "光合成とは何か、中学生にもわかるように2文で説明してください。",
    "週末に京都でやることを3つ、箇条書きで提案してください。",
    "「犬も歩けば棒に当たる」ということわざの意味を説明してください。",
    "英語の \"Please send me the report by Friday.\" を自然な日本語に訳してください。",
    "在宅勤務の利点と欠点を、それぞれ1つずつ挙げてください。",
    "コーヒーと紅茶では、一般にどちらがカフェインが多いですか。",
    "5、10、15、20 の次に来る数は何ですか。理由も述べてください。",
};
#define NCASES ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_ja_generation.c: %s\n", m);
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

/* Greedy assistant turn for one prompt; returns malloc'd UTF-8. */
static char *run_case(qwen_session *session, const h3_tokenizer *tok, int c) {
    char error[512];
    uint32_t *ids = NULL;
    size_t n = 0;
    qwen_chat_message msg = {QWEN_ROLE_USER, PROMPTS[c], NULL};
    if (!qwen_chat_tokenize(tok, &msg, 1, 1, &ids, &n, error, sizeof(error)))
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
        fprintf(stderr, "  case %d: %.120s%s\n", c, text,
                strlen(text) > 120 ? "..." : "");
        free(text);
    }
    fclose(f);
    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_free(tok);
    free(tp);
    free(wp);
    printf("emit: %d Japanese chat turns -> %s\n", NCASES, out_path);
}

static char **read_emit(const char *path) {
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
    return rows;
}

/* Count UTF-8 code points; also report if the byte stream is well-formed. */
static size_t utf8_count(const char *s, int *well_formed) {
    size_t cp = 0;
    const unsigned char *p = (const unsigned char *)s;
    *well_formed = 1;
    while (*p) {
        int extra;
        if (*p < 0x80) extra = 0;
        else if ((*p >> 5) == 0x6) extra = 1;
        else if ((*p >> 4) == 0xE) extra = 2;
        else if ((*p >> 3) == 0x1E) extra = 3;
        else { *well_formed = 0; return cp; }
        for (int k = 1; k <= extra; k++) {
            if ((p[k] >> 6) != 0x2) { *well_formed = 0; return cp; }
        }
        p += extra + 1;
        cp++;
    }
    return cp;
}

/* Longest run of one repeated code point, and longest immediately-repeated
 * short phrase (2..12 code points repeated back-to-back). Catches the two
 * runaway-decode failure shapes. */
static void repetition_stats(const char *s, int *max_cp_run,
                             int *max_phrase_reps) {
    /* Build a small array of code-point byte offsets. */
    size_t cap = strlen(s) + 1;
    size_t *off = malloc(cap * sizeof(*off));
    if (!off) fail("alloc");
    size_t ncp = 0;
    const unsigned char *p = (const unsigned char *)s;
    const unsigned char *base = p;
    while (*p) {
        off[ncp++] = (size_t)(p - base);
        int extra = *p < 0x80 ? 0 : (*p >> 5) == 0x6 ? 1 : (*p >> 4) == 0xE
                                                               ? 2
                                                               : 3;
        p += extra + 1;
    }
    off[ncp] = (size_t)(p - base);

    int run = ncp ? 1 : 0, best_run = run;
    for (size_t i = 1; i < ncp; i++) {
        size_t la = off[i] - off[i - 1], lb = off[i + 1] - off[i];
        if (la == lb && !memcmp(base + off[i - 1], base + off[i], la))
            run++;
        else
            run = 1;
        if (run > best_run) best_run = run;
    }
    *max_cp_run = best_run;

    int best_phrase = 1;
    for (int plen = 2; plen <= 12; plen++) {
        if ((size_t)(2 * plen) > ncp) break;
        for (size_t i = 0; i + (size_t)plen <= ncp; i++) {
            size_t a0 = off[i], a1 = off[i + (size_t)plen];
            int reps = 1;
            size_t j = i + (size_t)plen;
            while (j + (size_t)plen <= ncp) {
                size_t b0 = off[j], b1 = off[j + (size_t)plen];
                if (b1 - b0 != a1 - a0 ||
                    memcmp(base + a0, base + b0, a1 - a0))
                    break;
                reps++;
                j += (size_t)plen;
            }
            if (reps > best_phrase) best_phrase = reps;
        }
    }
    *max_phrase_reps = best_phrase;
    free(off);
}

static void compare(const char *bf16_path, const char *mixed_path) {
    char **b = read_emit(bf16_path);
    char **m = read_emit(mixed_path);

    int ok_nonempty = 0, ok_utf8 = 0, ok_norepeat = 0, ok_len = 0;
    int exact = 0;
    long total_prefix = 0, total_min = 0;

    for (int c = 0; c < NCASES; c++) {
        int wf_b = 0, wf_m = 0;
        size_t cpb = utf8_count(b[c], &wf_b);
        size_t cpm = utf8_count(m[c], &wf_m);
        int run_b = 0, ph_b = 0, run_m = 0, ph_m = 0;
        repetition_stats(b[c], &run_b, &ph_b);
        repetition_stats(m[c], &run_m, &ph_m);

        int nonempty = cpm >= 2;
        int utf8_ok = wf_m;
        /* >=8 identical code points in a row, or a short phrase repeated
         * >=4 times back-to-back, is a runaway decode. */
        int norepeat = run_m < 8 && ph_m < 4;
        /* Anti-degeneration only: a non-repeating answer of any length is
         * fine (some prompts ask for 3 explained bullet points). The ceiling
         * just catches a pathological non-tight run-on. */
        int len_ok = cpm >= 2 && cpm <= 600;

        ok_nonempty += nonempty;
        ok_utf8 += utf8_ok;
        ok_norepeat += norepeat;
        ok_len += len_ok;

        size_t lb = strlen(b[c]), lm = strlen(m[c]);
        size_t pre = 0, lim = lb < lm ? lb : lm;
        while (pre < lim && b[c][pre] == m[c][pre]) pre++;
        if (lb == lm && pre == lb) exact++;
        total_prefix += (long)pre;
        total_min += (long)lim;

        printf("--- case %d ---\n  prompt: %s\n  bf16 (%zu cp): %s\n"
               "  mixed(%zu cp): %s\n"
               "  bf16  wf=%d run=%d phrase=%d\n"
               "  mixed wf=%d run=%d phrase=%d  leading match %zu/%zu%s\n",
               c, PROMPTS[c], cpb, b[c], cpm, m[c], wf_b, run_b, ph_b, wf_m,
               run_m, ph_m, pre, lim,
               (lb == lm && pre == lb) ? "  (exact)" : "");

        require(nonempty, "mixed JA reply is empty");
        require(utf8_ok, "mixed JA reply is not valid UTF-8");
        require(norepeat, "mixed JA reply has a runaway repetition");
        require(len_ok, "mixed JA reply is pathologically long");
    }

    printf("\nQINT-009 Japanese chat generation (%d cases):\n", NCASES);
    printf("  non-empty                 : %d/%d\n", ok_nonempty, NCASES);
    printf("  valid UTF-8               : %d/%d\n", ok_utf8, NCASES);
    printf("  no runaway repetition     : %d/%d\n", ok_norepeat, NCASES);
    printf("  length not pathological   : %d/%d\n", ok_len, NCASES);
    printf("  exact-match vs bf16       : %d/%d\n", exact, NCASES);
    printf("  leading-char agreement    : %ld/%ld (%.0f%%)\n", total_prefix,
           total_min,
           100.0 * (double)total_prefix / (double)(total_min ? total_min : 1));
    puts("ok: QINT-009 Japanese generation (mechanical gates); "
         "eyeball the pairs above for meaning + fluency");

    for (int c = 0; c < NCASES; c++) { free(b[c]); free(m[c]); }
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
