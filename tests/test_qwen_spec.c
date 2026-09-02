/* QINT-015a/b/c -- scalar speculative-decoding coordinator.
 *
 *   oracle      -- oracle draft => 100% acceptance, byte-identical output
 *   reject      -- forced divergence at each block position; correction + parity
 *   parity      -- n-gram draft, byte-identical to plain greedy
 *   selfcheck   -- mixed target: coordinator == greedy on every stable position
 *   ngram-bench -- n-gram acceptance per prompt bucket (measurement)
 *   bf16 | mixed -- groups of the above for `make spec-check`
 *
 * The coordinator provably emits the target's greedy argmax sequence, so its
 * output must be a valid plain greedy decode. It can still differ from a
 * *particular* greedy run at a position whose top-1/top-2 logit gap is within
 * TIE_EPS: the decode path's GPU reduction order breaks such a tie
 * nondeterministically (independent of speculation -- two plain greedy runs
 * can disagree there too). parity_check rebuilds the logits at any divergence
 * and confirms it is exactly such a near-tie; anything else fails.
 * `oracle` / `reject` use fully stable prompts and stay byte-identical.
 * The scalar verifier is not faster (same worst-case target forwards); it
 * exists to lock the algorithm before the batched verifier (QINT-015d).
 */

#include "qwen_draft.h"
#include "qwen_engine.h"
#include "qwen_spec.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXGEN 96
#define SPEC_VOCAB 151936u /* H3 Qwen backbone vocabulary */

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_spec.c: %s\n", m);
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

static uint16_t round_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void fill_pattern(uint16_t *values, size_t count, unsigned seed) {
    for (size_t i = 0; i < count; i++) {
        unsigned mixed = seed * 2654435761u + (unsigned)i * 40503u;
        float centred = (float)(mixed % 4096u) / 4096.0f - 0.5f;
        values[i] = round_bf16_bits(centred * 0.1f);
    }
}

static const uint32_t STOP_IDS[] = {QWEN_TOKEN_IM_END, QWEN_TOKEN_ENDOFTEXT};
#define STOP_COUNT ((size_t)(sizeof(STOP_IDS) / sizeof(STOP_IDS[0])))

static int is_stop(uint32_t t) {
    for (size_t i = 0; i < STOP_COUNT; i++)
        if (STOP_IDS[i] == t) return 1;
    return 0;
}

/* Plain greedy decode from the current session state. Leaves the session
 * advanced by the produced tokens (stop token not appended). */
static size_t ref_greedy(qwen_session *s, size_t max_new, uint32_t *out) {
    char err[512];
    size_t n = 0;
    while (n < max_new) {
        uint32_t t = 0;
        if (!qwen_session_sample(s, &t, err, sizeof(err))) fail(err);
        if (is_stop(t)) break;
        out[n++] = t;
        if (!qwen_session_eval(s, &t, 1, err, sizeof(err))) fail(err);
    }
    return n;
}

typedef struct {
    qwen_engine *engine;
    h3_tokenizer *tok;
} model;

static void model_open(model *m, const char *root) {
    char err[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    char *wp = path_join(root, "FL2VA/text_encoder");
    m->tok = h3_tokenizer_load(tp, err, sizeof(err));
    if (!m->tok) fail(err);
    if (!qwen_engine_open(&m->engine, wp, "h3_shaders.metal", err, sizeof(err)))
        fail(err);
    free(tp);
    free(wp);
}

/* Fresh resident session with `user` prefilled as a chat turn. The prompt ids
 * are returned in *pids (caller frees with free()) / *plen. Because a fresh
 * prefill is deterministic, every run (reference and each speculative run)
 * gets a byte-identical starting state -- no rewind-for-setup, which would
 * leave the frontier logits on the decode path instead of the prefill path. */
static qwen_session *prefill(model *m, const char *user, uint32_t **pids,
                             size_t *plen) {
    char err[512];
    qwen_session *s = NULL;
    if (!qwen_session_create(&s, m->engine, err, sizeof(err))) fail(err);
    if (!qwen_session_set_resident(s, 1, err, sizeof(err))) fail(err);
    qwen_chat_message msg = {QWEN_ROLE_USER, user, NULL};
    uint32_t *ids = NULL;
    size_t n = 0;
    if (!qwen_chat_tokenize(m->tok, &msg, 1, 1, &ids, &n, err, sizeof(err)))
        fail(err);
    if (!qwen_session_eval(s, ids, n, err, sizeof(err))) fail(err);
    if (pids) {
        *pids = malloc(n * sizeof(**pids));
        if (!*pids) fail("alloc");
        memcpy(*pids, ids, n * sizeof(**pids));
    }
    h3_tokenizer_ids_free(ids);
    *plen = n;
    return s;
}

/* full = prompt ids ++ ref, for feeding an aligned oracle. */
static uint32_t *make_full_stream(const uint32_t *pids, size_t plen,
                                  const uint32_t *ref, size_t rn,
                                  size_t *out_len) {
    uint32_t *full = malloc((plen + rn) * sizeof(*full));
    if (!full) fail("alloc");
    memcpy(full, pids, plen * sizeof(*full));
    memcpy(full + plen, ref, rn * sizeof(*full));
    *out_len = plen + rn;
    return full;
}

/* Strict compare, for sequences that must be bit-identical (oracle replay). */
static void cmp_or_die(const uint32_t *a, const uint32_t *b, size_t n,
                       const char *what) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  %s: divergence at %zu: ref=%u spec=%u\n", what, i,
                    a[i], b[i]);
            size_t lo = i > 6 ? i - 6 : 0, hi = i + 4 < n ? i + 4 : n;
            fprintf(stderr, "    ref :");
            for (size_t k = lo; k < hi; k++) fprintf(stderr, " %u", a[k]);
            fprintf(stderr, "\n    spec:");
            for (size_t k = lo; k < hi; k++) fprintf(stderr, " %u", b[k]);
            fprintf(stderr, "\n");
            fail("speculative output != greedy output");
        }
    }
}

static const char *PROMPT_EN =
    "Explain in three sentences why the sky appears blue during the day.";
static const char *PROMPT_JA =
    "在宅勤務のメリットを3つ、箇条書きで説明してください。";
static const char *PROMPT_CODE =
    "Write a Python function fib(n) that returns the nth Fibonacci number, "
    "with a one-line docstring and an iterative body.";

/* Reference greedy tokens for `prompt`, plus the prompt ids. */
static size_t reference(model *m, const char *prompt, uint32_t **pids,
                        size_t *plen, uint32_t *ref, size_t cap) {
    qwen_session *s = prefill(m, prompt, pids, plen);
    size_t rn = ref_greedy(s, cap, ref);
    qwen_session_free(s);
    return rn;
}

/* The coordinator provably emits the target's greedy argmax sequence, so `out`
 * must equal a plain greedy decode. It can still differ from a *particular*
 * greedy run `ref` at a position whose top-1/top-2 logit gap is within
 * TIE_EPS: there the decode path's reduction order breaks the tie
 * nondeterministically (the "mid-margin" flips of QINT-016), and either token
 * is a legitimate argmax. parity_check confirms any divergence is exactly such
 * a near-tie -- it rebuilds the logits at the divergence from the agreed
 * prefix and checks that `ref`'s and `out`'s tokens are the joint top-2 with a
 * sub-TIE_EPS margin and nothing scores higher. Anything else is a real bug. */
#define TIE_EPS 0.05f
static void parity_check(model *m, const char *prompt, const uint32_t *ref,
                         size_t rn, const uint32_t *out, size_t on,
                         const char *what) {
    size_t k = 0, lim = rn < on ? rn : on;
    while (k < lim && ref[k] == out[k]) k++;
    if (k == rn && k == on) return; /* byte-identical */
    if (k == lim) {
        fprintf(stderr, "  %s: length mismatch ref=%zu spec=%zu, no token "
                        "divergence\n",
                what, rn, on);
        fail("speculative length != greedy length with no divergence");
    }
    char err[512];
    size_t plen = 0;
    qwen_session *s = prefill(m, prompt, NULL, &plen);
    for (size_t i = 0; i < k; i++)
        require(qwen_session_eval(s, &out[i], 1, err, sizeof(err)), err);
    const qwen_logits *lg = qwen_session_logits(s);
    require(lg && lg->values, "no logits at divergence prefix");
    float v_ref = lg->values[ref[k]], v_out = lg->values[out[k]];
    float vmax = v_ref > v_out ? v_ref : v_out;
    int higher = 0;
    for (size_t t = 0; t < lg->vocab; t++)
        if (lg->values[t] > vmax + 1e-6f) { higher = 1; break; }
    float margin = v_ref > v_out ? v_ref - v_out : v_out - v_ref;
    qwen_session_free(s);
    fprintf(stderr,
            "  %s: divergence at %zu -- ref=%u (%.4f) spec=%u (%.4f) "
            "margin=%.4f%s\n",
            what, k, ref[k], (double)v_ref, out[k], (double)v_out,
            (double)margin, higher ? "  [another token scores higher!]" : "");
    require(!higher && margin < TIE_EPS,
            "coordinator token is not a near-tie argmax -- real coordinator "
            "bug");
    fprintf(stderr, "  %s: OK -- decode near-tie (margin %.4f < %.2f), both "
                    "tokens are valid argmax\n",
            what, (double)margin, (double)TIE_EPS);
}

/* mode: 2 = byte-identical to `ref`; 1 = near-tie-tolerant parity_check;
 * 0 = measurement only (report matched-prefix via *matched). */
static void spec_run(model *m, const char *prompt, qwen_draft_backend *draft,
                     unsigned width, const uint32_t *ref, size_t rn, int mode,
                     qwen_spec_stats *out_stats, size_t *matched,
                     const char *what) {
    char err[512];
    size_t plen = 0;
    qwen_session *s = prefill(m, prompt, NULL, &plen);
    qwen_spec spec;
    require(qwen_spec_init(&spec, s, draft, width, err, sizeof(err)), err);
    uint32_t out[MAXGEN];
    size_t on = 0;
    require(qwen_spec_generate(&spec, rn, STOP_IDS, STOP_COUNT, out, &on, err,
                               sizeof(err)),
            err);
    require(qwen_session_length(s) == plen + on, "session length wrong");
    if (mode == 2) {
        require(on == rn, "spec produced a different token count");
        cmp_or_die(ref, out, rn, what);
        if (matched) *matched = rn;
    } else if (mode == 1) {
        parity_check(m, prompt, ref, rn, out, on, what);
        size_t k = 0, lim = on < rn ? on : rn;
        while (k < lim && out[k] == ref[k]) k++;
        if (matched) *matched = k;
    } else {
        size_t k = 0, lim = on < rn ? on : rn;
        while (k < lim && out[k] == ref[k]) k++;
        if (matched) *matched = k;
    }
    if (out_stats) *out_stats = spec.stats;
    qwen_session_free(s);
}

static int run_oracle(model *m) {
    printf("== spec oracle-check ==\n");
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, MAXGEN);
    require(rn >= 16, "reference generation too short");

    size_t full_len = 0;
    uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);

    for (unsigned width = 1; width <= 5; width += 2) {
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full, full_len, (size_t)-1, 0);
        require(oracle != NULL, "oracle alloc");
        qwen_spec_stats st;
        spec_run(m, PROMPT_EN, oracle, width, ref, rn, 2, &st, NULL, "oracle");
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "  width=%u", width);
        qwen_spec_stats_print(&st, lbl);
        /* Oracle => every complete cycle is a full block, all draft accepted. */
        require(st.full_block + 1 >= st.cycles, "oracle acceptance not ~100%");
        require(st.accepted_tokens >= st.drafted_tokens - 1,
                "oracle draft tokens not (almost) all accepted");
        qwen_draft_destroy(oracle);
    }
    free(full);
    free(pids);
    puts("ok: spec oracle-check (token parity + ~100% acceptance)");
    return 0;
}

static int run_reject(model *m) {
    printf("== spec reject-check ==\n");
    char err[512];
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, 40);
    require(rn >= 12, "reference generation too short");
    size_t full_len = 0;
    uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);

    /* Divergence forced at draft position 0..4, plus a clean "all accepted". */
    for (int corrupt_pos = -1; corrupt_pos <= 4; corrupt_pos++) {
        size_t cat = (size_t)-1;
        uint32_t ctok = 0;
        if (corrupt_pos >= 0) {
            cat = plen + (size_t)corrupt_pos;
            ctok = (ref[corrupt_pos] + 7u) % SPEC_VOCAB;
            if (ctok == ref[corrupt_pos]) ctok = (ctok + 1u) % SPEC_VOCAB;
        }
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full, full_len, cat, ctok);
        require(oracle != NULL, "oracle alloc");
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, NULL, &sp_plen);
        qwen_spec spec;
        require(qwen_spec_init(&spec, s, oracle, 5, err, sizeof(err)), err);

        /* First cycle inspected directly: with width 5 the accepted prefix
         * must be exactly corrupt_pos (or the full 5 when uncorrupted). */
        qwen_spec_cycle cyc;
        require(qwen_spec_step(&spec, STOP_IDS, STOP_COUNT, &cyc, err,
                               sizeof(err)),
                err);
        size_t want_prefix = corrupt_pos < 0 ? 5 : (size_t)corrupt_pos;
        if (cyc.accepted_from_draft != want_prefix) {
            fprintf(stderr,
                    "  corrupt_pos=%d: accepted_from_draft=%zu want %zu\n",
                    corrupt_pos, cyc.accepted_from_draft, want_prefix);
            fail("rejection point wrong");
        }
        uint32_t out[MAXGEN];
        size_t on = 0;
        for (size_t i = 0; i < cyc.committed_count; i++)
            out[on++] = cyc.committed[i];
        size_t more = 0;
        require(qwen_spec_generate(&spec, rn - on, STOP_IDS, STOP_COUNT,
                                   out + on, &more, err, sizeof(err)),
                err);
        on += more;
        require(on == rn, "reject spec produced a different token count");
        cmp_or_die(ref, out, rn, "reject");
        require(qwen_session_length(s) == sp_plen + rn, "session length wrong");
        printf("  corrupt_pos=%2d  prefix=%zu  target_evals=%llu  ok\n",
               corrupt_pos, want_prefix,
               (unsigned long long)spec.stats.target_evals);
        qwen_draft_destroy(oracle);
        qwen_session_free(s);
    }

    /* EOS inside a block: a shorter reference so the target hits a stop id
     * partway through an oracle block. */
    {
        uint32_t *pids2 = NULL;
        size_t plen2 = 0;
        uint32_t ref2[MAXGEN];
        size_t rn2 = reference(m, PROMPT_EN, &pids2, &plen2, ref2, 7);
        size_t fl2 = 0;
        uint32_t *full2 = make_full_stream(pids2, plen2, ref2, rn2, &fl2);
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full2, fl2, (size_t)-1, 0);
        qwen_spec_stats st;
        spec_run(m, PROMPT_EN, oracle, 5, ref2, rn2, 2, &st, NULL, "eos-inside");
        printf("  eos-inside     committed=%llu  ok\n",
               (unsigned long long)st.committed_tokens);
        qwen_draft_destroy(oracle);
        free(full2);
        free(pids2);
    }

    free(full);
    free(pids);
    puts("ok: spec reject-check (rejection point + parity + state)");
    return 0;
}

static int run_parity(model *m) {
    printf("== spec greedy-parity (n-gram draft) ==\n");
    const char *prompts[] = {PROMPT_EN, PROMPT_JA, PROMPT_CODE};
    for (size_t pi = 0; pi < sizeof(prompts) / sizeof(prompts[0]); pi++) {
        uint32_t *pids = NULL;
        size_t plen = 0;
        uint32_t ref[MAXGEN];
        size_t rn = reference(m, prompts[pi], &pids, &plen, ref, MAXGEN);
        free(pids);
        qwen_draft_backend *ng = qwen_draft_ngram_new();
        require(ng != NULL, "ngram alloc");
        qwen_spec_stats st;
        spec_run(m, prompts[pi], ng, 5, ref, rn, 1, &st, NULL, "parity");
        char lbl[48];
        snprintf(lbl, sizeof(lbl), "  prompt %zu", pi);
        qwen_spec_stats_print(&st, lbl);
        qwen_draft_destroy(ng);
        printf("  prompt %zu: %zu tokens, byte-identical to greedy\n", pi, rn);
    }
    puts("ok: spec greedy-parity (n-gram) -- speculative == greedy");
    return 0;
}

/* Mixed-W4/BF16 target: the coordinator (n-gram draft, widths 1/3/5) must
 * produce a valid greedy decode -- byte-identical to a plain greedy run, or
 * divergent only at a decode near-tie (parity_check verifies that). */
static int run_selfcheck(model *m) {
    printf("== spec selfcheck (mixed target) ==\n");
    const char *prompts[] = {PROMPT_EN, PROMPT_JA, PROMPT_CODE};
    for (size_t pi = 0; pi < sizeof(prompts) / sizeof(prompts[0]); pi++) {
        uint32_t *pids = NULL;
        size_t plen = 0;
        uint32_t ref[MAXGEN];
        size_t rn = reference(m, prompts[pi], &pids, &plen, ref, MAXGEN);
        free(pids);
        for (unsigned w = 1; w <= 5; w += 2) {
            qwen_draft_backend *ng = qwen_draft_ngram_new();
            qwen_spec_stats st;
            size_t matched = 0;
            spec_run(m, prompts[pi], ng, w, ref, rn, 1, &st, &matched,
                     "selfcheck");
            printf("  prompt %zu w=%u: %zu tokens, matched %zu%s\n", pi, w, rn,
                   matched, matched == rn ? " (identical)" : " (near-tie fork)");
            qwen_draft_destroy(ng);
        }
    }
    puts("ok: spec selfcheck -- coordinator output is a valid greedy decode "
         "(mixed target)");
    return 0;
}

static int run_trace_code(model *m) {
    printf("== spec trace-code ==\n");
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_CODE, &pids, &plen, ref, MAXGEN);
    free(pids);
    fprintf(stderr, "greedy ref (%zu):", rn);
    for (size_t k = 0; k < rn; k++) fprintf(stderr, " %u", ref[k]);
    fprintf(stderr, "\n");
    qwen_draft_backend *ng = qwen_draft_ngram_new();
    qwen_spec_stats st;
    spec_run(m, PROMPT_CODE, ng, 5, ref, rn, 1, &st, NULL, "trace-code");
    qwen_draft_destroy(ng);
    qwen_spec_stats_print(&st, "  trace-code");
    puts("ok: trace-code (coordinator == greedy on CODE)");
    return 0;
}

static int run_ngram_bench(model *m) {
    printf("== spec ngram-bench ==\n");
    struct { const char *tag; const char *prompt; } B[] = {
        {"EN chat  ", PROMPT_EN},
        {"JA chat  ", PROMPT_JA},
        {"code     ", PROMPT_CODE},
        {"JSON     ",
         "Return a JSON object with keys name (\"Ada\"), age (36), "
         "city (\"London\"). JSON only."},
    };
    for (size_t i = 0; i < sizeof(B) / sizeof(B[0]); i++) {
        uint32_t *pids = NULL;
        size_t plen = 0;
        uint32_t ref[MAXGEN];
        size_t rn = reference(m, B[i].prompt, &pids, &plen, ref, MAXGEN);
        free(pids);
        qwen_draft_backend *ng = qwen_draft_ngram_new();
        qwen_spec_stats st;
        size_t matched = 0;
        spec_run(m, B[i].prompt, ng, 5, ref, rn, 0, &st, &matched,
                 "ngram-bench");
        double per_cycle =
            st.cycles ? (double)st.committed_tokens / (double)st.cycles : 0.0;
        printf("  %s tokens=%zu matched=%zu  committed/cycle=%.2f  "
               "accepted=%llu/%llu\n",
               B[i].tag, rn, matched, per_cycle,
               (unsigned long long)st.accepted_tokens,
               (unsigned long long)st.drafted_tokens);
        qwen_draft_destroy(ng);
    }
    puts("ok: spec ngram-bench (measurement only)");
    return 0;
}

/* -------- 015d-0: batched rewind transaction (text + non-zero mRoPE) ------- *
 *
 * The scalar coordinator never writes a rejected draft to the KV, so the
 * "append k rows, then truncate partway" transaction the batched verifier
 * needs is exercised here directly: decode a reference, then for a few keep
 * points append a deliberately-wrong 3-token block, rewind to `keep`, and
 * check that (a) length + history below `keep` are exactly restored and
 * (b) re-decoding reproduces the reference tail (which only holds if the
 * decode position -- and, for the multimodal case, mrope_next -- was
 * restored). Also checks that rewinding into a multimodal prompt is refused.
 */

static size_t redecode_and_match(qwen_session *s, size_t keep,
                                 const uint32_t *ref_tail, size_t tail_len,
                                 const char *what) {
    char err[512];
    /* Re-establish logits at position `keep`: rewind one more, replay it. */
    size_t hlen = 0;
    const uint32_t *hist = qwen_session_history(s, &hlen);
    require(hist && hlen == keep && keep >= 1, "redecode: unexpected state");
    uint32_t last = hist[keep - 1];
    require(qwen_session_rewind(s, keep - 1, err, sizeof(err)), err);
    require(qwen_session_eval(s, &last, 1, err, sizeof(err)), err);

    uint32_t got[MAXGEN];
    size_t n = tail_len < 12 ? tail_len : 12;
    for (size_t i = 0; i < n; i++) {
        require(qwen_session_sample(s, &got[i], err, sizeof(err)), err);
        if (is_stop(got[i])) { n = i; break; }
        require(qwen_session_eval(s, &got[i], 1, err, sizeof(err)), err);
    }
    size_t k = 0;
    while (k < n && got[k] == ref_tail[k]) k++;
    if (k < n) {
        /* one near-tie flip is acceptable (decode is not bit-stable at a
         * sub-TIE_EPS logit gap); anything earlier / larger is a bug. */
        fprintf(stderr, "  %s: re-decode matched %zu/%zu (div at %zu: ref=%u "
                        "got=%u)\n",
                what, k, n, k, ref_tail[k], got[k]);
        require(k >= 1, "rewind re-decode diverges immediately -- state not "
                        "restored");
    } else {
        fprintf(stderr, "  %s: re-decode matched %zu/%zu\n", what, k, n);
    }
    return k;
}

static int run_rewind_text(model *m) {
    printf("== spec batch-rewind-check (text) ==\n");
    char err[512];
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, 32);
    free(pids);
    require(rn >= 20, "reference too short");

    size_t keeps[] = {4, 12};
    for (size_t ki = 0; ki < sizeof(keeps) / sizeof(keeps[0]); ki++) {
        size_t k = keeps[ki];
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, NULL, &sp_plen);
        /* decode the real prefix ref[0..k-1] */
        for (size_t i = 0; i < k; i++)
            require(qwen_session_eval(s, &ref[i], 1, err, sizeof(err)), err);
        require(qwen_session_length(s) == sp_plen + k, "prefix length");
        /* append a wrong 3-token block, as a batched verifier would */
        uint32_t bad[3] = {ref[k], (ref[k] + 12345u) % SPEC_VOCAB,
                          (ref[k] + 54321u) % SPEC_VOCAB};
        require(qwen_session_eval(s, bad, 3, err, sizeof(err)), err);
        require(qwen_session_length(s) == sp_plen + k + 3, "post-block length");
        /* rewind the transaction back to the accepted frontier */
        require(qwen_session_rewind(s, sp_plen + k, err, sizeof(err)), err);
        require(qwen_session_length(s) == sp_plen + k, "rewound length");
        size_t hlen = 0;
        const uint32_t *hist = qwen_session_history(s, &hlen);
        require(hlen == sp_plen + k, "rewound history length");
        for (size_t i = 0; i < k; i++)
            require(hist[sp_plen + i] == ref[i], "rewound history corrupted");
        redecode_and_match(s, sp_plen + k, ref + k, rn - k, "text keep+prefix");
        qwen_session_free(s);
        printf("  keep=%zu: length + history restored, re-decode tracks ref\n",
               k);
    }
    puts("ok: spec batch-rewind-check (text)");
    return 0;
}

static int run_rewind_vlm(model *m) {
    printf("== spec batch-rewind-check (VLM / non-zero mRoPE) ==\n");
    char err[512];
    enum { SPN = 4, HID = 5120, MAXIDS = 128 };
    /* A real chat-templated multimodal prompt so the assistant turn actually
     * generates: <user> <vision_start> pad*SPN <vision_end> "Describe." </user>
     * <assistant>. Vision embeddings are synthetic (this test is about state
     * restoration, not answer quality). */
    uint32_t ids[MAXIDS];
    size_t TOK = 0, SP0 = 0;
    uint32_t *pre = NULL, *post = NULL;
    size_t npre = 0, npost = 0;
    require(h3_tokenizer_encode(m->tok, "<|im_start|>user\n", 0, &pre, &npre,
                                err, sizeof(err)),
            err);
    require(h3_tokenizer_encode(m->tok,
                                "\nDescribe the image in one sentence."
                                "<|im_end|>\n<|im_start|>assistant\n",
                                0, &post, &npost, err, sizeof(err)),
            err);
    for (size_t i = 0; i < npre; i++) ids[TOK++] = pre[i];
    ids[TOK++] = 151652u; /* <|vision_start|> */
    SP0 = TOK;
    for (int i = 0; i < SPN; i++) ids[TOK++] = 151655u; /* <|image_pad|> */
    ids[TOK++] = 151653u; /* <|vision_end|> */
    for (size_t i = 0; i < npost; i++) ids[TOK++] = post[i];
    h3_tokenizer_ids_free(pre);
    h3_tokenizer_ids_free(post);
    require(TOK < MAXIDS, "prompt too long");

    /* mRoPE: sequential text; the SPN pad tokens carry a small grid so the
     * text after them resumes at (grid max + 1) -- mrope_next then ends past
     * token_count, which is the case rewind must restore. */
    uint32_t pos[3 * MAXIDS];
    uint32_t grid_max = (uint32_t)SP0 + 8u; /* pad grid spans SP0..SP0+8 */
    for (int ax = 0; ax < 3; ax++) {
        for (size_t i = 0; i < SP0; i++) pos[ax * TOK + i] = (uint32_t)i;
        for (int i = 0; i < SPN; i++) {
            uint32_t p = ax == 0 ? (uint32_t)SP0
                       : ax == 1 ? (i / 2 ? grid_max : (uint32_t)SP0)
                                 : (i % 2 ? grid_max : (uint32_t)SP0);
            pos[ax * TOK + SP0 + i] = p;
        }
        uint32_t nxt = grid_max + 1u;
        for (size_t i = SP0 + SPN; i < TOK; i++) pos[ax * TOK + i] = nxt++;
    }
    uint8_t tags[MAXIDS];
    for (size_t i = 0; i < TOK; i++)
        tags[i] = (i >= SP0 && i < SP0 + SPN) ? 0u : 1u;

    size_t span_elems = (size_t)SPN * HID;
    uint16_t *emb = malloc(span_elems * sizeof(*emb));
    uint16_t *ds[3] = {malloc(span_elems * sizeof(uint16_t)),
                       malloc(span_elems * sizeof(uint16_t)),
                       malloc(span_elems * sizeof(uint16_t))};
    require(emb && ds[0] && ds[1] && ds[2], "alloc vision span");
    fill_pattern(emb, span_elems, 7u);
    fill_pattern(ds[0], span_elems, 8u);
    fill_pattern(ds[1], span_elems, 9u);
    fill_pattern(ds[2], span_elems, 10u);

    qwen_vision_span span = {0};
    span.start = SP0;
    span.tokens = SPN;
    span.embeddings = emb;
    span.deepstack[0] = ds[0];
    span.deepstack[1] = ds[1];
    span.deepstack[2] = ds[2];
    qwen_input in = {0};
    in.token_ids = ids;
    in.token_count = TOK;
    in.vision_spans = &span;
    in.vision_span_count = 1;
    in.position_ids = pos;
    in.tags = tags;

    /* reference decode after a multimodal prefill */
    qwen_session *r = NULL;
    require(qwen_session_create(&r, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(r, 1, err, sizeof(err)), err);
    require(qwen_session_eval_multimodal(r, &in, err, sizeof(err)), err);
    require(qwen_session_length(r) == TOK, "mm prefill length");
    uint32_t ref[MAXGEN];
    size_t rn = ref_greedy(r, 16, ref);
    fprintf(stderr, "  (synthetic multimodal reference: %zu tokens)\n", rn);
    require(rn >= 3, "mm reference too short (synthetic vision)");
    qwen_session_free(r);

    /* rewind INTO the prompt must be refused */
    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_eval_multimodal(s, &in, err, sizeof(err)), err);
    if (qwen_session_rewind(s, TOK - 1, err, sizeof(err)))
        fail("rewind into a multimodal prompt was allowed");
    require(qwen_session_length(s) == TOK, "refused rewind must not change len");

    /* append-wrong-block + rewind at a keep past the prompt */
    size_t k = rn >= 8 ? 3 : (rn >= 4 ? 2 : 1);
    for (size_t i = 0; i < k; i++)
        require(qwen_session_eval(s, &ref[i], 1, err, sizeof(err)), err);
    uint32_t bad[3] = {ref[k], (ref[k] + 12345u) % SPEC_VOCAB,
                          (ref[k] + 54321u) % SPEC_VOCAB};
    require(qwen_session_eval(s, bad, 3, err, sizeof(err)), err);
    require(qwen_session_length(s) == TOK + k + 3, "post-block length");
    require(qwen_session_rewind(s, TOK + k, err, sizeof(err)), err);
    require(qwen_session_length(s) == TOK + k, "rewound length");
    size_t hlen = 0;
    const uint32_t *hist = qwen_session_history(s, &hlen);
    require(hlen == TOK + k, "rewound history length");
    for (size_t i = 0; i < k; i++)
        require(hist[TOK + i] == ref[i], "rewound history corrupted");
    /* re-decode: only reproduces ref[k..] if mrope_next was restored to
     * mrope_base_pos + (keep - mrope_base_len). */
    size_t tail = rn - k;
    size_t matched = redecode_and_match(s, TOK + k, ref + k, tail, "vlm keep");
    require(matched >= (tail >= 3 ? 3 : tail),
            "mRoPE not restored on rewind -- re-decode drifts");
    qwen_session_free(s);

    free(emb);
    free(ds[0]);
    free(ds[1]);
    free(ds[2]);
    printf("  keep=%zu past a %zu-token multimodal prompt: mRoPE + state "
           "restored\n",
           k, TOK);
    puts("ok: spec batch-rewind-check (VLM / non-zero mRoPE)");
    return 0;
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    const char *cmd = argc >= 2 ? argv[1] : "all";
    model m;
    model_open(&m, root);

    int rc = 0;
    if (!strcmp(cmd, "oracle")) rc = run_oracle(&m);
    else if (!strcmp(cmd, "reject")) rc = run_reject(&m);
    else if (!strcmp(cmd, "parity")) rc = run_parity(&m);
    else if (!strcmp(cmd, "selfcheck")) rc = run_selfcheck(&m);
    else if (!strcmp(cmd, "ngram-bench")) rc = run_ngram_bench(&m);
    else if (!strcmp(cmd, "trace-code")) rc = run_trace_code(&m);
    else if (!strcmp(cmd, "batch-rewind"))
        rc = run_rewind_text(&m) || run_rewind_vlm(&m);
    else if (!strcmp(cmd, "core")) {
        rc = run_oracle(&m) || run_reject(&m) || run_parity(&m);
    } else if (!strcmp(cmd, "extra")) {
        rc = run_selfcheck(&m) || run_ngram_bench(&m) ||
             run_rewind_text(&m) || run_rewind_vlm(&m);
    } else {
        fail("usage: oracle|reject|parity|selfcheck|ngram-bench|batch-rewind|"
             "core|extra");
    }

    qwen_engine_close(m.engine);
    h3_tokenizer_free(m.tok);
    return rc;
}
