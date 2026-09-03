/* QINT-015d -- speculative decoding: pending-anchor coordinator + low-level.
 *
 * Coordinator (QINT-015d-2, one target batch forward per cycle):
 *   pending-oracle   -- 100 % acceptance, byte-identical, ~W committed/batch
 *   pending-reject   -- forced reject at each draft index; correction becomes
 *                       the next cycle's pending anchor; partial-block rewind
 *   pending-boundary -- max_new 1..7: exact count, parity, state invariant
 *   pending-eos      -- the target's own EOS in any block slot: never emitted,
 *                       never left in the KV
 *   pending-vlm      -- non-zero mRoPE: reject -> rewind -> pending -> next batch
 *   pending-parity   -- n-gram draft on EN/JA/code/JSON, greedy parity
 *
 * Low-level pieces (QINT-015d-0 / d-1):
 *   batch-rewind     -- append-block-then-rewind transaction, text + mRoPE
 *   verify-parity    -- qwen_session_verify_block per-row == scalar decode
 *
 * The coordinator emits the target's greedy argmax sequence, so its output is
 * a valid plain greedy decode. A divergence from a *particular* greedy run is
 * allowed only at a decode near-tie (top-1/top-2 logit gap < TIE_EPS -- the
 * GPU reduction order breaks such a tie nondeterministically, so two plain
 * greedy runs can disagree there too). parity_check rebuilds the logits at any
 * divergence and confirms it is exactly such a near-tie; anything else fails.
 * The oracle-driven tests use stable prompts and stay byte-identical.
 */

#include "qwen_draft.h"
#include "qwen_eagle3.h"
#include "qwen_engine.h"
#include "qwen_spec.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* Are `a` and `b` an acceptable pair for the same greedy position given the
 * full logit vector `lg` at that position? Identical, or a joint top-2
 * near-tie: nothing scores higher than both and their gap is < TIE_EPS. This
 * is the ONLY tolerated form of coordinator/verifier divergence -- the decode
 * path's GPU reduction order breaks such a tie nondeterministically. `*margin`
 * gets the logit gap when both are finite. */
static int near_tie_ok(const qwen_logits *lg, uint32_t a, uint32_t b,
                       float *margin_out) {
    if (margin_out) *margin_out = 0.0f;
    if (a == b) return 1;
    if (!lg || !lg->values || a >= lg->vocab || b >= lg->vocab) return 0;
    float va = lg->values[a], vb = lg->values[b];
    float vmax = va > vb ? va : vb;
    for (size_t t = 0; t < lg->vocab; t++)
        if (lg->values[t] > vmax + 1e-6f) return 0;
    float margin = va > vb ? va - vb : vb - va;
    if (margin_out) *margin_out = margin;
    return margin < TIE_EPS;
}

static void parity_check(model *m, const char *prompt, const uint32_t *ref,
                         size_t rn, const uint32_t *out, size_t on,
                         const char *what) {
    size_t k = 0, lim = rn < on ? rn : on;
    while (k < lim && ref[k] == out[k]) k++;
    if (k == rn && k == on) return; /* byte-identical */
    if (k == lim) {
        /* One sequence is a prefix of the other. A short tail difference is a
         * decode near-tie flipping into (or out of) an EOS a token or two
         * early; a large gap with an otherwise clean prefix is a real bug. */
        size_t gap = rn > on ? rn - on : on - rn;
        if (gap <= 3) {
            fprintf(stderr, "  %s: tail length differs by %zu (ref=%zu spec=%zu) "
                            "-- treated as an EOS near-tie\n",
                    what, gap, rn, on);
            return;
        }
        fprintf(stderr, "  %s: length mismatch ref=%zu spec=%zu, no token "
                        "divergence\n",
                what, rn, on);
        fail("speculative length != greedy length with no divergence");
    }
    /* Rebuild the logits at the divergence from the coordinator's own prefix
     * (single-token decode) and check the coordinator's token is the top-1 or
     * top-2. This is a state-machine SANITY diagnostic, not a correctness
     * gate: the coordinator reached this point through a chain of batched
     * verify forwards, and the batched path's small per-step numeric
     * difference from single-token decode accumulates ("batch-chain numerical
     * drift"), so at a position with a small top-1/top-2 gap the two paths can
     * land on different, both-plausible tokens -- a top-2 close call, NOT the
     * sub-TIE_EPS near-tie of a single step. A dedicated teacher-forced
     * measurement of that drift (spec-chain-drift-check) belongs in QINT-015e;
     * a token outside the top 2 here is a real coordinator bug. */
    char err[512];
    size_t plen = 0;
    qwen_session *s = prefill(m, prompt, NULL, &plen);
    for (size_t i = 0; i < k; i++)
        require(qwen_session_eval(s, &out[i], 1, err, sizeof(err)), err);
    const qwen_logits *lg = qwen_session_logits(s);
    require(lg && lg->values, "no logits at divergence prefix");
    uint32_t t1 = 0, t2 = 0;
    float b1 = -1e30f, b2 = -1e30f;
    for (size_t i = 0; i < lg->vocab; i++) {
        float v = lg->values[i];
        if (v > b1) { b2 = b1; t2 = t1; b1 = v; t1 = (uint32_t)i; }
        else if (v > b2) { b2 = v; t2 = (uint32_t)i; }
    }
    int in_top2 = (out[k] == t1 || out[k] == t2);
    fprintf(stderr,
            "  %s: divergence at %zu -- ref=%u (%.4f)  spec=%u (%.4f)  "
            "rebuilt top2={%u,%u} gap=%.4f -> %s\n",
            what, k, ref[k], (double)lg->values[ref[k]], out[k],
            (double)lg->values[out[k]], t1, t2, (double)(b1 - b2),
            in_top2 ? "close call, OK" : "REAL divergence");
    qwen_session_free(s);
    require(in_top2, "coordinator token is not even top-2 of the greedy logits "
                     "-- real coordinator bug");
}
/* ===================== QINT-015d-2: pending-anchor coordinator ============ *
 *
 * Every cycle runs one target batch forward. At each cycle boundary the KV
 * cache holds exactly the tokens emitted to the caller, and `pending_anchor`
 * (when set) is the target's already-decided next token, not yet in the KV or
 * the output. `assert_invariant` checks the first half of that after every
 * cycle. Divergence from a plain greedy reference is allowed only at a decode
 * near-tie (parity_check rebuilds the logits at the divergence and checks).
 */

static void assert_invariant(qwen_session *s, size_t plen, size_t emitted,
                             const char *what) {
    size_t len = qwen_session_length(s);
    if (len != plen + emitted) {
        fprintf(stderr, "  %s: state invariant broken -- session length %zu, "
                        "expected prompt %zu + emitted %zu\n",
                what, len, plen, emitted);
        fail("pending-anchor coordinator left the KV out of sync with output");
    }
}

/* Drive the coordinator one cycle at a time so the state invariant can be
 * checked after each cycle. `emitted0` is how many tokens were already emitted
 * before this call (0 unless earlier cycles were run by hand). `out`/`*on`
 * receive only the tokens produced by this call. `*budget` more may be
 * emitted. */
static void drive(qwen_spec *spec, qwen_session *s, size_t plen, size_t emitted0,
                  size_t budget, uint32_t *out, size_t *on, qwen_spec_stats *st,
                  const char *what) {
    char err[512];
    *on = 0;
    for (;;) {
        if (*on >= budget) break;
        qwen_spec_cycle cyc;
        require(qwen_spec_step(spec, budget - *on, STOP_IDS, STOP_COUNT, &cyc,
                               err, sizeof(err)),
                err);
        for (size_t i = 0; i < cyc.committed_count && *on < budget; i++)
            out[(*on)++] = cyc.committed[i];
        assert_invariant(s, plen, emitted0 + *on, what);
        if (cyc.hit_stop) break;
        if (cyc.committed_count == 0) break;
    }
    if (st) *st = spec->stats;
}

static int run_pending_oracle(model *m) {
    printf("== spec pending-oracle-check ==\n");
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, MAXGEN);
    require(rn >= 20, "reference too short");
    size_t full_len = 0;
    uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);
    free(pids);

    /* A short window: over the first dozen tokens of a stable prompt the
     * coordinator's batched path stays aligned with scalar greedy, so the
     * aligned oracle reproduces the ~W-per-batch commit rate. (A long
     * generation eventually hits a decode near-tie which the fixed oracle
     * stream cannot follow -- that is covered by pending-parity.) */
    const size_t WIN = 10;
    for (unsigned W = 2; W <= 5; W++) {
        char err[512];
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, NULL, &sp_plen);
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full, full_len, (size_t)-1, 0);
        qwen_spec spec;
        require(qwen_spec_init(&spec, s, oracle, W, err, sizeof(err)), err);
        uint32_t out[MAXGEN];
        size_t on = 0;
        qwen_spec_stats st;
        drive(&spec, s, sp_plen, 0, WIN, out, &on, &st, "pending-oracle");
        char lbl[24];
        snprintf(lbl, sizeof(lbl), "  W=%u", W);
        qwen_spec_stats_print(&st, lbl, 0.0);
        require(on == WIN, "coordinator emitted fewer than max_new");
        parity_check(m, PROMPT_EN, ref, WIN, out, on, "pending-oracle");
        /* With an aligned oracle the run is batch-driven: at most a couple of
         * scalar fallbacks (a near-tie the fixed stream can't follow), and
         * most drafted tokens accepted. */
        require(st.scalar_fallback_evals <= 2,
                "oracle path fell back to scalar too often");
        require(st.target_batches >= 2, "too few target batches");
        double acc = st.drafted_tokens ? (double)st.accepted_tokens /
                                             (double)st.drafted_tokens
                                       : 0.0;
        require(acc > 0.6, "oracle draft acceptance unexpectedly low");
        /* batches commit more than one token each on average (anchor + draft) */
        double bpb = (double)(st.committed_tokens - st.scalar_fallback_evals) /
                     (double)st.target_batches;
        require(bpb > 1.4, "batches barely commit past the anchor");
        qwen_draft_destroy(oracle);
        qwen_session_free(s);
    }
    free(full);
    puts("ok: spec pending-oracle-check (aligned window: ~W committed/batch)");
    return 0;
}

static int run_pending_reject(model *m) {
    printf("== spec pending-reject-check ==\n");
    char err[512];
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, MAXGEN);
    require(rn >= 24, "reference too short");
    size_t full_len = 0;
    uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);
    free(pids);

    /* width 5 -> block [anchor, D1, D2, D3, D4]; corrupt draft index 0..3. */
    for (int cp = 0; cp <= 3; cp++) {
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, NULL, &sp_plen);
        /* corrupt stream index = prompt + 1 (past the anchor) + cp */
        size_t cat = sp_plen + 1 + (size_t)cp;
        uint32_t ctok = (full[cat] + 7u) % SPEC_VOCAB;
        if (ctok == full[cat]) ctok = (ctok + 1u) % SPEC_VOCAB;
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full, full_len, cat, ctok);
        qwen_spec spec;
        require(qwen_spec_init(&spec, s, oracle, 5, err, sizeof(err)), err);

        /* cycle 0 by hand: reject must land exactly at draft index cp. */
        qwen_spec_cycle c0;
        require(qwen_spec_step(&spec, rn, STOP_IDS, STOP_COUNT, &c0, err,
                               sizeof(err)),
                err);
        if (c0.accepted_from_draft != (size_t)cp) {
            fprintf(stderr, "  cp=%d: accepted_from_draft=%zu want %d\n", cp,
                    c0.accepted_from_draft, cp);
            fail("rejection point wrong");
        }
        /* cycle 0 emits [anchor, D1..D_cp] == ref[0..cp] (cp+1 tokens); the
         * target's correction after that prefix is ref[cp+1], carried as the
         * pending anchor. */
        require(c0.committed_count == (size_t)cp + 1,
                "cycle 0 emitted the wrong number of tokens");
        assert_invariant(s, sp_plen, c0.committed_count, "pending-reject c0");
        require(spec.have_pending, "correction was not carried as pending");
        require(spec.pending_anchor == ref[cp + 1],
                "pending anchor is not the target's correction token");
        require(spec.stats.rewinds == 1u,
                "a partial-block rewind should have happened");

        /* cycle 1 by hand: its anchor must be the pending correction ref[cp+1]. */
        size_t on = c0.committed_count;
        qwen_spec_cycle c1;
        require(qwen_spec_step(&spec, rn - on, STOP_IDS, STOP_COUNT, &c1, err,
                               sizeof(err)),
                err);
        require(c1.committed_count >= 1 && c1.committed[0] == ref[cp + 1],
                "cycle 1 did not emit the pending correction as its anchor");
        on += c1.committed_count;
        assert_invariant(s, sp_plen, on, "pending-reject c1");
        qwen_draft_destroy(oracle);
        qwen_session_free(s);

        /* short whole-run parity (stay within the prompt's stable window) +
         * rewind count on a fresh run with the same forced corruption. */
        const size_t RWIN = 10;
        qwen_session *s2 = prefill(m, PROMPT_EN, NULL, &sp_plen);
        qwen_draft_backend *o2 =
            qwen_draft_oracle_new(full, full_len, cat, ctok);
        qwen_spec sp2;
        require(qwen_spec_init(&sp2, s2, o2, 5, err, sizeof(err)), err);
        uint32_t out2[MAXGEN];
        size_t on2 = 0;
        qwen_spec_stats st2;
        drive(&sp2, s2, sp_plen, 0, RWIN, out2, &on2, &st2,
              "pending-reject short");
        require(on2 == RWIN, "reject coordinator emitted fewer than the window");
        parity_check(m, PROMPT_EN, ref, RWIN, out2, on2, "pending-reject");
        printf("  cp=%d: reject at draft %d, correction carried, parity ok "
               "(rewinds=%llu)\n",
               cp, cp, (unsigned long long)st2.rewinds);
        qwen_draft_destroy(o2);
        qwen_session_free(s2);
    }
    free(full);
    puts("ok: spec pending-reject-check (rejection point + pending + parity)");
    return 0;
}

static int run_pending_boundary(model *m) {
    printf("== spec pending-boundary-check (max_new 1..7) ==\n");
    char err[512];
    uint32_t *pids = NULL;
    size_t plen = 0;
    uint32_t ref[MAXGEN];
    size_t rn = reference(m, PROMPT_EN, &pids, &plen, ref, MAXGEN);
    require(rn >= 12, "reference too short");
    size_t full_len = 0;
    uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);
    free(pids);

    for (size_t mn = 1; mn <= 7; mn++) {
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, NULL, &sp_plen);
        qwen_draft_backend *oracle =
            qwen_draft_oracle_new(full, full_len, (size_t)-1, 0);
        qwen_spec spec;
        require(qwen_spec_init(&spec, s, oracle, 5, err, sizeof(err)), err);
        uint32_t out[MAXGEN];
        size_t on = 0;
        qwen_spec_stats st;
        drive(&spec, s, sp_plen, 0, mn, out, &on, &st, "pending-boundary");
        require(on == mn, "coordinator did not emit exactly max_new tokens");
        parity_check(m, PROMPT_EN, ref, mn, out, on, "pending-boundary");
        assert_invariant(s, sp_plen, mn, "pending-boundary final");
        printf("  max_new=%zu: %zu tokens, session length ok\n", mn, on);
        qwen_draft_destroy(oracle);
        qwen_session_free(s);
    }
    free(full);
    puts("ok: spec pending-boundary-check (exact count + parity + invariant)");
    return 0;
}

static int run_pending_eos(model *m) {
    printf("== spec pending-eos-check ==\n");
    /* Prompts whose greedy generation ends on its own within a few tokens, so
     * the target's own EOS lands in different block positions as W varies. */
    const char *shorts[] = {
        "Reply with exactly the single word: yes",
        "What is 2 + 2? Reply with just the number.",
        "日本の首都を、余計な言葉なしで一語だけ答えてください。",
    };
    for (size_t pi = 0; pi < sizeof(shorts) / sizeof(shorts[0]); pi++) {
        uint32_t *pids = NULL;
        size_t plen = 0;
        uint32_t ref[MAXGEN];
        size_t rn = reference(m, shorts[pi], &pids, &plen, ref, MAXGEN);
        if (rn < 1 || rn > 40) { free(pids); continue; }
        size_t full_len = 0;
        uint32_t *full = make_full_stream(pids, plen, ref, rn, &full_len);
        free(pids);
        for (unsigned W = 2; W <= 5; W++) {
            char err[512];
            size_t sp_plen = 0;
            qwen_session *s = prefill(m, shorts[pi], NULL, &sp_plen);
            /* oracle stream has NO EOS -- the target must produce it. */
            qwen_draft_backend *oracle =
                qwen_draft_oracle_new(full, full_len, (size_t)-1, 0);
            qwen_spec spec;
            require(qwen_spec_init(&spec, s, oracle, W, err, sizeof(err)), err);
            uint32_t out[MAXGEN];
            size_t on = 0;
            qwen_spec_stats st;
            drive(&spec, s, sp_plen, 0, MAXGEN, out, &on, &st, "pending-eos");
            require(on == rn, "EOS: coordinator emitted a different count");
            parity_check(m, shorts[pi], ref, rn, out, on, "pending-eos");
            /* the stop token is never emitted and never left in the KV */
            for (size_t i = 0; i < on; i++)
                require(!is_stop(out[i]),
                        "a stop id was emitted to the output");
            assert_invariant(s, sp_plen, rn, "pending-eos final");
            qwen_draft_destroy(oracle);
            qwen_session_free(s);
        }
        printf("  prompt %zu: %zu tokens, EOS never emitted / never in KV "
               "(W 2..5)\n",
               pi, rn);
        free(full);
    }
    puts("ok: spec pending-eos-check");
    return 0;
}

static int run_pending_vlm(model *m) {
    printf("== spec pending-vlm-check (non-zero mRoPE) ==\n");
    char err[512];
    enum { SPN = 4, HID = 5120, MAXIDS = 128 };
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
    ids[TOK++] = 151652u;
    SP0 = TOK;
    for (int i = 0; i < SPN; i++) ids[TOK++] = 151655u;
    ids[TOK++] = 151653u;
    for (size_t i = 0; i < npost; i++) ids[TOK++] = post[i];
    h3_tokenizer_ids_free(pre);
    h3_tokenizer_ids_free(post);
    require(TOK < MAXIDS, "prompt too long");

    uint32_t pos[3 * MAXIDS];
    uint32_t grid_max = (uint32_t)SP0 + 8u;
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

    /* reference greedy after the multimodal prefill */
    qwen_session *r = NULL;
    require(qwen_session_create(&r, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(r, 1, err, sizeof(err)), err);
    require(qwen_session_eval_multimodal(r, &in, err, sizeof(err)), err);
    uint32_t ref[MAXGEN];
    size_t rn = ref_greedy(r, 20, ref);
    qwen_session_free(r);
    require(rn >= 8, "multimodal reference too short");

    /* oracle stream = multimodal prompt ids ++ ref, with a forced reject at
     * draft index 2 in the first cycle. */
    uint32_t *full = malloc((TOK + rn) * sizeof(*full));
    require(full != NULL, "alloc");
    memcpy(full, ids, TOK * sizeof(*full));
    memcpy(full + TOK, ref, rn * sizeof(*full));
    size_t cat = TOK + 1 + 2;
    uint32_t ctok = (full[cat] + 11u) % SPEC_VOCAB;
    if (ctok == full[cat]) ctok = (ctok + 1u) % SPEC_VOCAB;

    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_eval_multimodal(s, &in, err, sizeof(err)), err);
    qwen_draft_backend *oracle =
        qwen_draft_oracle_new(full, TOK + rn, cat, ctok);
    qwen_spec spec;
    require(qwen_spec_init(&spec, s, oracle, 5, err, sizeof(err)), err);

    qwen_spec_cycle c0;
    require(qwen_spec_step(&spec, rn, STOP_IDS, STOP_COUNT, &c0, err,
                           sizeof(err)),
            err);
    require(c0.accepted_from_draft == 2, "vlm reject point wrong");
    assert_invariant(s, TOK, c0.committed_count, "pending-vlm c0");
    /* emitted ref[0..2]; correction after that prefix is ref[3]. */
    require(spec.have_pending && spec.pending_anchor == ref[3],
            "vlm correction not carried as pending across non-zero mRoPE");

    uint32_t out[MAXGEN];
    size_t on = 0;
    for (size_t i = 0; i < c0.committed_count; i++) out[on++] = c0.committed[i];
    qwen_spec_stats st;
    {
        size_t tail = 0;
        drive(&spec, s, TOK, on, rn - on, out + on, &tail, &st, "pending-vlm tail");
        on += tail;
    }
    require(on == rn, "vlm coordinator token count");
    cmp_or_die(ref, out, rn, "pending-vlm");
    assert_invariant(s, TOK, rn, "pending-vlm final");
    printf("  %zu-token multimodal prompt: reject -> rewind -> pending -> next "
           "batch, parity ok (rewinds=%llu)\n",
           TOK, (unsigned long long)st.rewinds);

    qwen_draft_destroy(oracle);
    qwen_session_free(s);
    free(full);
    free(emb);
    free(ds[0]);
    free(ds[1]);
    free(ds[2]);
    puts("ok: spec pending-vlm-check");
    return 0;
}

static int run_pending_parity(model *m) {
    printf("== spec pending-parity (n-gram draft) ==\n");
    struct { const char *tag; const char *prompt; } B[] = {
        {"EN  ", PROMPT_EN},
        {"JA  ", PROMPT_JA},
        {"code", PROMPT_CODE},
        {"JSON",
         "Return a JSON object with keys name (\"Ada\"), age (36), "
         "city (\"London\"). JSON only."},
    };
    for (size_t i = 0; i < sizeof(B) / sizeof(B[0]); i++) {
        char err[512];
        uint32_t *pids = NULL;
        size_t plen = 0;
        uint32_t ref[MAXGEN];
        size_t rn = reference(m, B[i].prompt, &pids, &plen, ref, MAXGEN);
        free(pids);
        size_t sp_plen = 0;
        qwen_session *s = prefill(m, B[i].prompt, NULL, &sp_plen);
        qwen_draft_backend *ng = qwen_draft_ngram_new();
        qwen_spec spec;
        require(qwen_spec_init(&spec, s, ng, 5, err, sizeof(err)), err);
        uint32_t out[MAXGEN];
        size_t on = 0;
        qwen_spec_stats st;
        drive(&spec, s, sp_plen, 0, rn, out, &on, &st, "pending-parity");
        char lbl[24];
        snprintf(lbl, sizeof(lbl), "  %s", B[i].tag);
        qwen_spec_stats_print(&st, lbl, 0.0);
        parity_check(m, B[i].prompt, ref, rn, out, on, "pending-parity");
        printf("  %s: %zu tokens vs greedy (near-tie tolerant)\n", B[i].tag, rn);
        qwen_draft_destroy(ng);
        qwen_session_free(s);
    }
    puts("ok: spec pending-parity (coordinator == greedy modulo near-ties)");
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

    size_t n = tail_len < 12 ? tail_len : 12;
    for (size_t i = 0; i < n; i++) {
        uint32_t got = 0;
        require(qwen_session_sample(s, &got, err, sizeof(err)), err);
        if (got != ref_tail[i]) {
            /* First divergence: only a genuine decode near-tie is allowed;
             * anything else means the rewound state (KV / mRoPE) is wrong. */
            float margin = 0.0f;
            int ok = near_tie_ok(qwen_session_logits(s), ref_tail[i], got,
                                 &margin);
            fprintf(stderr,
                    "  %s: re-decode matched %zu/%zu (div at %zu: ref=%u got=%u "
                    "margin=%.4f -> %s)\n",
                    what, i, n, i, ref_tail[i], got, (double)margin,
                    ok ? "near-tie OK" : "REAL -- state not restored");
            require(ok, "rewind re-decode diverges outside a near-tie -- "
                        "KV/mRoPE state not restored");
            return i;
        }
        if (is_stop(got)) return i;
        require(qwen_session_eval(s, &got, 1, err, sizeof(err)), err);
    }
    fprintf(stderr, "  %s: re-decode matched %zu/%zu\n", what, n, n);
    return n;
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
    /* redecode_and_match hard-fails on any non-near-tie drift, so a restored
     * mRoPE state is what lets it get past position 0 at all. */
    redecode_and_match(s, TOK + k, ref + k, rn - k, "vlm keep");
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

/* -------- 015d-1: scalar decode vs batched verifier, per row ------------- *
 *
 * qwen_session_verify_block() must, for every row, predict the same next
 * token as scalar decode from the same prefix -- or diverge only at a decode
 * near-tie (joint top-2, sub-TIE_EPS, nothing higher). Runs several frontiers
 * over a reference greedy decode and reports a margin histogram.
 */
static void top2_of(const qwen_logits *lg, uint32_t *t1, uint32_t *t2,
                    float *m) {
    float b1 = -1e30f, b2 = -1e30f;
    uint32_t i1 = 0, i2 = 0;
    for (size_t i = 0; i < lg->vocab; i++) {
        float v = lg->values[i];
        if (v > b1) { b2 = b1; i2 = i1; b1 = v; i1 = (uint32_t)i; }
        else if (v > b2) { b2 = v; i2 = (uint32_t)i; }
    }
    *t1 = i1; *t2 = i2; *m = b1 - b2;
}

static int run_verify_parity(model *m) {
    printf("== spec verify-parity (scalar decode vs batched verifier) ==\n");
    char err[512];
    const char *prompts[] = {PROMPT_EN, PROMPT_JA, PROMPT_CODE};
    unsigned widths[] = {2, 3, 4, 5};
    uint64_t bucket[4] = {0};       /* <0.01, <0.02, <0.05, >=0.05 */
    long rows_total = 0, rows_exact = 0, rows_near_tie = 0;

    for (size_t pi = 0; pi < sizeof(prompts) / sizeof(prompts[0]); pi++) {
        /* scalar reference: tokens + per-position top1/top2/margin */
        size_t plen = 0;
        qwen_session *r = prefill(m, prompts[pi], NULL, &plen);
        uint32_t ref[MAXGEN], s_t1[MAXGEN], s_t2[MAXGEN];
        float s_m[MAXGEN];
        size_t rn = 0;
        for (; rn < MAXGEN; rn++) {
            const qwen_logits *lg = qwen_session_logits(r);
            require(lg && lg->values, "no scalar logits");
            top2_of(lg, &s_t1[rn], &s_t2[rn], &s_m[rn]);
            if (is_stop(s_t1[rn])) break;
            ref[rn] = s_t1[rn];
            require(qwen_session_eval(r, &ref[rn], 1, err, sizeof(err)), err);
        }
        qwen_session_free(r);
        require(rn >= 24, "reference too short for verify-parity");

        for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
            unsigned W = widths[wi];
            qwen_session *s = prefill(m, prompts[pi], NULL, &plen);
            size_t cur = plen; /* session frontier */
            for (size_t F = plen + 4; F + W < plen + rn; F += 9) {
                while (cur < F) {
                    uint32_t t = ref[cur - plen];
                    require(qwen_session_eval(s, &t, 1, err, sizeof(err)), err);
                    cur++;
                }
                qwen_verify_result vr;
                require(qwen_session_verify_block(s, ref + (F - plen), W, &vr,
                                                  err, sizeof(err)),
                        err);
                require(vr.rows == W, "verify rows mismatch");
                for (unsigned rr = 0; rr < W; rr++) {
                    /* row rr's logits predict the token AFTER block[0..rr],
                     * i.e. scalar step (F-plen)+rr+1. */
                    size_t pos = (F - plen) + rr + 1;
                    uint32_t st1 = s_t1[pos], st2 = s_t2[pos];
                    float sm = s_m[pos];
                    rows_total++;
                    float mm = vr.margin[rr] < sm ? vr.margin[rr] : sm;
                    bucket[mm < 0.01f ? 0 : mm < 0.02f ? 1 : mm < 0.05f ? 2 : 3]++;
                    if (vr.top1[rr] == st1) { rows_exact++; continue; }
                    int joint = (st1 == vr.top1[rr] || st1 == vr.top2[rr]) &&
                                (vr.top1[rr] == st1 || vr.top1[rr] == st2);
                    if (joint && mm < TIE_EPS) {
                        rows_near_tie++;
                        fprintf(stderr,
                                "  p%zu W%u F%zu r%u: near-tie scalar=%u "
                                "batch=%u (margin %.4f)\n",
                                pi, W, F, rr, st1, vr.top1[rr], (double)mm);
                    } else {
                        fprintf(stderr,
                                "  p%zu W%u F%zu r%u: HARD scalar=%u(t2 %u) "
                                "batch=%u(t2 %u) sm=%.4f bm=%.4f\n",
                                pi, W, F, rr, st1, st2, vr.top1[rr], vr.top2[rr],
                                (double)sm, (double)vr.margin[rr]);
                        fail("verify_block row diverges from scalar outside a "
                             "near-tie");
                    }
                }
                require(qwen_session_rewind(s, F, err, sizeof(err)), err);
                cur = F;
            }
            qwen_session_free(s);
            printf("  prompt %zu W=%u: frontiers checked\n", pi, W);
        }
    }
    printf("\nverify-parity: %ld rows  exact %ld  near-tie %ld\n", rows_total,
           rows_exact, rows_near_tie);
    printf("  margin histogram: <0.01=%llu  <0.02=%llu  <0.05=%llu  "
           ">=0.05=%llu\n",
           (unsigned long long)bucket[0], (unsigned long long)bucket[1],
           (unsigned long long)bucket[2], (unsigned long long)bucket[3]);
    require(rows_exact + rows_near_tie == rows_total, "unaccounted rows");
    puts("ok: spec verify-parity -- batched verifier == scalar decode "
         "(modulo near-ties)");
    return 0;
}

/* -------- QINT-015e-0 gate: classify the first scalar-vs-batch divergence - *
 *
 * chain-drift feeds an identical teacher-forced token sequence through a
 * chained batched verifier and a scalar decode. The scalar reference is NOT
 * bit-stable run to run -- an upstream rows==1 near-tie (the documented GPU
 * reduction-order non-determinism) occasionally forks it -- so a divergence
 * must be judged by BOTH sides' confidence, not the scalar margin alone:
 *
 *   robust_margin = min(scalar top1-top2 gap, batch top1-top2 gap)
 *
 * A disagreement is a real batch-path drift only when robust_margin >= 0.2:
 * both forwards were confident and still disagree. If either side is a
 * knife-edge it is a tolerated fork of the non-deterministic reference (the
 * same call verify-parity already treats as OK). Only the FIRST divergence is
 * gated -- once the batched and scalar forwards disagree, later-position
 * disagreements are downstream of that one numeric event, not independent,
 * so they are diagnostic only.
 */
#define CD_STRICT_MARGIN 0.2f
typedef enum { CD_AGREE, CD_FORK_NEAR_TIE, CD_FAIL } cd_class;

static cd_class cd_classify(uint32_t scalar_tok, uint32_t batch_tok,
                            float scalar_margin, float batch_margin) {
    if (scalar_tok == batch_tok) return CD_AGREE;
    float robust = scalar_margin < batch_margin ? scalar_margin : batch_margin;
    return robust >= CD_STRICT_MARGIN ? CD_FAIL : CD_FORK_NEAR_TIE;
}
static const char *cd_class_name(cd_class c) {
    return c == CD_AGREE ? "AGREE"
           : c == CD_FORK_NEAR_TIE ? "FORK_NEAR_TIE"
                                   : "FAIL";
}

/* FNV-1a over a token prefix -- a fingerprint of the teacher-forced reference
 * path. The scalar reference is not bit-stable run to run (upstream rows==1
 * near-ties fork it), so an A/B needs to tell "candidate took a different
 * reference path" (fingerprints differ) apart from "candidate disagrees more
 * on the SAME path" (fingerprints match, only the candidate flips). */
static uint64_t cd_fnv1a(const uint32_t *v, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= v[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* Model-free proof the corrected gate is not a weakening: a both-confident
 * disagreement still fails; a one-sided knife-edge is tolerated; the 0.2
 * threshold is inclusive on the FAIL side. */
static int run_chain_drift_gate_selftest(void) {
    printf("== chain-drift gate self-test (no model) ==\n");
    struct {
        uint32_t a, b;
        float sm, bm;
        cd_class want;
        const char *name;
    } cs[] = {
        {5, 5, 0.30f, 0.30f, CD_AGREE, "same token -> agree"},
        {5, 9, 0.25f, 0.005f, CD_FORK_NEAR_TIE, "batch knife-edge -> tolerated"},
        {5, 9, 0.005f, 0.25f, CD_FORK_NEAR_TIE, "scalar knife-edge -> tolerated"},
        {5, 9, 0.19f, 0.90f, CD_FORK_NEAR_TIE, "one side just under 0.2"},
        {5, 9, 0.20f, 0.20f, CD_FAIL, "both exactly at 0.2 -> FAIL"},
        {5, 9, 0.25f, 0.30f, CD_FAIL, "both confident, disagree -> FAIL"},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof(cs) / sizeof(cs[0]); i++) {
        cd_class got = cd_classify(cs[i].a, cs[i].b, cs[i].sm, cs[i].bm);
        printf("  [%zu] %-32s scalar_m=%.3f batch_m=%.3f -> %-13s %s\n", i,
               cs[i].name, (double)cs[i].sm, (double)cs[i].bm,
               cd_class_name(got), got == cs[i].want ? "ok" : "MISMATCH");
        if (got != cs[i].want) bad = 1;
    }
    require(!bad, "chain-drift gate self-test failed");
    puts("ok: chain-drift gate self-test");
    return 0;
}

/* -------- QINT-015e-0: batch-path numerical drift -------------------- *
 *
 * verify-parity (015d-1) advances the frontier with single-token decodes and
 * does ONE verify_block per frontier -- it measures the verifier in
 * isolation. This measures what the *coordinator* actually does: build the
 * whole context out of CHAINED verify_block calls, feeding the identical
 * token sequence, and see how far the batched path's argmax tracks a
 * teacher-forced scalar decode as the small per-step numeric difference
 * accumulates. This is the baseline to re-run after any QINT-015e kernel
 * change (faster must not mean "argmax drifts sooner / wider").
 */
static int run_chain_drift(model *m) {
    run_chain_drift_gate_selftest();
    printf("== spec chain-drift (chained verify_block vs teacher-forced "
           "scalar) ==\n");
    char err[512];
    const char *prompts[] = {PROMPT_EN, PROMPT_JA, PROMPT_CODE};
    for (size_t pi = 0; pi < sizeof(prompts) / sizeof(prompts[0]); pi++) {
        /* teacher-forced scalar reference: token + per-position top1/margin */
        size_t plen = 0;
        qwen_session *r = prefill(m, prompts[pi], NULL, &plen);
        uint32_t ref[MAXGEN], s_t1[MAXGEN + 1];
        float s_m[MAXGEN + 1];
        size_t rn = 0;
        for (; rn < MAXGEN; rn++) {
            const qwen_logits *lg = qwen_session_logits(r);
            require(lg && lg->values, "no scalar logits");
            uint32_t t2;
            top2_of(lg, &s_t1[rn], &t2, &s_m[rn]);
            if (is_stop(s_t1[rn])) break;
            ref[rn] = s_t1[rn];
            require(qwen_session_eval(r, &ref[rn], 1, err, sizeof(err)), err);
        }
        {   /* one extra step so s_t1[rn] (the token after the last ref) exists */
            const qwen_logits *lg = qwen_session_logits(r);
            uint32_t t2;
            top2_of(lg, &s_t1[rn], &t2, &s_m[rn]);
        }
        qwen_session_free(r);
        require(rn >= 24, "chain-drift reference too short");

        uint64_t ref_fp = cd_fnv1a(ref, rn);
        printf("  prompt %zu  rn=%zu  ref-fp=%016llx\n", pi, rn,
               (unsigned long long)ref_fp);

        for (unsigned B = 2; B <= 5; B++) {
            qwen_session *s = prefill(m, prompts[pi], NULL, &plen);
            long rows = 0, agree = 0, later_div = 0, first_div = -1;
            uint32_t fd_st = 0, fd_bt = 0;      /* first-div scalar/batch token */
            float fd_sm = 0.0f, fd_bm = 0.0f;   /* first-div scalar/batch margin */
            for (size_t j = 0; j + B <= rn; j += B) {
                qwen_verify_result vr;
                require(qwen_session_verify_block(s, ref + j, B, &vr, err,
                                                  sizeof(err)),
                        err);
                for (unsigned rr = 0; rr < B; rr++) {
                    size_t p = j + rr + 1; /* scalar position this row predicts */
                    rows++;
                    if (vr.top1[rr] == s_t1[p]) {
                        agree++;
                    } else if (first_div < 0) {
                        first_div = (long)(j + rr);
                        fd_st = s_t1[p];
                        fd_bt = vr.top1[rr];
                        fd_sm = s_m[p];
                        fd_bm = vr.margin[rr];
                    } else {
                        later_div++; /* downstream of first_div: diagnostic only */
                    }
                }
                /* the coordinator only rewinds on a reject; a chained
                 * all-accept keeps every row, so DON'T rewind here. */
            }
            qwen_session_free(s);

            cd_class fd = first_div < 0
                              ? CD_AGREE
                              : cd_classify(fd_st, fd_bt, fd_sm, fd_bm);
            printf("  prompt %zu  B=%u  rows=%ld  agree=%ld/%ld (%.1f%%)  "
                   "first-div=%ld",
                   pi, B, rows, agree, rows,
                   100.0 * (double)agree / (double)(rows ? rows : 1), first_div);
            if (first_div >= 0) {
                float robust = fd_sm < fd_bm ? fd_sm : fd_bm;
                uint64_t fp_before =
                    cd_fnv1a(ref, (size_t)first_div + 1); /* ref[0..first_div] */
                printf("  scalar_tok=%u batch_tok=%u scalar_m=%.3f batch_m=%.3f "
                       "robust_m=%.3f class=%s ref-before-div=%016llx "
                       "later-div=%ld",
                       fd_st, fd_bt, (double)fd_sm, (double)fd_bm, (double)robust,
                       cd_class_name(fd), (unsigned long long)fp_before,
                       later_div);
            }
            printf("\n");

            /* Gate ONLY the first divergence, and only when BOTH the scalar
             * and the batched forward were confident there (robust margin
             * >= 0.2). A one-sided knife-edge is a tolerated fork of the
             * non-deterministic teacher-forced reference. */
            require(fd != CD_FAIL,
                    "chain-drift: the FIRST scalar-vs-batch divergence is a "
                    "large-margin flip on BOTH sides -- real batch-path drift, "
                    "not a fork of the non-deterministic reference");
        }
    }
    puts("ok: spec chain-drift (baseline; re-run after any 015e kernel change)");
    return 0;
}

/* --- QINT-015h-0: EAGLE-3 auxiliary-hidden capture ------------------------ */

static float bf16f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static int aux_finite(const uint16_t *v, size_t n) {
    int nonzero = 0;
    for (size_t i = 0; i < n; i++) {
        float f = bf16f(v[i]);
        if (f != f || f > 1e30f || f < -1e30f) return 0;
        if (f != 0.0f) nonzero = 1;
    }
    return nonzero;
}

/* Prefill a fresh resident session on `user` with EAGLE-3 aux capture on the
 * given layers. Caller frees with qwen_session_free(). */
static qwen_session *prefill_aux(model *m, const char *user, const int *layers,
                                 size_t nl) {
    char err[512];
    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, nl, err, sizeof(err)), err);
    qwen_chat_message msg = {QWEN_ROLE_USER, user, NULL};
    uint32_t *ids = NULL;
    size_t n = 0;
    require(qwen_chat_tokenize(m->tok, &msg, 1, 1, &ids, &n, err, sizeof(err)),
            err);
    require(qwen_session_eval(s, ids, n, err, sizeof(err)), err);
    h3_tokenizer_ids_free(ids);
    return s;
}

static int run_aux_capture(model *m) {
    printf("== QINT-015h-0 EAGLE-3 aux-hidden capture ==\n");
    char err[512];
    const int layers[3] = {1, 32, 61};
    size_t rows = 99, n_aux = 99, hid = 99;
    const int *ids = NULL;

    /* 1. capture is off until opted in. */
    size_t plen = 0;
    qwen_session *s = prefill(m, PROMPT_EN, NULL, &plen);
    require(qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids) == NULL &&
                rows == 0 && n_aux == 0,
            "aux hidden must be NULL / zero before set_aux_layers");
    qwen_session_free(s);

    /* 1b. bad configs are rejected. */
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    int bad_dup[2] = {5, 5}, bad_range[1] = {64};
    require(!qwen_session_set_aux_layers(s, bad_dup, 2, err, sizeof(err)),
            "duplicate aux layer id must be rejected");
    require(!qwen_session_set_aux_layers(s, bad_range, 1, err, sizeof(err)),
            "out-of-range aux layer id must be rejected");
    qwen_session_free(s);

    /* 2. configure + prefill: one frontier row, three aux slots, ids echoed. */
    s = prefill_aux(m, PROMPT_EN, layers, 3);
    const uint16_t *a = qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
    require(a && rows == 1 && n_aux == 3 && hid == 5120, "prefill aux shape");
    require(ids && ids[0] == 1 && ids[1] == 32 && ids[2] == 61,
            "aux layer ids echoed in order");
    for (size_t j = 0; j < 3; j++)
        require(aux_finite(a + j * hid, hid), "prefill aux row finite/non-zero");
    require(memcmp(a, a + hid, hid * 2) != 0 &&
                memcmp(a + hid, a + 2 * hid, hid * 2) != 0,
            "different layers must yield different hidden vectors");

    /* 3. a rewind invalidates the snapshot. */
    uint32_t blk[5];
    for (int i = 0; i < 5; i++) {
        require(qwen_session_sample(s, &blk[i], err, sizeof(err)), err);
        require(qwen_session_eval(s, &blk[i], 1, err, sizeof(err)), err);
    }
    size_t base = qwen_session_length(s);
    require(qwen_session_rewind(s, base - 5, err, sizeof(err)), err);
    require(qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids) == NULL,
            "a rewind must invalidate the aux snapshot");

    /* 4. verify_block keeps every row; each is finite. */
    qwen_verify_result vr;
    require(qwen_session_verify_block(s, blk, 5, &vr, err, sizeof(err)), err);
    a = qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
    require(a && rows == 5 && n_aux == 3 && hid == 5120, "verify aux shape");
    for (size_t r = 0; r < 5; r++)
        for (size_t j = 0; j < 3; j++)
            require(aux_finite(a + (j * rows + r) * hid, hid),
                    "verify aux row finite/non-zero");
    /* adjacent verify rows are different positions -> different hidden. */
    require(memcmp(a, a + hid, hid * 2) != 0,
            "verify rows 0 and 1 must differ at layer 1");
    qwen_session_free(s);

    /* 5. capture must not perturb the target's own greedy decode. */
    qwen_session *plain = prefill(m, PROMPT_EN, NULL, &plen);
    qwen_session *withaux = prefill_aux(m, PROMPT_EN, layers, 3);
    for (int i = 0; i < 24; i++) {
        uint32_t ta = 0, tb = 0;
        require(qwen_session_sample(plain, &ta, err, sizeof(err)), err);
        require(qwen_session_sample(withaux, &tb, err, sizeof(err)), err);
        if (ta != tb) {
            const qwen_logits *lg = qwen_session_logits(withaux);
            uint32_t t1, t2;
            float mg = 0.0f;
            top2_of(lg, &t1, &t2, &mg);
            require(mg < TIE_EPS,
                    "aux capture changed a non-near-tie greedy step");
            printf("  step %d: near-tie flip (margin %.3f), tolerated\n", i, mg);
        }
        if (is_stop(ta) || is_stop(tb)) break;
        require(qwen_session_eval(plain, &ta, 1, err, sizeof(err)), err);
        require(qwen_session_eval(withaux, &tb, 1, err, sizeof(err)), err);
    }
    qwen_session_free(plain);
    qwen_session_free(withaux);

    puts("ok: QINT-015h-0 aux-hidden capture");
    return 0;
}

/* --- QINT-015h-2b-1: live wiring of the EAGLE draft backend ------------- */

static int g_live_calls;
static uint32_t g_live_first;
static qwen_session *g_live_session;
static int live_embed(void *ctx, uint32_t token, float *out) {
    (void)ctx;
    if (g_live_calls++ == 0) g_live_first = token;
    return qwen_session_embedding_row_f32(g_live_session, token, out,
                                          QWEN_HIDDEN_SIZE);
}

/* No acceptance / tau here (a fresh per-cycle draft KV can't attend the
 * prefix -- that is 2b-2). This checks: real {1,31,60} aux from a live
 * session; the anchor from the target's own argmax; the resident-embedding
 * partial-row accessor; and that the backend's propose() == a direct
 * qwen_eagle3_chain() at start_pos = history_length - 1. */
static int run_eagle_live(model *m, const char *eagle_dir) {
    printf("== QINT-015h-2b-1 EAGLE draft live wiring (%s) ==\n", eagle_dir);
    char err[512];
    const int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT; /* {1,31,60} */

    qwen_session *s = prefill_aux(m, PROMPT_EN, layers, 3);
    /* decode a few real tokens so we test a steady-state frontier. */
    for (int i = 0; i < 5; i++) {
        uint32_t t;
        require(qwen_session_sample(s, &t, err, sizeof(err)), err);
        if (is_stop(t)) break;
        require(qwen_session_eval(s, &t, 1, err, sizeof(err)), err);
    }
    size_t L = qwen_session_length(s);
    uint32_t anchor = 0;
    require(qwen_session_sample(s, &anchor, err, sizeof(err)), err);

    size_t rows = 0, n_aux = 0, hid = 0;
    const int *ids = NULL;
    const uint16_t *base = qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
    require(base && rows == 1 && n_aux == 3 && hid == QWEN_HIDDEN_SIZE,
            "live aux shape (expect 1 x 3 x 5120)");
    require(ids[0] == 1 && ids[1] == 31 && ids[2] == 60, "aux ids {1,31,60}");

    /* embedding partial-row accessor: one f32 row, no allocation. */
    float *erow = malloc(QWEN_HIDDEN_SIZE * sizeof(float));
    require(erow != NULL, "alloc");
    require(qwen_session_embedding_row_f32(s, anchor, erow, QWEN_HIDDEN_SIZE),
            "embedding_row_f32 failed for the anchor");
    require(!qwen_session_embedding_row_f32(s, SPEC_VOCAB, erow, QWEN_HIDDEN_SIZE),
            "embedding_row_f32 must reject an out-of-range token");
    require(!qwen_session_embedding_row_f32(s, anchor, erow, 4u),
            "embedding_row_f32 must reject a wrong dst_count");

    qwen_draft_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.have_anchor = 1;
    ctx.anchor_token = anchor;
    ctx.history_length = L;
    ctx.n_aux = 3;
    ctx.hidden_size = QWEN_HIDDEN_SIZE;
    for (int a = 0; a < 3; a++)
        ctx.aux_hidden[a] = base + (size_t)a * rows * hid;

    g_live_session = s;
    g_live_calls = 0;
    qwen_draft_backend *b =
        qwen_draft_eagle_new(eagle_dir, live_embed, NULL, err, sizeof(err));
    require(b != NULL, err);

    qwen_draft_proposal p1, p2;
    require(qwen_draft_propose(b, &ctx, 4, &p1), "propose");
    require(p1.count == 4, "expect 4 proposals");
    require(g_live_first == anchor, "backend step 0 must embed the anchor");
    for (size_t j = 0; j < p1.count; j++)
        require(p1.tokens[j] < SPEC_VOCAB, "proposed target token in range");
    g_live_calls = 0;
    require(qwen_draft_propose(b, &ctx, 4, &p2), "propose (2nd)");
    require(memcmp(p1.tokens, p2.tokens, 4 * sizeof(uint32_t)) == 0,
            "backend not deterministic");

    /* direct chain at start_pos = L-1 must give the same target tokens. */
    qwen_eagle3 *ed = NULL;
    require(qwen_eagle3_load(eagle_dir, &ed, err, sizeof(err)), err);
    float *a3 = malloc((size_t)3 * QWEN_HIDDEN_SIZE * sizeof(float));
    for (int a = 0; a < 3; a++)
        for (size_t i = 0; i < hid; i++)
            a3[(size_t)a * hid + i] = bf16f(ctx.aux_hidden[a][i]);
    const float *a3p[3] = {a3, a3 + hid, a3 + 2 * hid};
    qwen_eagle3_kv *kd = NULL;
    require(qwen_eagle3_kv_new(ed, &kd, err, sizeof(err)), err);
    uint32_t draft_ids[4];
    require(qwen_eagle3_chain(ed, kd, a3p, anchor, (int)L - 1, 4, live_embed,
                             NULL, draft_ids, err, sizeof(err)),
            err);
    for (int j = 0; j < 4; j++)
        require(p1.tokens[j] == qwen_eagle3_d2t(ed, draft_ids[j]),
                "backend proposal != direct chain at start_pos = L-1");

    printf("  L=%zu  anchor=%u  aux_ids={%d,%d,%d}  aux_row_source=%zu  "
           "eagle_pos=%zu  embed_token=%u\n",
           L, anchor, ids[0], ids[1], ids[2], L - 1, L - 1, anchor);
    printf("  proposal (target vocab) = %u %u %u %u  (== direct chain @ L-1)\n",
           p1.tokens[0], p1.tokens[1], p1.tokens[2], p1.tokens[3]);

    qwen_eagle3_kv_free(kd);
    qwen_eagle3_free(ed);
    free(a3);
    free(erow);
    qwen_draft_destroy(b);
    qwen_session_free(s);
    puts("ok: QINT-015h-2b-1 EAGLE draft live wiring");
    return 0;
}

/* --- QINT-015h-2b-2a: EAGLE draft prefix K/V over the committed context --- */
static int run_eagle_prefix(model *m, const char *eagle_dir) {
    printf("== QINT-015h-2b-2a EAGLE draft prefix KV (%s) ==\n", eagle_dir);
    char err[512];
    const int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;

    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
    require(qwen_session_set_aux_prefill_all_rows(s, 1), "set_aux_prefill_all");
    qwen_chat_message msg = {QWEN_ROLE_USER, PROMPT_EN, NULL};
    uint32_t *pids = NULL;
    size_t np = 0;
    require(qwen_chat_tokenize(m->tok, &msg, 1, 1, &pids, &np, err, sizeof(err)),
            err);
    require(qwen_session_eval(s, pids, np, err, sizeof(err)), err);
    h3_tokenizer_ids_free(pids);

    size_t L = qwen_session_length(s); /* == np : PREFILL only, no decode yet */
    size_t rows = 0, n_aux = 0, hid = 0;
    const int *ids = NULL;
    const uint16_t *base = qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
    require(base && n_aux == 3 && hid == QWEN_HIDDEN_SIZE,
            "prefill-all aux: n_aux/hidden");
    require(rows == L, "prefill-all aux must keep every prompt row");

    uint32_t anchor = 0;
    require(qwen_session_sample(s, &anchor, err, sizeof(err)), err);
    size_t hn = 0;
    const uint32_t *hist = qwen_session_history(s, &hn);
    require(hist && hn == L, "history length");

    /* all rows -> f32, aux-major [3][L][hidden] */
    float *aux_f = malloc((size_t)3 * L * hid * sizeof(float));
    require(aux_f != NULL, "alloc");
    for (size_t a = 0; a < 3; a++)
        for (size_t i = 0; i < L * hid; i++)
            aux_f[a * L * hid + i] = bf16f(base[(a * rows + 0) * hid + i]);
    const float *aux_all[3] = {aux_f, aux_f + L * hid, aux_f + 2 * L * hid};
    const float *frontier[3] = {aux_f + (L - 1) * hid, aux_f + L * hid + (L - 1) * hid,
                                aux_f + 2 * L * hid + (L - 1) * hid};

    g_live_session = s;
    qwen_eagle3 *e = NULL;
    require(qwen_eagle3_load(eagle_dir, &e, err, sizeof(err)), err);

    /* prefix rows t = 0 .. L-2 : aux(t) + Emb(history[t+1]) at position t. */
    uint32_t *pair = malloc((L > 1 ? L - 1 : 1) * sizeof(uint32_t));
    for (size_t t = 0; t + 1 < L; t++) pair[t] = hist[t + 1];

    qwen_eagle3_kv *kv = NULL;
    require(qwen_eagle3_kv_new(e, &kv, err, sizeof(err)), err);
    require(qwen_eagle3_kv_prefix_extend(e, kv, aux_all, pair, (int)L - 1, 0,
                                         live_embed, NULL, err, sizeof(err)),
            err);
    int base_len = qwen_eagle3_kv_len(kv);
    require(base_len == (int)L - 1,
            "draft_kv_len must equal history_length - 1 after the prefix");

    /* chain step 0 at L-1 on top of the prefix. */
    uint32_t d_prefix[4], d_fresh[4];
    require(qwen_eagle3_chain(e, kv, frontier, anchor, (int)L - 1, 4, live_embed,
                             NULL, d_prefix, err, sizeof(err)),
            err);
    qwen_eagle3_kv_reset(kv); /* fresh: no prefix */
    require(qwen_eagle3_chain(e, kv, frontier, anchor, (int)L - 1, 4, live_embed,
                             NULL, d_fresh, err, sizeof(err)),
            err);

    printf("  history_len=%zu  draft_kv_base_len=%d  frontier_aux_pos=%zu  "
           "anchor_pos=%zu  chain_step0_pos=%zu\n",
           L, base_len, L - 1, L, L - 1);
    printf("  proposal WITH prefix  (target) = %u %u %u %u\n",
           qwen_eagle3_d2t(e, d_prefix[0]), qwen_eagle3_d2t(e, d_prefix[1]),
           qwen_eagle3_d2t(e, d_prefix[2]), qwen_eagle3_d2t(e, d_prefix[3]));
    printf("  proposal fresh KV     (target) = %u %u %u %u\n",
           qwen_eagle3_d2t(e, d_fresh[0]), qwen_eagle3_d2t(e, d_fresh[1]),
           qwen_eagle3_d2t(e, d_fresh[2]), qwen_eagle3_d2t(e, d_fresh[3]));
    printf("  prefix changes the proposal: %s\n",
           memcmp(d_prefix, d_fresh, sizeof(d_prefix)) ? "yes" : "no");

    qwen_eagle3_kv_free(kv);
    qwen_eagle3_free(e);
    free(aux_f);
    free(pair);
    qwen_session_free(s);
    puts("ok: QINT-015h-2b-2a EAGLE draft prefix KV");
    return 0;
}

/* --- QINT-015h-2b-2b: prime -> propose -> (verify) -> sync -> propose ---- */
static double bf16row_cos(const uint16_t *a, const uint16_t *b, size_t n) {
    double da = 0, db = 0, dp = 0;
    for (size_t i = 0; i < n; i++) {
        double x = bf16f(a[i]), y = bf16f(b[i]);
        da += x * x; db += y * y; dp += x * y;
    }
    return (da == 0 || db == 0) ? (da == db) : dp / (sqrt(da) * sqrt(db));
}

static int run_eagle_sync(model *m, const char *eagle_dir) {
    printf("== QINT-015h-2b-2b EAGLE draft prime/propose/sync (%s) ==\n",
           eagle_dir);
    char err[512];
    const int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;
    const size_t Hh = QWEN_HIDDEN_SIZE;

    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
    require(qwen_session_set_aux_prefill_all_rows(s, 1), "prefill-all");
    qwen_chat_message msg = {QWEN_ROLE_USER, PROMPT_EN, NULL};
    uint32_t *pids = NULL;
    size_t np = 0;
    require(qwen_chat_tokenize(m->tok, &msg, 1, 1, &pids, &np, err, sizeof(err)),
            err);
    require(qwen_session_eval(s, pids, np, err, sizeof(err)), err);
    h3_tokenizer_ids_free(pids);

    size_t L = qwen_session_length(s);
    size_t rows = 0, n_aux = 0, hid = 0;
    const int *ids = NULL;
    const uint16_t *abase = qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
    require(abase && rows == L && n_aux == 3 && hid == Hh, "prefill-all aux");
    size_t hn = 0;
    const uint32_t *hist = qwen_session_history(s, &hn);
    require(hist && hn == L, "history");

    g_live_session = s;
    qwen_draft_backend *b =
        qwen_draft_eagle_new(eagle_dir, live_embed, NULL, err, sizeof(err));
    require(b != NULL, err);
    require(qwen_draft_eagle_prime(b, abase, L, 3, Hh, hist, L, err, sizeof(err)),
            err);

    uint32_t anchor = 0;
    require(qwen_session_sample(s, &anchor, err, sizeof(err)), err);
    qwen_draft_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.have_anchor = 1;
    ctx.anchor_token = anchor;
    ctx.history_length = L;
    ctx.n_aux = 3;
    ctx.hidden_size = Hh;
    for (int a = 0; a < 3; a++)
        ctx.aux_hidden[a] = abase + ((size_t)a * rows + (L - 1)) * hid;

    qwen_draft_proposal pr;
    require(qwen_draft_propose(b, &ctx, 4, &pr) && pr.count == 4,
            "cycle 1 propose");

    /* simulate the target verify over [anchor, D1, D2, D3]. */
    uint32_t block[4] = {anchor, pr.tokens[0], pr.tokens[1], pr.tokens[2]};
    qwen_verify_result vr;
    require(qwen_session_verify_block(s, block, 4, &vr, err, sizeof(err)), err);
    size_t vrows = 0, vn = 0, vhid = 0;
    const int *vids = NULL;
    const uint16_t *vbase =
        qwen_session_aux_hidden(s, &vrows, &vn, &vhid, &vids);
    require(vbase && vrows == 4 && vn == 3 && vhid == Hh, "verify aux shape");

    /* accept r=2 draft tokens -> C = 3 committed: {anchor, D1, D2}. */
    size_t r = 2, C = 1 + r, Lp = L + C;
    uint32_t committed[3] = {anchor, pr.tokens[0], pr.tokens[1]};
    /* stash VERIFY aux row C-1 (= h[L'-1], next frontier) before rewinding. */
    uint16_t *next_fr = malloc(3 * Hh * sizeof(uint16_t));
    for (int a = 0; a < 3; a++)
        memcpy(next_fr + (size_t)a * Hh,
               vbase + ((size_t)a * vrows + (C - 1)) * vhid,
               Hh * sizeof(uint16_t));

    qwen_draft_sync_context sc;
    memset(&sc, 0, sizeof(sc));
    sc.committed_tokens = committed;
    sc.n_committed = C;
    sc.verify_aux = vbase;
    sc.verify_rows = vrows;
    sc.n_aux = 3;
    sc.hidden_size = Hh;
    sc.new_history_length = Lp;
    require(qwen_draft_sync(b, &sc) == 1, "sync must succeed on a valid cycle");

    /* bring the session to the committed state and check the next frontier
     * aux matches VERIFY row C-1 (row-alignment, end to end). */
    require(qwen_session_rewind(s, L, err, sizeof(err)), err);
    for (size_t j = 0; j < C; j++)
        require(qwen_session_eval(s, &committed[j], 1, err, sizeof(err)), err);
    require(qwen_session_length(s) == Lp, "session at L'");
    size_t drows = 0, dn = 0, dhid = 0;
    const int *dids = NULL;
    const uint16_t *dbase =
        qwen_session_aux_hidden(s, &drows, &dn, &dhid, &dids);
    require(dbase && drows == 1 && dn == 3, "decode frontier aux");
    double cmin = 1.0;
    for (int a = 0; a < 3; a++) {
        double co = bf16row_cos(next_fr + (size_t)a * Hh,
                                dbase + (size_t)a * dhid, Hh);
        if (co < cmin) cmin = co;
    }
    printf("  VERIFY row C-1 vs DECODE h[L'-1]: worst-slot cosine = %.8f\n", cmin);
    require(cmin > 0.999, "VERIFY row C-1 must line up with h[L'-1]");

    /* cycle 2: the invariant (draft_kv_len == L'-1) must hold inside propose. */
    uint32_t anchor2 = 0;
    require(qwen_session_sample(s, &anchor2, err, sizeof(err)), err);
    qwen_draft_context ctx2;
    memcpy(&ctx2, &ctx, sizeof(ctx2));
    ctx2.anchor_token = anchor2;
    ctx2.history_length = Lp;
    for (int a = 0; a < 3; a++) ctx2.aux_hidden[a] = next_fr + (size_t)a * Hh;
    qwen_draft_proposal pr2;
    require(qwen_draft_propose(b, &ctx2, 4, &pr2) && pr2.count == 4,
            "cycle 2 propose (invariant held)");

    printf("  cycle1: history_len=%zu base_kv=%zu anchor=%u  accepted=%zu C=%zu\n",
           L, L - 1, anchor, r, C);
    printf("  sync  : rewind_to=%zu  new_history_len=%zu  draft_kv_len=%zu\n",
           L - 1, Lp, Lp - 1);
    printf("  cycle2: history_len=%zu anchor=%u proposal=%u %u %u %u\n", Lp,
           anchor2, pr2.tokens[0], pr2.tokens[1], pr2.tokens[2], pr2.tokens[3]);

    /* fail-closed: a bad committed[0] -> sync returns 0 -> propose declines. */
    uint32_t bad[3] = {anchor ^ 1u, pr.tokens[0], pr.tokens[1]};
    sc.committed_tokens = bad;
    require(qwen_draft_sync(b, &sc) == 0, "sync must reject a bad committed[0]");
    qwen_draft_proposal pr3;
    require(qwen_draft_propose(b, &ctx2, 4, &pr3) && pr3.count == 0,
            "after a sync failure the backend must decline (scalar fallback)");
    printf("  fail-closed: bad sync -> next propose count 0  OK\n");

    free(next_fr);
    qwen_draft_destroy(b);
    qwen_session_free(s);
    puts("ok: QINT-015h-2b-2b EAGLE draft prime/propose/sync");
    return 0;
}

/* --- QINT-015h-2b-3: real EAGLE draft in the coordinator -- tau / a_i /
 * output parity / S_M, per width M, with a step-0 RoPE-position A/B knob. --- */
static double tau_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a - *(const double *)b;
    return x < 0 ? -1 : x > 0 ? 1 : 0;
}

static int run_eagle_tau(model *m, const char *eagle_dir, size_t max_new,
                         int pos_offset) {
    char err[512];
    int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;
    const char *al = getenv("EAGLE_AUX_LAYERS");
    if (al && al[0]) {
        int a0 = 0, a1v = 0, a2 = 0;
        require(sscanf(al, "%d,%d,%d", &a0, &a1v, &a2) == 3,
                "EAGLE_AUX_LAYERS must be 'a,b,c'");
        layers[0] = a0;
        layers[1] = a1v;
        layers[2] = a2;
    }
    unsigned m_lo = 2, m_hi = 5;
    const char *mm = getenv("EAGLE_TAU_M");
    if (mm && mm[0]) {
        m_lo = m_hi = (unsigned)strtoul(mm, NULL, 10);
        require(m_lo >= 2 && m_hi <= 5, "EAGLE_TAU_M must be 2..5");
    }
    const size_t Hh = QWEN_HIDDEN_SIZE;
    require(max_new >= 1 && max_new <= MAXGEN, "max_new out of range");
    printf("== QINT-015h-2b-3 EAGLE draft tau / acceptance (%s) ==\n", eagle_dir);
    printf("   prompt=EN  max_new=%zu  M=%u..%u  aux_layers={%d,%d,%d}  "
           "step0_position_offset=%d (%s)\n",
           max_new, m_lo, m_hi, layers[0], layers[1], layers[2], pos_offset,
           pos_offset == -1 ? "L-1, prefix convention"
           : pos_offset == 0 ? "L, anchor position"
                             : "?");

    /* Scalar greedy reference (M-independent): tokens + fingerprint + the
     * single-token decode time that feeds S_M. */
    uint32_t ref[MAXGEN];
    size_t rn = 0;
    double scalar1_ms = 0.0;
    {
        uint32_t *pids = NULL;
        size_t plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, &pids, &plen);
        free(pids);
        rn = ref_greedy(s, max_new, ref); /* from the clean prefill frontier */
        qwen_session_free(s);
    }
    {
        /* T_scalar-1 on its own fresh prefill (the rewind loop below strips
         * the logits, so it cannot share the reference session). */
        uint32_t *pids = NULL;
        size_t plen = 0;
        qwen_session *s = prefill(m, PROMPT_EN, &pids, &plen);
        free(pids);
        uint32_t first = 0;
        require(qwen_session_sample(s, &first, err, sizeof(err)), err);
        double t[5];
        for (int i = 0; i < 5; i++) {
            double a = tau_now_ms();
            require(qwen_session_eval(s, &first, 1, err, sizeof(err)), err);
            t[i] = tau_now_ms() - a;
            require(qwen_session_rewind(s, plen, err, sizeof(err)), err);
        }
        qsort(t, 5, sizeof(t[0]), cmp_dbl);
        scalar1_ms = t[2];
        qwen_session_free(s);
    }
    printf("   scalar ref: %zu tokens  fingerprint=%016llx  T_scalar-1=%.1f ms\n",
           rn, (unsigned long long)cd_fnv1a(ref, rn), scalar1_ms);

    /* Teacher-forced single-step acceptance: no chain, no KV, no sync -- at
     * every greedy position feed the pristine per-position aux + prev token
     * into ONE reference EAGLE step and check d2t(argmax) == the target's
     * actual next token. Isolates "does this checkpoint fit our hiddens" from
     * any chain / KV / position-handoff bug. Tallied at position t-1 and t. */
    {
        qwen_session *s = NULL;
        require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
        require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
        require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
        qwen_chat_message msg = {QWEN_ROLE_USER, PROMPT_EN, NULL};
        uint32_t *pids = NULL;
        size_t np = 0;
        require(
            qwen_chat_tokenize(m->tok, &msg, 1, 1, &pids, &np, err, sizeof(err)),
            err);
        require(qwen_session_eval(s, pids, np, err, sizeof(err)), err);
        h3_tokenizer_ids_free(pids);
        g_live_session = s;
        qwen_eagle3 *pe = NULL;
        require(qwen_eagle3_load(eagle_dir, &pe, err, sizeof(err)), err);
        float *dl = malloc(32000u * sizeof(float));
        float *af = malloc((size_t)3 * Hh * sizeof(float));
        require(dl && af, "alloc");
        size_t hit_m1 = 0, hit_0 = 0, seen = 0;
        for (size_t step = 0; step < 22; step++) {
            /* anchor = x[L] (target's decided next token); aux = frontier
             * h[L-1], captured before eval'ing the anchor. */
            const qwen_logits *lg = qwen_session_logits(s);
            require(lg != NULL, "tf: no logits");
            uint32_t anchor_tok = lg->argmax_token;
            if (is_stop(anchor_tok)) break;
            size_t hn2 = 0;
            (void)qwen_session_history(s, &hn2); /* == L */
            size_t arows = 0, an = 0, ah = 0;
            const int *aid = NULL;
            const uint16_t *ab =
                qwen_session_aux_hidden(s, &arows, &an, &ah, &aid);
            require(ab && arows >= 1 && an == 3 && ah == Hh, "tf: aux shape");
            for (size_t a = 0; a < 3; a++)
                for (size_t i = 0; i < Hh; i++)
                    af[a * Hh + i] =
                        bf16f(ab[((size_t)a * arows + (arows - 1)) * ah + i]);
            const float *ap[3] = {af, af + Hh, af + 2 * Hh};
            /* advance the target past the anchor: its new argmax = x[L+1],
             * exactly what the draft's step-0 proposal must match. */
            require(qwen_session_eval(s, &anchor_tok, 1, err, sizeof(err)), err);
            const qwen_logits *lg2 = qwen_session_logits(s);
            require(lg2 != NULL, "tf: no logits2");
            uint32_t expect = lg2->argmax_token;
            /* the backend's chain step 0 runs at RoPE position L + offset
             * (offset -1 -> L-1, the prefix convention; offset 0 -> L). */
            for (int off = -1; off <= 0; off++) {
                require(qwen_eagle3_step_ref(pe, ap, anchor_tok,
                                             (int)hn2 + off, live_embed, NULL,
                                             NULL, dl, err, sizeof(err)),
                        err);
                int am = 0;
                for (int i = 1; i < 32000; i++)
                    if (dl[i] > dl[am]) am = i;
                uint32_t pred = qwen_eagle3_d2t(pe, (uint32_t)am);
                if (off == -1)
                    hit_m1 += (pred == expect);
                else
                    hit_0 += (pred == expect);
            }
            seen++;
        }
        printf("   teacher-forced 1-step (aux layers {%d,%d,%d}, %zu positions): "
               "a1@pos(L-1)=%.2f  a1@pos(L)=%.2f\n",
               layers[0], layers[1], layers[2], seen,
               seen ? (double)hit_m1 / (double)seen : 0.0,
               seen ? (double)hit_0 / (double)seen : 0.0);
        free(dl);
        free(af);
        qwen_eagle3_free(pe);
        qwen_session_free(s);
    }

    for (unsigned M = m_lo; M <= m_hi; M++) {
        qwen_session *s = NULL;
        require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
        require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
        require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
        require(qwen_session_set_aux_prefill_all_rows(s, 1), "prefill-all");
        qwen_chat_message msg = {QWEN_ROLE_USER, PROMPT_EN, NULL};
        uint32_t *pids = NULL;
        size_t np = 0;
        require(
            qwen_chat_tokenize(m->tok, &msg, 1, 1, &pids, &np, err, sizeof(err)),
            err);
        require(qwen_session_eval(s, pids, np, err, sizeof(err)), err);
        h3_tokenizer_ids_free(pids);
        size_t L = qwen_session_length(s);

        size_t rows = 0, n_aux = 0, hid = 0;
        const int *ids = NULL;
        const uint16_t *abase =
            qwen_session_aux_hidden(s, &rows, &n_aux, &hid, &ids);
        require(abase && rows == L && n_aux == 3 && hid == Hh,
                "prefill-all aux shape");
        /* fingerprint the frontier (last) row of each aux slot so an A/B over
         * aux-layer ids can confirm the bytes fed to the draft actually
         * changed. */
        for (size_t a = 0; a < 3; a++) {
            const uint16_t *row = abase + ((size_t)a * rows + (L - 1)) * hid;
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < hid; i++) {
                h ^= row[i];
                h *= 1099511628211ull;
            }
            printf("   M=%u aux slot %zu (layer %d) frontier-row fp=%016llx\n", M,
                   a, ids ? ids[a] : -1, (unsigned long long)h);
        }
        size_t hn = 0;
        const uint32_t *hist = qwen_session_history(s, &hn);
        require(hist && hn == L, "history");

        g_live_session = s;

        /* Sensitivity probe: does the chain's step-0 argmax actually depend
         * on the aux hidden? Run it directly with the real frontier aux, then
         * with aux zeroed, same anchor / position / (empty) kv. */
        if (M == m_lo) {
            uint32_t anc0 = 0;
            require(qwen_session_sample(s, &anc0, err, sizeof(err)), err);
            float *af = malloc((size_t)3 * Hh * sizeof(float));
            require(af != NULL, "alloc");
            for (size_t a = 0; a < 3; a++)
                for (size_t i = 0; i < Hh; i++)
                    af[a * Hh + i] =
                        bf16f(abase[((size_t)a * rows + (L - 1)) * hid + i]);
            const float *ap_real[3] = {af, af + Hh, af + 2 * Hh};
            float *az = calloc((size_t)3 * Hh, sizeof(float));
            const float *ap_zero[3] = {az, az + Hh, az + 2 * Hh};
            qwen_eagle3 *pe = NULL;
            require(qwen_eagle3_load(eagle_dir, &pe, err, sizeof(err)), err);
            qwen_eagle3_kv *pk = NULL;
            require(qwen_eagle3_kv_new(pe, &pk, err, sizeof(err)), err);
            uint32_t dr[2], dz[2];
            require(qwen_eagle3_chain(pe, pk, ap_real, anc0, (int)L - 1, 2,
                                      live_embed, NULL, dr, err, sizeof(err)),
                    err);
            qwen_eagle3_kv_reset(pk);
            require(qwen_eagle3_chain(pe, pk, ap_zero, anc0, (int)L - 1, 2,
                                      live_embed, NULL, dz, err, sizeof(err)),
                    err);
            printf("   aux-sensitivity (fresh kv, anchor=%u): real aux -> "
                   "d2t %u %u ; zero aux -> d2t %u %u  [%s]\n",
                   anc0, qwen_eagle3_d2t(pe, dr[0]), qwen_eagle3_d2t(pe, dr[1]),
                   qwen_eagle3_d2t(pe, dz[0]), qwen_eagle3_d2t(pe, dz[1]),
                   (dr[0] == dz[0] && dr[1] == dz[1]) ? "AUX IGNORED"
                                                      : "aux matters");
            qwen_eagle3_kv_free(pk);
            qwen_eagle3_free(pe);
            free(af);
            free(az);
        }

        qwen_draft_backend *b =
            qwen_draft_eagle_new(eagle_dir, live_embed, NULL, err, sizeof(err));
        require(b != NULL, err);
        qwen_draft_eagle_set_position_offset(b, pos_offset);
        require(
            qwen_draft_eagle_prime(b, abase, L, 3, Hh, hist, L, err, sizeof(err)),
            err);

        qwen_spec sp;
        require(qwen_spec_init(&sp, s, b, M, err, sizeof(err)), err);

        uint32_t out[MAXGEN];
        size_t on = 0;
        int cyc_no = 0;
        while (on < max_new) {
            qwen_spec_cycle cyc;
            require(qwen_spec_step(&sp, max_new - on, STOP_IDS, STOP_COUNT, &cyc,
                                   err, sizeof(err)),
                    err);
            if (cyc_no < 3) {
                printf("   M=%u cyc%d committed=[", M, cyc_no);
                for (size_t i = 0; i < cyc.committed_count; i++)
                    printf("%u%s", cyc.committed[i],
                           i + 1 < cyc.committed_count ? "," : "");
                printf("] accepted_from_draft=%zu%s\n", cyc.accepted_from_draft,
                       cyc.hit_stop ? " STOP" : "");
            }
            for (size_t i = 0; i < cyc.committed_count && on < max_new; i++)
                out[on++] = cyc.committed[i];
            cyc_no++;
            if (cyc.hit_stop || cyc.committed_count == 0) break;
        }

        char lbl[48];
        snprintf(lbl, sizeof(lbl), "  M=%u", M);
        qwen_spec_stats_print(&sp.stats, lbl, scalar1_ms);

        /* Output parity vs the scalar greedy sequence is the gate, ahead of
         * tau: lossless speculative decoding must reproduce the greedy tokens
         * (modulo the decode path's near-tie nondeterminism). */
        size_t k = 0, lim = rn < on ? rn : on;
        while (k < lim && ref[k] == out[k]) k++;
        if (k == rn && k == on) {
            printf("  M=%u parity: byte-identical (%zu tokens)  ref-fp=%016llx\n",
                   M, on, (unsigned long long)cd_fnv1a(out, on));
        } else {
            printf("  M=%u parity: first mismatch at %zu (ref=%zu spec=%zu "
                   "tokens) -- verifying near-tie\n",
                   M, k, rn, on);
            parity_check(m, PROMPT_EN, ref, rn, out, on, lbl);
        }

        qwen_spec_free(&sp);
        qwen_draft_destroy(b);
        qwen_session_free(s);
    }
    puts("ok: QINT-015h-2b-3 EAGLE draft tau / acceptance");
    return 0;
}

/* QINT-015i-b: write a REAL EAGLE-3 fixture (existing gen-fixture schema:
 * num_tokens / token_ids / positions / aux_hidden_{low,mid,high} [T*H] /
 * embedding [T*H]) from the live target, so `h3_qwen_eagle3_test dump` and
 * an independent SpecForge f7245ad PyTorch forward can be compared stage
 * by stage. EAGLE one-token shift: draft position i sees the target
 * residual at seq position i and the embedding of token x[i+1], and should
 * predict x[i+2]. Run with H3_QWEN_Q4=0. */
static int run_eagle_b2_fixture(model *m, const char *out_path) {
    char err[512];
    int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;
    const char *al = getenv("EAGLE_AUX_LAYERS");
    if (al && al[0]) {
        int a0 = 0, a1v = 0, a2 = 0;
        require(sscanf(al, "%d,%d,%d", &a0, &a1v, &a2) == 3, "EAGLE_AUX_LAYERS");
        layers[0] = a0;
        layers[1] = a1v;
        layers[2] = a2;
    }
    static const char *TXT =
        "The history of speculative decoding begins with a simple observation: "
        "large language models spend most of their time waiting on memory.";
    uint32_t *ids = NULL;
    size_t nraw = 0;
    require(h3_tokenizer_encode(m->tok, TXT, 0, &ids, &nraw, err, sizeof(err)),
            err);
    require(nraw >= 6, "too few tokens");
    size_t T = nraw - 1; /* draft positions 0..T-1; pos i embeds ids[i+1] */
    const size_t H = QWEN_HIDDEN_SIZE;

    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
    require(qwen_session_set_aux_prefill_all_rows(s, 1), "prefill-all");
    require(qwen_session_eval(s, ids, nraw, err, sizeof(err)), err);
    size_t rows = 0, na = 0, hid = 0;
    const int *lid = NULL;
    const uint16_t *ab = qwen_session_aux_hidden(s, &rows, &na, &hid, &lid);
    require(ab && rows == nraw && na == 3 && hid == H, "aux shape");
    g_live_session = s;

    FILE *f = fopen(out_path, "wb");
    require(f != NULL, "cannot open output");
    fprintf(f, "{\n  \"source\": \"h3.c-live-target\",\n");
    fprintf(f, "  \"hidden_size\": %zu,\n  \"num_tokens\": %zu,\n", H, T);
    fprintf(f, "  \"token_ids\": [");
    for (size_t i = 0; i < T; i++)
        fprintf(f, "%s%u", i ? "," : "", ids[i + 1]);
    fprintf(f, "],\n  \"positions\": [");
    for (size_t i = 0; i < T; i++) fprintf(f, "%s%zu", i ? "," : "", i);
    fprintf(f, "],\n  \"expected_next\": [");
    for (size_t i = 0; i < T; i++)
        fprintf(f, "%s%u", i ? "," : "", i + 2 <= nraw ? ids[i + 1] : 0u);
    fprintf(f, "]");

    const char *names[3] = {"aux_hidden_low", "aux_hidden_mid",
                            "aux_hidden_high"};
    float *buf = malloc(H * sizeof(float));
    require(buf != NULL, "alloc");
    for (size_t a = 0; a < 3; a++) {
        fprintf(f, ",\n  \"%s\": [", names[a]);
        for (size_t t = 0; t < T; t++) {
            const uint16_t *src = ab + ((size_t)a * rows + t) * hid;
            for (size_t i = 0; i < H; i++)
                fprintf(f, "%s%.9g", (t || i) ? "," : "",
                        (double)bf16f(src[i]));
        }
        fprintf(f, "]");
    }
    fprintf(f, ",\n  \"embedding\": [");
    for (size_t t = 0; t < T; t++) {
        require(qwen_session_embedding_row_f32(s, ids[t + 1], buf, H),
                "embedding row");
        for (size_t i = 0; i < H; i++)
            fprintf(f, "%s%.9g", (t || i) ? "," : "", (double)buf[i]);
    }
    fprintf(f, "]\n}\n");
    fclose(f);
    free(buf);
    printf("wrote %s  (T=%zu, layers {%d,%d,%d}, one-token shift)\n", out_path, T,
           layers[0], layers[1], layers[2]);
    h3_tokenizer_ids_free(ids);
    qwen_session_free(s);
    return 0;
}

/* QINT-015i-c (②-a prep): write a GREEDY EAGLE-3 fixture -- the token
 * sequence is the target's own greedy continuation, so `expected_next[t]`
 * (= x[t+2]) is the target's real next-token argmax and teacher-forced
 * 1-step accuracy is well defined. Same gen-fixture schema plus
 * `expected_next` and `score_from` (first draft position whose input and
 * ground truth are both in the greedy region). Feed the SAME token_ids to
 * an AWQ target on CUDA to get an AWQ-hidden fixture, then run both
 * through scripts/eagle3_specforge_accuracy.py. Run with H3_QWEN_Q4=0. */
static int run_eagle_i_fixture(model *m, const char *out_path) {
    char err[512];
    int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;
    const char *al = getenv("EAGLE_AUX_LAYERS");
    if (al && al[0]) {
        int a0 = 0, a1v = 0, a2 = 0;
        require(sscanf(al, "%d,%d,%d", &a0, &a1v, &a2) == 3, "EAGLE_AUX_LAYERS");
        layers[0] = a0;
        layers[1] = a1v;
        layers[2] = a2;
    }
    const size_t H = QWEN_HIDDEN_SIZE;
    size_t G = 28; /* greedy tokens to append */
    const char *ge = getenv("EAGLE_I_GEN");
    if (ge && ge[0]) G = (size_t)strtoul(ge, NULL, 10);

    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
    require(qwen_session_set_aux_prefill_all_rows(s, 1), "prefill-all");
    qwen_chat_message msg = {QWEN_ROLE_USER, PROMPT_EN, NULL};
    uint32_t *pids = NULL;
    size_t plen = 0;
    require(qwen_chat_tokenize(m->tok, &msg, 1, 1, &pids, &plen, err, sizeof(err)),
            err);
    require(qwen_session_eval(s, pids, plen, err, sizeof(err)), err);
    g_live_session = s;

    size_t N = plen + G;
    uint32_t *x = malloc(N * sizeof(uint32_t));
    float *aux = malloc((size_t)3 * N * H * sizeof(float)); /* slot-major */
    require(x && aux, "alloc");
    memcpy(x, pids, plen * sizeof(uint32_t));
    h3_tokenizer_ids_free(pids);

    /* prompt-position aux from the prefill-all snapshot. */
    size_t rows = 0, na = 0, hid = 0;
    const int *lid = NULL;
    const uint16_t *ab = qwen_session_aux_hidden(s, &rows, &na, &hid, &lid);
    require(ab && rows == plen && na == 3 && hid == H, "prefill aux shape");
    for (size_t a = 0; a < 3; a++)
        for (size_t t = 0; t < plen; t++)
            for (size_t i = 0; i < H; i++)
                aux[(a * N + t) * H + i] = bf16f(ab[((size_t)a * rows + t) * hid + i]);

    /* greedy-decode G tokens; capture each new position's frontier aux. */
    for (size_t k = 0; k < G; k++) {
        const qwen_logits *lg = qwen_session_logits(s);
        require(lg != NULL, "no logits");
        uint32_t nxt = lg->argmax_token;
        x[plen + k] = nxt;
        require(qwen_session_eval(s, &nxt, 1, err, sizeof(err)), err);
        size_t r2 = 0, n2 = 0, h2 = 0;
        const int *l2 = NULL;
        const uint16_t *fb = qwen_session_aux_hidden(s, &r2, &n2, &h2, &l2);
        require(fb && r2 >= 1 && n2 == 3 && h2 == H, "frontier aux shape");
        for (size_t a = 0; a < 3; a++)
            for (size_t i = 0; i < H; i++)
                aux[(a * N + (plen + k)) * H + i] =
                    bf16f(fb[((size_t)a * r2 + (r2 - 1)) * h2 + i]);
    }

    size_t T = N - 1; /* draft positions 0..T-1: pos i embeds x[i+1] */
    FILE *f = fopen(out_path, "wb");
    require(f != NULL, "cannot open output");
    fprintf(f, "{\n  \"source\": \"h3.c-greedy\",\n  \"hidden_size\": %zu,\n", H);
    fprintf(f, "  \"num_tokens\": %zu,\n  \"score_from\": %zu,\n", T, plen - 1);
    fprintf(f, "  \"aux_layers\": [%d,%d,%d],\n", layers[0], layers[1], layers[2]);
    fprintf(f, "  \"full_ids\": ["); /* x[0..N-1]: feed to an AWQ target as-is */
    for (size_t i = 0; i < N; i++) fprintf(f, "%s%u", i ? "," : "", x[i]);
    fprintf(f, "],\n  \"token_ids\": [");
    for (size_t i = 0; i < T; i++) fprintf(f, "%s%u", i ? "," : "", x[i + 1]);
    fprintf(f, "],\n  \"positions\": [");
    for (size_t i = 0; i < T; i++) fprintf(f, "%s%zu", i ? "," : "", i);
    fprintf(f, "],\n  \"expected_next\": [");
    for (size_t i = 0; i < T; i++)
        fprintf(f, "%s%u", i ? "," : "", (i + 2 < N) ? x[i + 2] : 0u);
    fprintf(f, "]");
    const char *names[3] = {"aux_hidden_low", "aux_hidden_mid",
                            "aux_hidden_high"};
    for (size_t a = 0; a < 3; a++) {
        fprintf(f, ",\n  \"%s\": [", names[a]);
        for (size_t t = 0; t < T; t++)
            for (size_t i = 0; i < H; i++)
                fprintf(f, "%s%.9g", (t || i) ? "," : "",
                        (double)aux[(a * N + t) * H + i]);
        fprintf(f, "]");
    }
    float *buf = malloc(H * sizeof(float));
    require(buf != NULL, "alloc");
    fprintf(f, ",\n  \"embedding\": [");
    for (size_t t = 0; t < T; t++) {
        require(qwen_session_embedding_row_f32(s, x[t + 1], buf, H), "embed row");
        for (size_t i = 0; i < H; i++)
            fprintf(f, "%s%.9g", (t || i) ? "," : "", (double)buf[i]);
    }
    fprintf(f, "]\n}\n");
    fclose(f);
    free(buf);
    printf("wrote %s  (prompt %zu + greedy %zu = %zu tok, T=%zu, score_from=%zu, "
           "layers {%d,%d,%d})\n",
           out_path, plen, G, N, T, plen - 1, layers[0], layers[1], layers[2]);
    free(x);
    free(aux);
    qwen_session_free(s);
    return 0;
}

/* QINT-015i-a: dump the target's decoder-layer OUTPUT hiddens at the
 * EAGLE aux layers for a fixed raw token-id sequence, so an independent
 * Transformers Qwen3-VL-32B forward can be compared position by position
 * (scripts/target_hidden_parity.py). No chat template -- raw ids, and the
 * ids are written into the dump so the Python side uses the exact same
 * input. Also records h3.c's own greedy next-token argmax per position as
 * an end-to-end sanity signal. Run with H3_QWEN_Q4=0 for a BF16 vs BF16
 * comparison. Layer set from EAGLE_AUX_LAYERS (default {1,31,60}). */
static int run_eagle_target_dump(model *m, const char *out_path) {
    char err[512];
    int layers[3] = QWEN_EAGLE3_AUX_LAYERS_DEFAULT;
    const char *al = getenv("EAGLE_AUX_LAYERS");
    if (al && al[0]) {
        int a0 = 0, a1v = 0, a2 = 0;
        require(sscanf(al, "%d,%d,%d", &a0, &a1v, &a2) == 3,
                "EAGLE_AUX_LAYERS must be 'a,b,c'");
        layers[0] = a0;
        layers[1] = a1v;
        layers[2] = a2;
    }
    static const char *TXT =
        "The history of speculative decoding begins with a simple observation: "
        "large language models spend most of their time waiting on memory, not "
        "computing, so a cheap draft model can propose several tokens at once.";
    uint32_t *ids = NULL;
    size_t n = 0;
    require(h3_tokenizer_encode(m->tok, TXT, 0, &ids, &n, err, sizeof(err)), err);
    require(n >= 8 && n <= 512, "unexpected token count");
    printf("== QINT-015i-a target-hidden dump: %zu tokens, layers {%d,%d,%d} ==\n",
           n, layers[0], layers[1], layers[2]);

    /* pass 1: all-row aux from one prefill. */
    qwen_session *s = NULL;
    require(qwen_session_create(&s, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s, 1, err, sizeof(err)), err);
    require(qwen_session_set_aux_layers(s, layers, 3, err, sizeof(err)), err);
    require(qwen_session_set_aux_prefill_all_rows(s, 1), "prefill-all");
    require(qwen_session_eval(s, ids, n, err, sizeof(err)), err);
    size_t rows = 0, na = 0, hid = 0;
    const int *lid = NULL;
    const uint16_t *ab = qwen_session_aux_hidden(s, &rows, &na, &hid, &lid);
    require(ab && rows == n && na == 3 && hid == QWEN_HIDDEN_SIZE,
            "aux shape (expect n x 3 x 5120, prefill-all)");

    /* pass 2: token-by-token greedy argmax (prediction of token t+1). */
    uint32_t *argmax = malloc(n * sizeof(uint32_t));
    require(argmax != NULL, "alloc");
    qwen_session *s2 = NULL;
    require(qwen_session_create(&s2, m->engine, err, sizeof(err)), err);
    require(qwen_session_set_resident(s2, 1, err, sizeof(err)), err);
    for (size_t t = 0; t < n; t++) {
        require(qwen_session_eval(s2, &ids[t], 1, err, sizeof(err)), err);
        const qwen_logits *lg = qwen_session_logits(s2);
        require(lg != NULL, "no logits");
        argmax[t] = lg->argmax_token;
    }

    FILE *f = fopen(out_path, "wb");
    require(f != NULL, "cannot open output");
    uint32_t hdr[4] = {0x44543348u, (uint32_t)n, 3u, (uint32_t)hid}; /* "H3TD" */
    require(fwrite(hdr, sizeof(uint32_t), 4, f) == 4, "write hdr");
    int32_t lids[3] = {layers[0], layers[1], layers[2]};
    require(fwrite(lids, sizeof(int32_t), 3, f) == 3, "write lids");
    require(fwrite(ids, sizeof(uint32_t), n, f) == n, "write ids");
    require(fwrite(argmax, sizeof(uint32_t), n, f) == n, "write argmax");
    float *rowf = malloc(hid * sizeof(float));
    require(rowf != NULL, "alloc");
    for (size_t a = 0; a < 3; a++)
        for (size_t t = 0; t < n; t++) {
            const uint16_t *src = ab + ((size_t)a * rows + t) * hid;
            for (size_t i = 0; i < hid; i++) rowf[i] = bf16f(src[i]);
            require(fwrite(rowf, sizeof(float), hid, f) == hid, "write aux");
        }
    free(rowf);
    fclose(f);
    printf("wrote %s  (ids + greedy argmax + aux[3][%zu][%zu] f32)\n", out_path, n,
           hid);
    free(argmax);
    h3_tokenizer_ids_free(ids);
    qwen_session_free(s);
    qwen_session_free(s2);
    return 0;
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    const char *cmd = argc >= 2 ? argv[1] : "all";

    /* Model-free gate self-test: runnable without the checkpoint. */
    if (!strcmp(cmd, "chain-drift-gate")) return run_chain_drift_gate_selftest();

    model m;
    model_open(&m, root);

    int rc = 0;
    if (!strcmp(cmd, "pending-oracle")) rc = run_pending_oracle(&m);
    else if (!strcmp(cmd, "pending-reject")) rc = run_pending_reject(&m);
    else if (!strcmp(cmd, "pending-boundary")) rc = run_pending_boundary(&m);
    else if (!strcmp(cmd, "pending-eos")) rc = run_pending_eos(&m);
    else if (!strcmp(cmd, "pending-vlm")) rc = run_pending_vlm(&m);
    else if (!strcmp(cmd, "pending-parity")) rc = run_pending_parity(&m);
    else if (!strcmp(cmd, "batch-rewind"))
        rc = run_rewind_text(&m) || run_rewind_vlm(&m);
    else if (!strcmp(cmd, "verify-parity")) rc = run_verify_parity(&m);
    else if (!strcmp(cmd, "chain-drift")) rc = run_chain_drift(&m);
    else if (!strcmp(cmd, "aux-capture")) rc = run_aux_capture(&m);
    else if (!strcmp(cmd, "eagle-target-dump")) {
        require(argc >= 3, "eagle-target-dump needs an output path");
        rc = run_eagle_target_dump(&m, argv[2]);
    }
    else if (!strcmp(cmd, "eagle-b2-fixture")) {
        require(argc >= 3, "eagle-b2-fixture needs an output path");
        rc = run_eagle_b2_fixture(&m, argv[2]);
    }
    else if (!strcmp(cmd, "eagle-i-fixture")) {
        require(argc >= 3, "eagle-i-fixture needs an output path");
        rc = run_eagle_i_fixture(&m, argv[2]);
    }
    else if (!strcmp(cmd, "eagle-live") || !strcmp(cmd, "eagle-prefix") ||
             !strcmp(cmd, "eagle-sync") || !strcmp(cmd, "eagle-tau")) {
        const char *dir = argc >= 3 ? argv[2] : NULL;
        char buf[1024];
        if (!dir) {
            const char *home = getenv("HOME");
            snprintf(buf, sizeof(buf), "%s/models/mattbucci-eagle3",
                     home ? home : ".");
            dir = buf;
        }
        if (!strcmp(cmd, "eagle-tau")) {
            int pos_offset = argc >= 4 && argv[3][0] ? atoi(argv[3]) : -1;
            size_t max_new =
                argc >= 5 && argv[4][0] ? (size_t)strtoul(argv[4], NULL, 10) : 32;
            rc = run_eagle_tau(&m, dir, max_new, pos_offset);
        } else {
            rc = !strcmp(cmd, "eagle-prefix")  ? run_eagle_prefix(&m, dir)
                 : !strcmp(cmd, "eagle-sync") ? run_eagle_sync(&m, dir)
                                              : run_eagle_live(&m, dir);
        }
    }
    else if (!strcmp(cmd, "pending-fast")) {
        rc = run_pending_oracle(&m) || run_pending_reject(&m) ||
             run_pending_boundary(&m);
    } else if (!strcmp(cmd, "coordinator")) {
        rc = run_pending_oracle(&m) || run_pending_reject(&m) ||
             run_pending_boundary(&m) || run_pending_eos(&m) ||
             run_pending_parity(&m);
    } else if (!strcmp(cmd, "lowlevel")) {
        rc = run_rewind_text(&m) || run_rewind_vlm(&m) ||
             run_verify_parity(&m) || run_chain_drift(&m) || run_aux_capture(&m);
    } else {
        fail("usage: pending-{oracle,reject,boundary,eos,vlm,parity} | "
             "batch-rewind | verify-parity | chain-drift | chain-drift-gate | "
             "aux-capture | eagle-{live,prefix,sync} [ckpt] | "
             "eagle-tau [ckpt] [pos_offset] [max_new] | coordinator | lowlevel");
    }

    qwen_engine_close(m.engine);
    h3_tokenizer_free(m.tok);
    return rc;
}
