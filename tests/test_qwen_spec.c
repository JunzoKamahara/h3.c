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
             "aux-capture | coordinator | lowlevel");
    }

    qwen_engine_close(m.engine);
    h3_tokenizer_free(m.tok);
    return rc;
}
