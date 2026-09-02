#include "qwen_spec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int spec_trace(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("H3_SPEC_TRACE");
        v = e && e[0] && e[0] != '0';
    }
    return v;
}

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static int is_stop(uint32_t token, const uint32_t *stop_ids, size_t stop_count) {
    for (size_t i = 0; i < stop_count; i++)
        if (stop_ids[i] == token) return 1;
    return 0;
}

int qwen_spec_init(qwen_spec *spec, qwen_session *session,
                   qwen_draft_backend *draft, unsigned width,
                   char *error, size_t error_size) {
    if (!spec || !session) {
        set_error(error, error_size, "qwen_spec_init requires a session");
        return 0;
    }
    if (width < 2 || width > QWEN_VERIFY_MAX) {
        set_error(error, error_size,
                  "qwen_spec width is the verify-row count, 2..QWEN_VERIFY_MAX");
        return 0;
    }
    memset(spec, 0, sizeof(*spec));
    spec->session = session;
    spec->draft = draft;
    spec->width = width;
    return 1;
}

/* Append one token to the target and read back its new next-token prediction.
 * The target wall time is added to stats->target_ns. */
static int scalar_step(qwen_session *session, uint32_t token, uint32_t *next_out,
                       qwen_spec_stats *stats, char *error, size_t error_size) {
    uint64_t t0 = now_ns();
    int ok = qwen_session_eval(session, &token, 1, error, error_size);
    stats->target_ns += now_ns() - t0;
    if (!ok) return 0;
    const qwen_logits *lg = qwen_session_logits(session);
    if (!lg) {
        set_error(error, error_size, "target produced no logits");
        return 0;
    }
    *next_out = lg->argmax_token;
    return 1;
}

int qwen_spec_step(qwen_spec *spec, size_t remaining, const uint32_t *stop_ids,
                   size_t stop_count, qwen_spec_cycle *cycle, char *error,
                   size_t error_size) {
    if (!spec || !spec->session || !cycle) {
        set_error(error, error_size, "qwen_spec_step requires init");
        return 0;
    }
    memset(cycle, 0, sizeof(*cycle));
    if (remaining == 0) return 1;

    /* 1. anchor: the token the target has already decided comes next. */
    uint32_t anchor;
    if (spec->have_pending) {
        anchor = spec->pending_anchor;
    } else {
        const qwen_logits *lg = qwen_session_logits(spec->session);
        if (!lg) {
            set_error(error, error_size,
                      "qwen_spec_step: no logits (prefill first)");
            return 0;
        }
        anchor = lg->argmax_token;
    }
    if (is_stop(anchor, stop_ids, stop_count)) {
        cycle->hit_stop = 1;
        spec->have_pending = 0;
        return 1;
    }

    spec->stats.cycles++;

    /* 2. draft budget: block rows = 1 anchor + up to (width-1) draft tokens,
     * and a cycle may emit at most `remaining` tokens (anchor counts). */
    size_t budget = spec->width - 1;
    if (remaining != (size_t)-1 && remaining - 1 < budget) budget = remaining - 1;

    qwen_draft_proposal proposal;
    proposal.count = 0;
    if (budget > 0) {
        size_t hlen = 0;
        const uint32_t *hist = qwen_session_history(spec->session, &hlen);
        qwen_draft_context ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.history = hist;
        ctx.history_length = hlen;
        ctx.have_anchor = 1;
        ctx.anchor_token = anchor;
        uint64_t t0 = now_ns();
        int ok = qwen_draft_propose(spec->draft, &ctx, budget, &proposal);
        spec->stats.draft_ns += now_ns() - t0;
        if (!ok) {
            set_error(error, error_size, "draft backend failed");
            return 0;
        }
    }
    size_t k = proposal.count;
    if (k > budget) k = budget;
    spec->stats.drafted_tokens += k;

    if (spec_trace()) {
        size_t hl = qwen_session_length(spec->session);
        fprintf(stderr, "[spec] cyc=%llu len=%zu anchor=%u budget=%zu draft(%zu)=[",
                (unsigned long long)spec->stats.cycles, hl, anchor, budget, k);
        for (size_t j = 0; j < k; j++)
            fprintf(stderr, "%u%s", proposal.tokens[j], j + 1 < k ? "," : "");
        fprintf(stderr, "]\n");
    }

    /* 3a. no draft tokens -> a single scalar target step for the anchor. */
    if (k == 0) {
        uint32_t next = 0;
        if (!scalar_step(spec->session, anchor, &next, &spec->stats, error,
                         error_size))
            return 0;
        spec->stats.scalar_fallback_evals++;
        cycle->committed[cycle->committed_count++] = anchor;
        spec->stats.committed_tokens++;
        spec->stats.accept_len[0]++;
        if (is_stop(next, stop_ids, stop_count)) {
            cycle->hit_stop = 1;
            spec->have_pending = 0;
        } else {
            spec->pending_anchor = next;
            spec->have_pending = 1;
        }
        return 1;
    }

    /* 3b. verify [anchor, D1..Dk] in one batch. */
    uint32_t block[QWEN_SPEC_MAX + 1];
    block[0] = anchor;
    for (size_t i = 0; i < k; i++) block[i + 1] = proposal.tokens[i];
    size_t rows = k + 1;

    qwen_verify_result vr;
    uint64_t tv0 = now_ns();
    int vok = qwen_session_verify_block(spec->session, block, rows, &vr, error,
                                       error_size);
    spec->stats.target_ns += now_ns() - tv0;
    if (!vok) return 0;
    spec->stats.target_batches++;
    spec->stats.target_rows += rows;

    /* 4. accepted draft prefix: D_{i+1} == result[i] (argmax after block[0..i]).
     * Record position-wise conditional acceptance for QINT-015f. */
    size_t accepted = 0;
    while (accepted < k) {
        if (accepted < QWEN_SPEC_MAX) spec->stats.pos_reached[accepted]++;
        if (block[accepted + 1] != vr.top1[accepted]) break;
        if (accepted < QWEN_SPEC_MAX) spec->stats.pos_accepted[accepted]++;
        accepted++;
    }
    spec->stats.accepted_tokens += accepted;
    if (accepted <= QWEN_SPEC_MAX) spec->stats.accept_len[accepted]++;
    if (accepted == k) spec->stats.full_block++;

    /* Emit [anchor, D1..D_accepted], stopping at the first stop id. */
    cycle->committed[cycle->committed_count++] = anchor; /* checked non-stop */
    size_t emitted = 1;
    int stop = 0;
    for (size_t i = 0; i < accepted; i++) {
        uint32_t t = block[i + 1];
        if (is_stop(t, stop_ids, stop_count)) { stop = 1; break; }
        cycle->committed[cycle->committed_count++] = t;
        emitted++;
    }
    cycle->accepted_from_draft = emitted - 1;
    spec->stats.committed_tokens += emitted;

    uint32_t next_pending = vr.top1[accepted];
    if (!stop && is_stop(next_pending, stop_ids, stop_count)) stop = 1;

    /* Trim the KV back to exactly the tokens emitted this cycle. */
    size_t cur = qwen_session_length(spec->session);
    size_t keep = cur - rows + emitted;
    if (keep != cur) {
        if (!qwen_session_rewind(spec->session, keep, error, error_size))
            return 0;
        spec->stats.rewinds++;
    }

    if (spec_trace()) {
        fprintf(stderr, "[spec]   accepted=%zu emitted=%zu keep=%zu %s "
                        "next_pending=%u\n",
                accepted, emitted, keep,
                stop ? "STOP" : "pending", next_pending);
    }

    if (stop) {
        cycle->hit_stop = 1;
        spec->have_pending = 0;
        return 1;
    }
    spec->pending_anchor = next_pending;
    spec->have_pending = 1;
    return 1;
}

int qwen_spec_generate(qwen_spec *spec, size_t max_new,
                       const uint32_t *stop_ids, size_t stop_count,
                       uint32_t *out, size_t *out_count,
                       char *error, size_t error_size) {
    if (!spec || !out || !out_count) {
        set_error(error, error_size, "qwen_spec_generate requires buffers");
        return 0;
    }
    *out_count = 0;
    while (*out_count < max_new) {
        qwen_spec_cycle cycle;
        if (!qwen_spec_step(spec, max_new - *out_count, stop_ids, stop_count,
                            &cycle, error, error_size))
            return 0;
        for (size_t i = 0; i < cycle.committed_count && *out_count < max_new;
             i++)
            out[(*out_count)++] = cycle.committed[i];
        if (cycle.hit_stop) break;
        if (cycle.committed_count == 0) break; /* defensive: no progress */
    }
    return 1;
}

void qwen_spec_stats_print(const qwen_spec_stats *stats, const char *label,
                           double scalar_ref_ms) {
    if (!stats) return;
    double cycles = stats->cycles ? (double)stats->cycles : 1.0;
    double rows_per_batch =
        stats->target_batches
            ? (double)stats->target_rows / (double)stats->target_batches
            : 0.0;
    double accept_rate =
        stats->drafted_tokens
            ? (double)stats->accepted_tokens / (double)stats->drafted_tokens
            : 0.0;
    double tau = (double)stats->committed_tokens / cycles;          /* commit/cycle */
    double draft_ms = (double)stats->draft_ns / 1e6 / cycles;       /* per cycle    */
    double verify_ms =
        stats->target_batches
            ? (double)stats->target_ns / 1e6 / (double)stats->target_batches
            : (double)stats->target_ns / 1e6 / cycles;              /* per batch    */

    printf("%s: cycles=%llu batches=%llu rows/batch=%.2f committed=%llu "
           "tau=%.2f\n",
           label ? label : "spec", (unsigned long long)stats->cycles,
           (unsigned long long)stats->target_batches, rows_per_batch,
           (unsigned long long)stats->committed_tokens, tau);
    printf("  drafted=%llu accepted=%llu (%.1f%%)  full-block=%llu/%llu  "
           "scalar-fallback=%llu  rewinds=%llu\n",
           (unsigned long long)stats->drafted_tokens,
           (unsigned long long)stats->accepted_tokens, 100.0 * accept_rate,
           (unsigned long long)stats->full_block,
           (unsigned long long)stats->cycles,
           (unsigned long long)stats->scalar_fallback_evals,
           (unsigned long long)stats->rewinds);
    printf("  position acceptance a1..:");
    for (size_t j = 0; j < QWEN_SPEC_MAX; j++) {
        if (!stats->pos_reached[j]) break;
        printf(" %.2f", (double)stats->pos_accepted[j] /
                            (double)stats->pos_reached[j]);
    }
    printf("\n  T_draft/cycle=%.2f ms  T_verify/batch=%.2f ms", draft_ms,
           verify_ms);
    if (scalar_ref_ms > 0.0) {
        double s = tau * scalar_ref_ms / (verify_ms + draft_ms);
        printf("  ->  S = tau*%.1f/(%.1f+%.1f) = %.2f  (>1 beats scalar)",
               scalar_ref_ms, verify_ms, draft_ms, s);
    }
    printf("\n");
}
