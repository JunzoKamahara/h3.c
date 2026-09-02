#include "qwen_spec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (width < 1 || width > QWEN_SPEC_MAX) {
        set_error(error, error_size, "qwen_spec width must be 1..8");
        return 0;
    }
    memset(spec, 0, sizeof(*spec));
    spec->session = session;
    spec->draft = draft;
    spec->width = width;
    return 1;
}

/* Append one token to the target and refresh its next-token prediction.
 * `*pred_out` receives the new argmax (the target's guess for the position
 * after `token`). */
static int target_eval(qwen_session *session, uint32_t token, uint32_t *pred_out,
                       uint64_t *eval_counter, char *error, size_t error_size) {
    if (!qwen_session_eval(session, &token, 1, error, error_size)) return 0;
    (*eval_counter)++;
    const qwen_logits *logits = qwen_session_logits(session);
    if (!logits) {
        set_error(error, error_size, "target produced no logits");
        return 0;
    }
    *pred_out = logits->argmax_token;
    return 1;
}

int qwen_spec_step(qwen_spec *spec, const uint32_t *stop_ids, size_t stop_count,
                   qwen_spec_cycle *cycle, char *error, size_t error_size) {
    if (!spec || !spec->session || !cycle) {
        set_error(error, error_size, "qwen_spec_step requires init");
        return 0;
    }
    memset(cycle, 0, sizeof(*cycle));

    const qwen_logits *logits = qwen_session_logits(spec->session);
    if (!logits) {
        set_error(error, error_size,
                  "qwen_spec_step: session has no logits (prefill first)");
        return 0;
    }
    /* Target's prediction for the current frontier position. */
    uint32_t pred = logits->argmax_token;

    /* Draft proposal from the full history. */
    size_t history_length = 0;
    const uint32_t *history =
        qwen_session_history(spec->session, &history_length);
    qwen_draft_proposal proposal;
    if (!qwen_draft_propose(spec->draft, history, history_length,
                            spec->width, &proposal)) {
        set_error(error, error_size, "draft backend failed");
        return 0;
    }
    size_t d = proposal.count;
    if (d > spec->width) d = spec->width;

    if (spec_trace()) {
        fprintf(stderr, "[spec] cyc=%llu len=%zu pred=%u width=%u draft(%zu)=[",
                (unsigned long long)spec->stats.cycles, history_length, pred,
                spec->width, d);
        for (size_t j = 0; j < d; j++)
            fprintf(stderr, "%u%s", proposal.tokens[j], j + 1 < d ? "," : "");
        fprintf(stderr, "]\n");
    }

    spec->stats.cycles++;
    spec->stats.drafted_tokens += d;

    size_t i = 0;
    for (;;) {
        if (i < d && proposal.tokens[i] == pred) {
            /* Draft token i is what the target would greedily emit. */
            if (is_stop(pred, stop_ids, stop_count)) {
                cycle->hit_stop = 1;
                break;
            }
            uint32_t next_pred = 0;
            if (!target_eval(spec->session, pred, &next_pred,
                             &spec->stats.target_evals, error, error_size))
                return 0;
            cycle->committed[cycle->committed_count++] = pred;
            cycle->accepted_from_draft++;
            pred = next_pred;
            i++;
            continue;
        }
        /* Divergence (or draft exhausted): the target's own `pred` is the
         * correction / bonus token. Commit it and end the cycle. */
        if (is_stop(pred, stop_ids, stop_count)) {
            cycle->hit_stop = 1;
            break;
        }
        uint32_t next_pred = 0;
        if (!target_eval(spec->session, pred, &next_pred,
                         &spec->stats.target_evals, error, error_size))
            return 0;
        cycle->committed[cycle->committed_count++] = pred;
        pred = next_pred;
        break;
    }

    spec->stats.accepted_tokens += cycle->accepted_from_draft;
    spec->stats.committed_tokens += cycle->committed_count;
    if (cycle->accepted_from_draft <= QWEN_SPEC_MAX)
        spec->stats.accept_len[cycle->accepted_from_draft]++;
    if (d > 0 && cycle->accepted_from_draft == d) spec->stats.full_block++;

    if (spec_trace()) {
        fprintf(stderr, "[spec]   accepted=%zu committed(%zu)=[",
                cycle->accepted_from_draft, cycle->committed_count);
        for (size_t j = 0; j < cycle->committed_count; j++)
            fprintf(stderr, "%u%s", cycle->committed[j],
                    j + 1 < cycle->committed_count ? "," : "");
        fprintf(stderr, "]%s\n", cycle->hit_stop ? " STOP" : "");
    }
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
    unsigned orig_width = spec->width;
    while (*out_count < max_new) {
        /* Clamp the block so a cycle never commits past max_new: with `r`
         * tokens still wanted, at most `r - 1` may come from the draft (the
         * cycle always adds one target/bonus token). Keeps the session
         * frontier exactly at the requested boundary. */
        size_t remaining = max_new - *out_count;
        spec->width =
            (remaining - 1) < orig_width ? (unsigned)(remaining - 1) : orig_width;
        qwen_spec_cycle cycle;
        int ok = qwen_spec_step(spec, stop_ids, stop_count, &cycle, error,
                                error_size);
        spec->width = orig_width;
        if (!ok) return 0;
        for (size_t i = 0; i < cycle.committed_count && *out_count < max_new;
             i++)
            out[(*out_count)++] = cycle.committed[i];
        if (cycle.hit_stop) break;
        if (cycle.committed_count == 0) break; /* defensive: no progress */
    }
    return 1;
}

void qwen_spec_stats_print(const qwen_spec_stats *stats, const char *label) {
    if (!stats) return;
    double mean_prefix =
        stats->cycles ? (double)stats->accepted_tokens / (double)stats->cycles
                      : 0.0;
    double tokens_per_cycle =
        stats->cycles ? (double)stats->committed_tokens / (double)stats->cycles
                      : 0.0;
    double accept_rate =
        stats->drafted_tokens
            ? (double)stats->accepted_tokens / (double)stats->drafted_tokens
            : 0.0;
    printf("%s: cycles=%llu drafted=%llu accepted=%llu (%.1f%%) "
           "committed=%llu target_evals=%llu\n",
           label ? label : "spec", (unsigned long long)stats->cycles,
           (unsigned long long)stats->drafted_tokens,
           (unsigned long long)stats->accepted_tokens, 100.0 * accept_rate,
           (unsigned long long)stats->committed_tokens,
           (unsigned long long)stats->target_evals);
    printf("  mean accepted prefix = %.2f   committed/cycle = %.2f   "
           "full-block = %llu/%llu\n",
           mean_prefix, tokens_per_cycle,
           (unsigned long long)stats->full_block,
           (unsigned long long)stats->cycles);
    printf("  accept-len histogram:");
    for (size_t k = 0; k <= QWEN_SPEC_MAX; k++)
        printf(" %zu:%llu", k, (unsigned long long)stats->accept_len[k]);
    printf("\n");
}
