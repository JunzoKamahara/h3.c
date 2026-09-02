#ifndef QWEN_SPEC_H
#define QWEN_SPEC_H

/* QINT-015 -- greedy speculative-decoding coordinator.
 *
 * QINT-015a scope: a *scalar* verifier. Each cycle asks the draft backend for
 * up to `width` continuation tokens, then verifies them one at a time on the
 * existing rows==1 target decode path (Mixed-W4/BF16 by default). This is not
 * faster than plain decode -- it does the same number of target forwards in
 * the worst case -- but it establishes the accept/reject/rewind algorithm and,
 * critically, produces a token sequence *identical* to a plain greedy decode
 * from the same session state. The small-batch verifier that makes it fast is
 * QINT-015d.
 *
 * Only temperature == 0 (greedy) is supported; sampling is QINT-015j.
 */

#include "qwen_draft.h"
#include "qwen_engine.h"

#include <stddef.h>
#include <stdint.h>

#define QWEN_SPEC_MAX 8u

typedef struct {
    uint64_t cycles;
    uint64_t drafted_tokens;   /* draft tokens proposed across all cycles     */
    uint64_t accepted_tokens;  /* draft tokens that matched the target        */
    uint64_t committed_tokens; /* tokens actually appended (draft + bonus)    */
    uint64_t target_evals;     /* scalar target forward passes               */
    uint64_t full_block;       /* cycles where every drafted token accepted   */
    /* Histogram of the accepted draft-prefix length per cycle, 0..QWEN_SPEC_MAX
     * (index k == "k draft tokens accepted before divergence / exhaustion"). */
    uint64_t accept_len[QWEN_SPEC_MAX + 1];
} qwen_spec_stats;

typedef struct {
    qwen_session *session;      /* target; its KV frontier is the gen head    */
    qwen_draft_backend *draft;  /* borrowed                                   */
    unsigned width;             /* 1..QWEN_SPEC_MAX                           */
    qwen_spec_stats stats;
} qwen_spec;

int qwen_spec_init(qwen_spec *spec, qwen_session *session,
                   qwen_draft_backend *draft, unsigned width,
                   char *error, size_t error_size);

/* One speculative cycle. Reports what was committed this cycle. */
typedef struct {
    uint32_t committed[QWEN_SPEC_MAX + 1];
    size_t committed_count;    /* 0 only if a stop id was the very first pred */
    size_t accepted_from_draft;
    int hit_stop;              /* a stop id was reached (not appended)        */
} qwen_spec_cycle;

int qwen_spec_step(qwen_spec *spec, const uint32_t *stop_ids, size_t stop_count,
                   qwen_spec_cycle *cycle, char *error, size_t error_size);

/* Greedy generate up to `max_new` tokens, appending them to `out` (capacity
 * >= max_new) and stopping early at any id in `stop_ids`. The stop token is
 * NOT written to `out` (matching the usual greedy loop). *out_count gets the
 * number produced. The sequence equals a plain greedy decode from the current
 * session state. */
int qwen_spec_generate(qwen_spec *spec, size_t max_new,
                       const uint32_t *stop_ids, size_t stop_count,
                       uint32_t *out, size_t *out_count,
                       char *error, size_t error_size);

void qwen_spec_stats_print(const qwen_spec_stats *stats, const char *label);

#endif
