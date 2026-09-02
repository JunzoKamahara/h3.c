#ifndef QWEN_SPEC_H
#define QWEN_SPEC_H

/* QINT-015d-2 -- greedy speculative-decoding coordinator, pending-anchor form.
 *
 * Each cycle runs ONE target batch forward and can commit up to `width`
 * tokens (the verify block is `[anchor, D1 .. D_{width-1}]`, so `width` is the
 * total number of verify rows, 2..QWEN_VERIFY_MAX -- NOT the draft count).
 *
 * State invariant, true at every cycle boundary:
 *
 *   KV cache / session history
 *       == the tokens already emitted to the caller
 *          (prompt length + emitted count == qwen_session_length()).
 *
 *   pending_anchor (when have_pending)
 *       == the single token the target has ALREADY decided comes next
 *          (the argmax after the committed prefix), but which has NOT yet been
 *          written to the KV or handed to the caller. It becomes row 0 of the
 *          next cycle's verify block, and is emitted then.
 *
 * A cycle:
 *   1. anchor T0 = pending_anchor, or the current frontier argmax.
 *      If T0 is a stop id -> stop, emit nothing.
 *   2. draft_budget = min(width - 1, remaining - 1).
 *      If draft_budget == 0 -> scalar step: append T0, emit it, refresh the
 *      pending anchor from the new logits. (This is also the fallback when the
 *      draft proposes nothing.)
 *   3. block = [T0, D1 .. Dk]  (k = accepted proposal length, 1 <= k, so the
 *      block is 2..width rows). qwen_session_verify_block appends the block.
 *   4. accepted = length of the matching prefix D1..Dj == result[0..j-1].
 *      Emit [T0, D1 .. D_accepted]; pending_anchor = result[accepted].
 *      If accepted < k, rewind the KV to drop D_{accepted+1}..Dk (a real
 *      partial-block rewind). If accepted == k, no rewind is needed.
 *   5. A stop id among the emitted tokens, or as the new pending anchor,
 *      ends generation; the stop id itself is never emitted or left in the KV.
 *
 * Steady state: 1 committed token .. `width` committed tokens per target batch
 * forward -- no extra scalar 32B sweep for the correction/bonus token, which
 * is what the old scalar coordinator paid every cycle.
 *
 * Greedy only (temperature 0); sampling is QINT-015j.
 */

#include "qwen_draft.h"
#include "qwen_engine.h"

#include <stddef.h>
#include <stdint.h>

#define QWEN_SPEC_MAX 8u /* proposal / histogram buffer headroom */

typedef struct {
    uint64_t cycles;
    uint64_t target_batches;       /* qwen_session_verify_block calls          */
    uint64_t target_rows;          /* verify rows across all batches           */
    uint64_t scalar_fallback_evals;/* single-token target steps (draft empty)  */
    uint64_t rewinds;              /* partial-block rewinds                     */
    uint64_t drafted_tokens;       /* draft tokens proposed                     */
    uint64_t accepted_tokens;      /* draft tokens that matched the target      */
    uint64_t committed_tokens;     /* tokens emitted to the caller              */
    uint64_t full_block;           /* cycles where every drafted token accepted */
    /* Histogram of accepted draft-prefix length, 0..QWEN_SPEC_MAX. */
    uint64_t accept_len[QWEN_SPEC_MAX + 1];
} qwen_spec_stats;

typedef struct {
    qwen_session *session;     /* target; KV frontier == emitted output        */
    qwen_draft_backend *draft; /* borrowed                                     */
    unsigned width;            /* total verify rows, 2..QWEN_VERIFY_MAX        */

    uint32_t pending_anchor;
    int have_pending;

    qwen_spec_stats stats;
} qwen_spec;

int qwen_spec_init(qwen_spec *spec, qwen_session *session,
                   qwen_draft_backend *draft, unsigned width,
                   char *error, size_t error_size);

/* Report of one cycle. `committed` holds the tokens emitted this cycle (0..width).
 * `hit_stop` means a stop id was reached (never itself emitted). */
typedef struct {
    uint32_t committed[QWEN_SPEC_MAX + 1];
    size_t committed_count;
    size_t accepted_from_draft;
    int hit_stop;
} qwen_spec_cycle;

/* One speculative cycle. `remaining` bounds how many more tokens may be
 * emitted (pass SIZE_MAX for "no bound"). */
int qwen_spec_step(qwen_spec *spec, size_t remaining, const uint32_t *stop_ids,
                   size_t stop_count, qwen_spec_cycle *cycle, char *error,
                   size_t error_size);

/* Greedy-generate up to `max_new` tokens, appending them to `out`
 * (capacity >= max_new). A stop id ends generation and is NOT written to
 * `out`. The produced token sequence equals a plain greedy decode from the
 * current session state (modulo the decode path's near-tie nondeterminism --
 * see qwen_verify_result / TIE_EPS in the tests). */
int qwen_spec_generate(qwen_spec *spec, size_t max_new,
                       const uint32_t *stop_ids, size_t stop_count,
                       uint32_t *out, size_t *out_count,
                       char *error, size_t error_size);

void qwen_spec_stats_print(const qwen_spec_stats *stats, const char *label);

#endif
