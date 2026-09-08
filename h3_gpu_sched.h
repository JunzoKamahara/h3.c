#ifndef H3_GPU_SCHED_H
#define H3_GPU_SCHED_H

/* P8-SCHED-01b: a process-wide cooperative GPU scheduler so an interactive
 * chat request stays usable while a video job runs. There is one GPU; the
 * chat decode path and the diffusion transformer run on separate Metal
 * command queues but contend for the hardware.
 *
 * The chat path brackets each GPU work unit (prefill, then every decoded
 * token) with enter()/leave(). The diffusion transformer calls
 * yield_point() at every block boundary: if a chat work unit is in flight it
 * parks there -- submitting no further diffusion work -- until that unit
 * finishes or a bounded cap elapses (so diffusion is never starved). It is
 * an interleave, never "pause until the whole chat response is done":
 * `waiters` drops to zero between every token.
 *
 * Tunables (read once, on first use):
 *   H3_GPU_SCHED=0            disable (diffusion never yields)
 *   H3_GPU_SCHED_MAX_MS=<n>   cap on one yield-point wait (default 60)
 *   H3_GPU_SCHED_EVERY=<n>    only check every n-th block boundary (default 1)
 */

void h3_gpu_sched_chat_enter(void);
void h3_gpu_sched_chat_leave(void);

/* Called by the diffusion transformer at a block boundary. `block_index` lets
 * it honour H3_GPU_SCHED_EVERY. No-op when the scheduler is disabled or no
 * chat work is waiting. */
void h3_gpu_sched_diffusion_yield_point(unsigned block_index);

/* Test / introspection. */
int h3_gpu_sched_enabled(void);
void h3_gpu_sched_set_enabled(int on);          /* runtime override (tests) */
unsigned long h3_gpu_sched_yield_count(void);   /* times diffusion actually waited */
double h3_gpu_sched_yield_seconds(void);        /* total diffusion wait */
void h3_gpu_sched_reset_stats(void);

#endif
