#ifndef QWEN_Q4_H
#define QWEN_Q4_H

/* Group-wise symmetric INT4 weights for chat decode (chat-speedup step #2).
 *
 * Only the process-wide resident weight set in qwen_kv.c is quantised, and only
 * the rows==1 decode path consumes it (h3_gpu_linear_q4_gemv). Prefill, the
 * streaming session, qwen_lm_decode_tail() and the H3 media-conditioning path
 * in h3_text_encoder.c all stay BF16, so every layer-49 parity gate is
 * unaffected.
 *
 * Opt-in: set `H3_QWEN_Q4=1` to quantise the resident decode weights (default
 * off). Naive round-to-nearest INT4 costs ~5-15% relative logit error, and the
 * decode speedup needs the per-layer submit / K-V-cache fusion (chat-speedup
 * step #3) to be worthwhile. `H3_QWEN_Q4_HEAD=1` also quantises lm_head, which
 * is off by default: it is a single matmul and the largest error contributor. */

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

#define QWEN_Q4_GROUP 128u

typedef struct {
    h3_gpu_tensor *packed;  /* I8, rows * cols/2 bytes  */
    h3_gpu_tensor *scales;  /* BF16, rows * cols/group  */
    uint32_t rows;
    uint32_t cols;
} qwen_q4_weight;

/* 1 unless H3_QWEN_Q4 is set to "0". */
int qwen_q4_enabled(void);

/* Quantise a [rows, cols] BF16 weight tensor (cols a multiple of QWEN_Q4_GROUP)
 * to `out`. Reads the source once on the host; allocates two GPU buffers. */
int qwen_q4_quantize(h3_gpu *gpu, const h3_gpu_tensor *src_bf16, uint32_t rows,
                     uint32_t cols, qwen_q4_weight *out, char *error,
                     size_t error_size);

void qwen_q4_weight_free(qwen_q4_weight *weight);

#endif
