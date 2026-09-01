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
    h3_gpu_tensor *packed;         /* I8, rows * cols/2 bytes            */
    h3_gpu_tensor *scales;         /* BF16, rows * cols/group           */
    h3_gpu_tensor *awq_inv_scale;  /* BF16 [cols], NULL for plain RTN   */
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

/* AWQ (QINT-006/007): with a per-input-channel activation scale `act_scale`
 * [cols] (mean |x_j| over calibration), search a per-channel weight scale
 * s[j] = act_scale[j]^alpha over an alpha grid, quantise diag(s)·W group-wise,
 * and pick the alpha with the least reconstruction error. `out->awq_inv_scale`
 * receives 1/s [cols] BF16; the decode GEMV folds it into the x load, so the
 * effective compute is (diag(s)·W)_q · (x/s). Falls back to plain RTN when
 * `act_scale` is NULL. */
int qwen_q4_quantize_awq(h3_gpu *gpu, const h3_gpu_tensor *src_bf16,
                         uint32_t rows, uint32_t cols, const float *act_scale,
                         qwen_q4_weight *out, char *error, size_t error_size);

void qwen_q4_weight_free(qwen_q4_weight *weight);

/* ---- AWQ calibration capture ------------------------------------------------
 * Accumulates mean |x_j| per input channel for the four distinct decoder-layer
 * projection inputs, over whatever tokens are fed through a session while
 * H3_QWEN_AWQ_CALIB names an output path. */

typedef struct qwen_awq_calib qwen_awq_calib;

enum {                     /* projection-input slots (cols) */
    QWEN_AWQ_QKV_IN = 0,   /* input RMSNorm output    -> q_proj, k_proj, v_proj */
    QWEN_AWQ_O_IN = 1,     /* attention output        -> o_proj                */
    QWEN_AWQ_MLP_IN = 2,   /* post-attn RMSNorm output-> gate_proj, up_proj    */
    QWEN_AWQ_DOWN_IN = 3,  /* SwiGLU output           -> down_proj             */
    QWEN_AWQ_SLOTS = 4
};

qwen_awq_calib *qwen_awq_calib_new(void);
void qwen_awq_calib_free(qwen_awq_calib *calib);
/* Add `rows` rows of `cols` BF16 activations for (layer, slot). */
void qwen_awq_calib_add(qwen_awq_calib *calib, int layer, int slot,
                        const uint16_t *rows_bf16, uint32_t rows, uint32_t cols);
int qwen_awq_calib_write(const qwen_awq_calib *calib, const char *path,
                         char *error, size_t error_size);
/* Load a written file: fills `act_scale[QWEN_AWQ_SLOTS]` with malloc'd [cols]
 * arrays for `layer` (caller frees). Returns 0 on error / missing layer. */
int qwen_awq_calib_load_layer(const char *path, int layer,
                              float *act_scale_out[QWEN_AWQ_SLOTS],
                              uint32_t cols_out[QWEN_AWQ_SLOTS], char *error,
                              size_t error_size);

#endif
