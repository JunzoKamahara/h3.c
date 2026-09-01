#include "qwen_q4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int qwen_q4_enabled(void) {
    const char *value = getenv("H3_QWEN_Q4");
    return value && value[0] && !(value[0] == '0' && value[1] == '\0');
}

void qwen_q4_weight_free(qwen_q4_weight *weight) {
    if (!weight) return;
    h3_gpu_tensor_free(weight->packed);
    h3_gpu_tensor_free(weight->scales);
    h3_gpu_tensor_free(weight->awq_inv_scale);
    memset(weight, 0, sizeof(*weight));
}

/* Group-wise symmetric RTN of `w` [rows, cols] (optionally pre-scaled per
 * column by `col_scale`, may be NULL) into `packed` (rows*cols/2 bytes) and
 * `scales` (rows*groups BF16). */
static void quantize_rtn(const float *w, uint32_t rows, uint32_t cols,
                         const float *col_scale, uint8_t *packed,
                         uint16_t *scales) {
    size_t groups_per_row = cols / QWEN_Q4_GROUP;
    for (uint32_t r = 0; r < rows; r++) {
        const float *wr = w + (size_t)r * cols;
        for (size_t g = 0; g < groups_per_row; g++) {
            size_t c0 = g * QWEN_Q4_GROUP;
            float amax = 0.0f;
            for (uint32_t i = 0; i < QWEN_Q4_GROUP; i++) {
                float v = wr[c0 + i];
                if (col_scale) v *= col_scale[c0 + i];
                v = fabsf(v);
                if (v > amax) amax = v;
            }
            float scale = amax > 0.0f ? amax / 8.0f : 1.0f;
            scales[(size_t)r * groups_per_row + g] = f32_to_bf16(scale);
            float inv = 1.0f / scale;
            for (uint32_t i = 0; i < QWEN_Q4_GROUP; i += 2) {
                float v0 = wr[c0 + i];
                float v1 = wr[c0 + i + 1];
                if (col_scale) { v0 *= col_scale[c0 + i]; v1 *= col_scale[c0 + i + 1]; }
                long q0 = lroundf(v0 * inv);
                long q1 = lroundf(v1 * inv);
                if (q0 < -8) q0 = -8; else if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8; else if (q1 > 7) q1 = 7;
                uint8_t lo = (uint8_t)(q0 + 8) & 0x0Fu;
                uint8_t hi = (uint8_t)(q1 + 8) & 0x0Fu;
                packed[((size_t)r * cols + c0 + i) / 2] =
                    (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

static int emit_weight(h3_gpu *gpu, uint32_t rows, uint32_t cols,
                       const uint8_t *packed, const uint16_t *scales,
                       const uint16_t *inv_scale, qwen_q4_weight *out,
                       char *error, size_t error_size) {
    size_t packed_bytes = (size_t)rows * cols / 2;
    size_t scale_count = (size_t)rows * (cols / QWEN_Q4_GROUP);
    out->packed = h3_gpu_tensor_from_i8(gpu, packed, packed_bytes);
    out->scales = h3_gpu_tensor_from_bf16(gpu, scales, scale_count);
    out->awq_inv_scale =
        inv_scale ? h3_gpu_tensor_from_bf16(gpu, inv_scale, cols) : NULL;
    if (!out->packed || !out->scales || (inv_scale && !out->awq_inv_scale)) {
        if (error) snprintf(error, error_size,
                            "qwen_q4: cannot allocate GPU buffers: %s",
                            h3_gpu_error(gpu));
        qwen_q4_weight_free(out);
        return 0;
    }
    out->rows = rows;
    out->cols = cols;
    return 1;
}

static int read_src_f32(const h3_gpu_tensor *src_bf16, size_t count, float **out,
                        char *error, size_t error_size) {
    uint16_t *bf = malloc(count * sizeof(*bf));
    float *f = malloc(count * sizeof(*f));
    if (!bf || !f) {
        if (error) snprintf(error, error_size, "qwen_q4: out of memory");
        free(bf); free(f); return 0;
    }
    if (!h3_gpu_tensor_read_bf16(src_bf16, bf, count)) {
        if (error) snprintf(error, error_size,
                            "qwen_q4: cannot read source weight");
        free(bf); free(f); return 0;
    }
    for (size_t i = 0; i < count; i++) f[i] = bf16_to_f32(bf[i]);
    free(bf);
    *out = f;
    return 1;
}

int qwen_q4_quantize(h3_gpu *gpu, const h3_gpu_tensor *src_bf16, uint32_t rows,
                     uint32_t cols, qwen_q4_weight *out, char *error,
                     size_t error_size) {
    return qwen_q4_quantize_awq(gpu, src_bf16, rows, cols, NULL, out, error,
                                error_size);
}

int qwen_q4_quantize_awq(h3_gpu *gpu, const h3_gpu_tensor *src_bf16,
                         uint32_t rows, uint32_t cols, const float *act_scale,
                         qwen_q4_weight *out, char *error, size_t error_size) {
    memset(out, 0, sizeof(*out));
    if (cols == 0 || cols % QWEN_Q4_GROUP != 0 || (cols & 1u) != 0) {
        if (error) snprintf(error, error_size,
                            "qwen_q4: cols=%u not a multiple of %u", cols,
                            QWEN_Q4_GROUP);
        return 0;
    }
    size_t count = (size_t)rows * cols;
    float *w = NULL;
    if (!read_src_f32(src_bf16, count, &w, error, error_size)) return 0;

    uint8_t *packed = malloc(count / 2);
    uint16_t *scales = malloc((size_t)rows * (cols / QWEN_Q4_GROUP) *
                              sizeof(*scales));
    if (!packed || !scales) {
        if (error) snprintf(error, error_size, "qwen_q4: out of memory");
        free(w); free(packed); free(scales);
        return 0;
    }

    uint16_t *inv_scale = NULL;
    if (act_scale) {
        /* AWQ: s[j] = (act_scale[j] / mean_j act_scale)^alpha, grid-searched.
         * Reconstruction error is activation-weighted and row-subsampled. */
        float *s = malloc(cols * sizeof(*s));
        float *best_s = malloc(cols * sizeof(*best_s));
        inv_scale = malloc(cols * sizeof(*inv_scale));
        if (!s || !best_s || !inv_scale) {
            if (error) snprintf(error, error_size, "qwen_q4 AWQ: out of memory");
            free(w); free(packed); free(scales);
            free(s); free(best_s); free(inv_scale);
            return 0;
        }
        double amean = 0.0;
        for (uint32_t j = 0; j < cols; j++) amean += act_scale[j];
        amean = amean / cols;
        if (amean <= 0.0) amean = 1.0;

        uint32_t row_step = rows > 1024 ? rows / 512 : 1;
        size_t gpr = cols / QWEN_Q4_GROUP;
        double best_err = -1.0;
        for (int step = 1; step <= 9; step++) {
            float alpha = 0.1f * (float)step;         /* 0.1 .. 0.9 */
            float smin = 1e30f, smax = -1e30f;
            for (uint32_t j = 0; j < cols; j++) {
                float sv = powf((float)(act_scale[j] / amean), alpha);
                if (sv < 1e-4f) sv = 1e-4f;
                s[j] = sv;
                if (sv < smin) smin = sv;
                if (sv > smax) smax = sv;
            }
            float norm = 1.0f / sqrtf(smax * smin);   /* keep s centered ~1 */
            for (uint32_t j = 0; j < cols; j++) s[j] *= norm;

            double err = 0.0;
            for (uint32_t r = 0; r < rows; r += row_step) {
                const float *wr = w + (size_t)r * cols;
                for (size_t g = 0; g < gpr; g++) {
                    size_t c0 = g * QWEN_Q4_GROUP;
                    float amax = 0.0f;
                    for (uint32_t i = 0; i < QWEN_Q4_GROUP; i++) {
                        float v = fabsf(wr[c0 + i] * s[c0 + i]);
                        if (v > amax) amax = v;
                    }
                    float qs = amax > 0.0f ? amax / 8.0f : 1.0f;
                    float inv = 1.0f / qs;
                    for (uint32_t i = 0; i < QWEN_Q4_GROUP; i++) {
                        size_t j = c0 + i;
                        long q = lroundf(wr[j] * s[j] * inv);
                        if (q < -8) q = -8; else if (q > 7) q = 7;
                        float wq = (float)q * qs / s[j];
                        float d = (wq - wr[j]) * (float)(act_scale[j] / amean);
                        err += (double)d * d;
                    }
                }
            }
            if (best_err < 0.0 || err < best_err) {
                best_err = err;
                memcpy(best_s, s, cols * sizeof(*s));
            }
        }
        for (uint32_t j = 0; j < cols; j++)
            inv_scale[j] = f32_to_bf16(1.0f / best_s[j]);
        quantize_rtn(w, rows, cols, best_s, packed, scales);
        free(s); free(best_s);
    } else {
        quantize_rtn(w, rows, cols, NULL, packed, scales);
    }

    int ok = emit_weight(gpu, rows, cols, packed, scales, inv_scale, out, error,
                         error_size);
    free(w); free(packed); free(scales); free(inv_scale);
    return ok;
}

/* ---- AWQ calibration capture --------------------------------------------- */

#define AWQ_LAYERS 64

struct qwen_awq_calib {
    double *sum[QWEN_AWQ_SLOTS][AWQ_LAYERS];  /* [cols] running Σ|x_j|        */
    uint64_t count[QWEN_AWQ_SLOTS][AWQ_LAYERS];
    uint32_t cols[QWEN_AWQ_SLOTS][AWQ_LAYERS];
};

qwen_awq_calib *qwen_awq_calib_new(void) {
    return calloc(1, sizeof(qwen_awq_calib));
}

void qwen_awq_calib_free(qwen_awq_calib *calib) {
    if (!calib) return;
    for (int s = 0; s < QWEN_AWQ_SLOTS; s++)
        for (int l = 0; l < AWQ_LAYERS; l++) free(calib->sum[s][l]);
    free(calib);
}

void qwen_awq_calib_add(qwen_awq_calib *calib, int layer, int slot,
                        const uint16_t *rows_bf16, uint32_t rows,
                        uint32_t cols) {
    if (!calib || layer < 0 || layer >= AWQ_LAYERS || slot < 0 ||
        slot >= QWEN_AWQ_SLOTS)
        return;
    if (!calib->sum[slot][layer]) {
        calib->sum[slot][layer] = calloc(cols, sizeof(double));
        calib->cols[slot][layer] = cols;
        if (!calib->sum[slot][layer]) return;
    }
    if (calib->cols[slot][layer] != cols) return;
    double *acc = calib->sum[slot][layer];
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t *x = rows_bf16 + (size_t)r * cols;
        for (uint32_t j = 0; j < cols; j++)
            acc[j] += fabs((double)bf16_to_f32(x[j]));
    }
    calib->count[slot][layer] += rows;
}

#define AWQ_MAGIC 0x43515741u  /* "AWQC" */

int qwen_awq_calib_write(const qwen_awq_calib *calib, const char *path,
                         char *error, size_t error_size) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        if (error) snprintf(error, error_size, "cannot create %s", path);
        return 0;
    }
    uint32_t header[3] = {AWQ_MAGIC, QWEN_AWQ_SLOTS, AWQ_LAYERS};
    int ok = fwrite(header, sizeof(header), 1, f) == 1;
    for (int s = 0; s < QWEN_AWQ_SLOTS && ok; s++) {
        for (int l = 0; l < AWQ_LAYERS && ok; l++) {
            uint32_t cols = calib->sum[s][l] ? calib->cols[s][l] : 0;
            uint64_t cnt = calib->count[s][l];
            ok = fwrite(&cols, sizeof(cols), 1, f) == 1 &&
                 fwrite(&cnt, sizeof(cnt), 1, f) == 1;
            if (ok && cols) {
                float *mean = malloc(cols * sizeof(*mean));
                if (!mean) { ok = 0; break; }
                double d = cnt ? (double)cnt : 1.0;
                for (uint32_t j = 0; j < cols; j++)
                    mean[j] = (float)(calib->sum[s][l][j] / d);
                ok = fwrite(mean, sizeof(*mean), cols, f) == cols;
                free(mean);
            }
        }
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok && error) snprintf(error, error_size, "short write to %s", path);
    return ok;
}

int qwen_awq_calib_load_layer(const char *path, int layer,
                              float *act_scale_out[QWEN_AWQ_SLOTS],
                              uint32_t cols_out[QWEN_AWQ_SLOTS], char *error,
                              size_t error_size) {
    for (int s = 0; s < QWEN_AWQ_SLOTS; s++) {
        act_scale_out[s] = NULL;
        cols_out[s] = 0;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (error) snprintf(error, error_size, "cannot open %s", path);
        return 0;
    }
    uint32_t header[3];
    int ok = fread(header, sizeof(header), 1, f) == 1 &&
             header[0] == AWQ_MAGIC && header[1] == QWEN_AWQ_SLOTS &&
             (int)header[2] == AWQ_LAYERS && layer >= 0 && layer < AWQ_LAYERS;
    for (int s = 0; s < QWEN_AWQ_SLOTS && ok; s++) {
        for (int l = 0; l < AWQ_LAYERS && ok; l++) {
            uint32_t cols;
            uint64_t cnt;
            ok = fread(&cols, sizeof(cols), 1, f) == 1 &&
                 fread(&cnt, sizeof(cnt), 1, f) == 1;
            if (!ok) break;
            if (!cols) continue;
            if (l == layer) {
                float *v = malloc(cols * sizeof(*v));
                ok = v && fread(v, sizeof(*v), cols, f) == cols;
                if (ok) { act_scale_out[s] = v; cols_out[s] = cols; }
                else free(v);
            } else {
                ok = fseek(f, (long)(cols * sizeof(float)), SEEK_CUR) == 0;
            }
        }
    }
    fclose(f);
    if (!ok) {
        for (int s = 0; s < QWEN_AWQ_SLOTS; s++) {
            free(act_scale_out[s]);
            act_scale_out[s] = NULL;
        }
        if (error && error[0] == '\0')
            snprintf(error, error_size, "malformed calib file %s", path);
    }
    return ok;
}
