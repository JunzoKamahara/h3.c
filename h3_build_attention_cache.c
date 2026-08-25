/* Builds a pre-quantized int8 cache of every DiT weight matrix (QKV,
 * attention-output, FC1, FC2), so the runtime can either keep the MLP
 * portion resident (as before) or stream all four matrices for every
 * block - the latter is what a long (~15s/362-frame) run needs, since
 * even int8-resident MLP for all 50 blocks (~10.8 GiB) stops being cheap
 * once per-sequence activations grow with frame count. See h3_dit.c's
 * H3_ATTENTION_CACHE / H3_INT8_STREAM_MLP handling for the reader side.
 *
 * Cache layout: a 64-byte header, then per block (in block order):
 *   qkv_int8[INNER*3*HIDDEN]  qkv_scales[INNER*3]   (f32)
 *   out_int8[HIDDEN*INNER]    out_scales[HIDDEN]    (f32)
 *   fc1_int8[FFN*2*HIDDEN]    fc1_scales[FFN*2]     (f32)
 *   fc2_int8[HIDDEN*FFN]      fc2_scales[HIDDEN]    (f32)
 * Quantization uses the exact same GPU routine (h3_gpu_quantize_weight_int8)
 * as the existing resident-int8 path, so results match it bit for bit; they
 * are not expected to match a BF16-only run.
 *
 * Usage: build_attention_cache <FL2VA/transformer dir> <output cache file>
 */
#include "h3_gpu.h"
#include "h3_weights.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    DIT_BLOCKS = 50,
};

#define CACHE_MAGIC "H3AC"
#define CACHE_VERSION 2u

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t reserved[10];
} cache_header;

static int quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                              const char *name, uint32_t rows,
                              uint32_t columns, FILE *out,
                              char *error, size_t error_size) {
    uint64_t shape[2] = { rows, columns };
    h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                              error, error_size);
    if (!bf16) return 0;

    size_t elements = (size_t)rows * columns;
    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
    int ok = i8 && scales &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(gpu, i8, scales, bf16, rows,
                                         columns) &&
             h3_gpu_submit(gpu);
    h3_gpu_tensor_free(bf16);
    if (!ok) {
        if (error && error_size && !error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", name,
                     h3_gpu_error(gpu));
        h3_gpu_tensor_free(i8);
        h3_gpu_tensor_free(scales);
        return 0;
    }

    int8_t *i8_host = malloc(elements);
    float *scale_host = malloc((size_t)rows * sizeof(float));
    int result = i8_host && scale_host &&
        h3_gpu_tensor_read_i8(i8, i8_host, elements) &&
        h3_gpu_tensor_read_f32(scales, scale_host, rows) &&
        fwrite(i8_host, 1, elements, out) == elements &&
        fwrite(scale_host, sizeof(float), rows, out) == rows;
    free(i8_host);
    free(scale_host);
    h3_gpu_tensor_free(i8);
    h3_gpu_tensor_free(scales);
    if (!result && error && error_size)
        snprintf(error, error_size, "cannot write cache payload for %s: %s",
                 name, strerror(errno));
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> <output cache file>\n",
                argv[0]);
        return 1;
    }
    char error[512] = {0};

    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                   sizeof(error));
    if (!store) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3: %s\n", error);
        h3_weight_store_free(store);
        return 1;
    }
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fprintf(stderr,
                "h3: this GPU lacks the int8 path the cache is built for\n");
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 1;
    }

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "h3: cannot open %s: %s\n", argv[2],
                strerror(errno));
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 1;
    }

    cache_header header = {0};
    memcpy(header.magic, CACHE_MAGIC, 4);
    header.version = CACHE_VERSION;
    header.block_count = DIT_BLOCKS;
    header.hidden = HIDDEN;
    header.inner = INNER;
    header.ffn = FFN;
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        fprintf(stderr, "h3: cannot write cache header: %s\n",
                strerror(errno));
        fclose(out);
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 1;
    }

    for (uint32_t block = 0; block < DIT_BLOCKS; block++) {
        char name[160];
        error[0] = '\0';
        snprintf(name, sizeof(name), "blocks.%u.attn.qkv_proj.weight",
                 block);
        if (!quantize_and_write(gpu, store, name, INNER * 3, HIDDEN, out,
                                error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_weight_store_free(store);
            return 1;
        }
        error[0] = '\0';
        snprintf(name, sizeof(name), "blocks.%u.attn.out_proj.weight",
                 block);
        if (!quantize_and_write(gpu, store, name, HIDDEN, INNER, out,
                                error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_weight_store_free(store);
            return 1;
        }
        error[0] = '\0';
        snprintf(name, sizeof(name), "blocks.%u.mlp.fc1.weight", block);
        if (!quantize_and_write(gpu, store, name, FFN * 2, HIDDEN, out,
                                error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_weight_store_free(store);
            return 1;
        }
        error[0] = '\0';
        snprintf(name, sizeof(name), "blocks.%u.mlp.fc2.weight", block);
        if (!quantize_and_write(gpu, store, name, HIDDEN, FFN, out,
                                error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_weight_store_free(store);
            return 1;
        }
        fprintf(stderr, "h3: attention cache block %2u/%u\n", block + 1,
                DIT_BLOCKS);
    }

    fclose(out);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    fprintf(stderr, "h3: wrote attention cache to %s\n", argv[2]);
    return 0;
}
