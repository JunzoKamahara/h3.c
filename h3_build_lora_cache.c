/* Fuses a diffusers/peft-style LoRA adapter (separate to_q/to_k/to_v/
 * to_out/ff.net LoRA pairs, e.g. lightx2v/Minimax-h3-Turbo) into the base
 * BF16 DiT weights, then quantizes the result into exactly the same H3AC
 * v2 cache format h3_build_attention_cache.c produces.
 *
 * This is the whole trick for getting "runtime LoRA composition" out of
 * h3.c without touching the hot path at all: the fusion happens once,
 * offline, on the CPU (the low-rank deltas are tiny - rank 128 in the
 * lightx2v release - so a BLAS sgemm per projection is instant compared to
 * reading the ~62GB base model). The output file is byte-for-byte the same
 * layout an ordinary (non-LoRA) attention cache would have, so
 * H3_ATTENTION_CACHE in h3_dit.c needs zero changes to stream it - it has
 * no way to tell the difference, and that is the point.
 *
 * h3.c's own checkpoint stores QKV already fused into one
 * "attn.qkv_proj.weight" matrix ([INNER*3, HIDDEN]), while the LoRA has
 * three separate low-rank adapters (to_q/to_k/to_v, [INNER, HIDDEN] each).
 * We compute each delta separately and concatenate them along the output
 * (row) dimension in Q,K,V order to match. attn.out_proj / mlp.fc1 /
 * mlp.fc2 map onto to_out.0 / ff.net.0.proj / ff.net.2 one-to-one with no
 * concatenation needed. LoRA rank is read from each lora_A tensor's own
 * shape rather than assumed, so adapters with a different rank per
 * projection (or a different rank than the 128 lightx2v ships) still work.
 *
 * The lightx2v Minimax-h3-Turbo release ships alpha == rank (both 128), so
 * the default lora_scale of 1.0 applies no extra scaling. Pass a different
 * scale as the 4th argument for adapters where that does not hold
 * (delta = scale * B @ A).
 *
 * Usage: build_lora_cache <FL2VA/transformer dir> <lora .safetensors> \
 *                          <output cache file> [lora_scale]
 */
#include "h3_gpu.h"
#include "h3_weights.h"
#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>
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

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Reads one 2D BF16 tensor (base checkpoint or LoRA file - both go through
 * the same h3_st_header/h3_st_tensor interface) with an exactly-known
 * shape and returns it as a freshly allocated F32 host array, row-major. */
static float *read_tensor_f32(const h3_st_header *header, const char *name,
                              uint64_t rows, uint64_t columns, char *error,
                              size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        snprintf(error, error_size, "tensor %s not found", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        snprintf(error, error_size,
                 "tensor %s has unexpected dtype/shape (want BF16 [%llu,%llu])",
                 name, (unsigned long long)rows, (unsigned long long)columns);
        return NULL;
    }
    size_t elements = (size_t)rows * columns;
    uint16_t *raw = malloc(elements * sizeof(uint16_t));
    float *values = malloc(elements * sizeof(float));
    if (!raw || !values) {
        snprintf(error, error_size, "out of memory reading %s", name);
        free(raw);
        free(values);
        return NULL;
    }
    int ok = h3_st_read_data(header, tensor, raw, elements * sizeof(uint16_t),
                             error, error_size);
    if (ok) {
        for (size_t i = 0; i < elements; i++) values[i] = bf16_to_f32(raw[i]);
    }
    free(raw);
    if (!ok) {
        free(values);
        return NULL;
    }
    return values;
}

/* Loads one lora_A/lora_B pair for a projection of known [out_dim, in_dim]
 * shape, reading the rank straight off lora_A's own shape (validated
 * consistent with lora_B) rather than assuming it. */
static int load_lora_pair(const h3_st_header *lora, const char *a_name,
                          const char *b_name, uint64_t out_dim,
                          uint64_t in_dim, float **out_a, float **out_b,
                          uint64_t *out_rank, char *error, size_t error_size) {
    const h3_st_tensor *a_tensor = h3_st_find(lora, a_name);
    if (!a_tensor) {
        snprintf(error, error_size, "lora tensor %s not found", a_name);
        return 0;
    }
    if (a_tensor->dtype != H3_DTYPE_BF16 || a_tensor->ndim != 2 ||
        a_tensor->shape[1] != in_dim) {
        snprintf(error, error_size,
                 "lora_A %s has unexpected dtype/shape (want BF16 [rank,%llu])",
                 a_name, (unsigned long long)in_dim);
        return 0;
    }
    uint64_t rank = a_tensor->shape[0];

    float *a = read_tensor_f32(lora, a_name, rank, in_dim, error, error_size);
    if (!a) return 0;
    float *b = read_tensor_f32(lora, b_name, out_dim, rank, error, error_size);
    if (!b) {
        free(a);
        return 0;
    }
    *out_a = a;
    *out_b = b;
    *out_rank = rank;
    return 1;
}

/* delta[out_dim, in_dim] = scale * B[out_dim, rank] @ A[rank, in_dim],
 * added directly into base[rows_total, in_dim] starting at row_offset (so
 * three calls back to back build the concatenated QKV delta in place). */
static int fuse_lora_projection(const h3_st_header *lora,
                                const char *a_name, const char *b_name,
                                uint64_t out_dim, uint64_t in_dim,
                                float scale, float *base, uint64_t row_offset,
                                char *error, size_t error_size) {
    float *a, *b;
    uint64_t rank;
    if (!load_lora_pair(lora, a_name, b_name, out_dim, in_dim, &a, &b, &rank,
                        error, error_size))
        return 0;

    float *delta = malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!delta) {
        snprintf(error, error_size, "out of memory computing delta for %s",
                 b_name);
        free(a);
        free(b);
        return 0;
    }
    /* The LP64 cblas_sgemm is deprecated in favor of an ILP64 interface
     * behind ACCELERATE_NEW_LAPACK, but these matrices (rank <= a few
     * hundred) never approach the 32-bit index range that distinction is
     * about - silence the notice rather than take on ILP64's wider types
     * project-wide for this one tool. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               (int)out_dim, (int)in_dim, (int)rank, 1.0f, b, (int)rank, a,
               (int)in_dim, 0.0f, delta, (int)in_dim);
#pragma clang diagnostic pop
    for (uint64_t r = 0; r < out_dim; r++) {
        float *dest = base + (row_offset + r) * in_dim;
        const float *source = delta + r * in_dim;
        for (uint64_t c = 0; c < in_dim; c++) dest[c] += scale * source[c];
    }
    free(delta);
    free(b);
    free(a);
    return 1;
}

typedef struct {
    const char *a_name;
    const char *b_name;
    uint64_t out_dim;
    uint64_t row_offset;
} lora_source;

/* Loads the base BF16 weight, fuses zero or more LoRA projections into it
 * (each already given as a row-offset within the [rows, columns] matrix so
 * QKV's three separate adapters land in the right place), casts back to
 * BF16, uploads it, quantizes it exactly like h3_build_attention_cache.c
 * does, and appends the int8 payload + per-row scales to the cache file. */
static int fuse_quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                                   const h3_st_header *lora,
                                   const char *base_name, uint64_t rows,
                                   uint64_t columns, const lora_source *sources,
                                   int source_count, float scale, FILE *out,
                                   char *error, size_t error_size) {
    const h3_st_header *base_header = NULL;
    const h3_st_tensor *base_tensor = h3_weight_find(store, base_name,
                                                      &base_header);
    if (!base_tensor) {
        snprintf(error, error_size, "base tensor %s not found", base_name);
        return 0;
    }
    float *fused = read_tensor_f32(base_header, base_name, rows, columns,
                                   error, error_size);
    if (!fused) return 0;

    for (int i = 0; i < source_count; i++) {
        const lora_source *source = &sources[i];
        if (!fuse_lora_projection(lora, source->a_name, source->b_name,
                                  source->out_dim, columns, scale, fused,
                                  source->row_offset, error, error_size)) {
            free(fused);
            return 0;
        }
    }

    size_t elements = (size_t)rows * columns;
    uint16_t *fused_bf16 = malloc(elements * sizeof(uint16_t));
    if (!fused_bf16) {
        snprintf(error, error_size, "out of memory for fused %s", base_name);
        free(fused);
        return 0;
    }
    for (size_t i = 0; i < elements; i++) fused_bf16[i] = f32_to_bf16(fused[i]);
    free(fused);

    h3_gpu_tensor *bf16 = h3_gpu_tensor_from_bf16(gpu, fused_bf16, elements);
    free(fused_bf16);
    if (!bf16) {
        snprintf(error, error_size, "cannot upload fused %s: %s", base_name,
                 h3_gpu_error(gpu));
        return 0;
    }

    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
    int ok = i8 && scales &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(gpu, i8, scales, bf16, (uint32_t)rows,
                                         (uint32_t)columns) &&
             h3_gpu_submit(gpu);
    h3_gpu_tensor_free(bf16);
    if (!ok) {
        if (!error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", base_name,
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
    if (!result && !error[0])
        snprintf(error, error_size, "cannot write cache payload for %s: %s",
                 base_name, strerror(errno));
    return result;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> <lora .safetensors> "
                "<output cache file> [lora_scale]\n", argv[0]);
        return 1;
    }
    float lora_scale = argc == 5 ? (float)atof(argv[4]) : 1.0f;
    char error[512] = {0};

    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                   sizeof(error));
    if (!store) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    h3_st_header lora;
    if (!h3_st_read_header(argv[2], &lora, error, sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
        h3_weight_store_free(store);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3: %s\n", error);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fprintf(stderr,
                "h3: this GPU lacks the int8 path the cache is built for\n");
        h3_gpu_free(gpu);
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }

    FILE *out = fopen(argv[3], "wb");
    if (!out) {
        fprintf(stderr, "h3: cannot open %s: %s\n", argv[3],
                strerror(errno));
        h3_gpu_free(gpu);
        h3_st_free_header(&lora);
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
        h3_st_free_header(&lora);
        h3_weight_store_free(store);
        return 1;
    }

    fprintf(stderr, "h3: fusing LoRA (scale=%.4f) into %u blocks\n",
            (double)lora_scale, (unsigned)DIT_BLOCKS);

    for (uint32_t block = 0; block < DIT_BLOCKS; block++) {
        char base_name[160];
        char qa[160], qb[160], ka[160], kb[160], va[160], vb[160];
        error[0] = '\0';

        snprintf(base_name, sizeof(base_name),
                 "blocks.%u.attn.qkv_proj.weight", block);
        snprintf(qa, sizeof(qa),
                 "transformer_blocks.%u.attn.to_q.lora_A.default.weight",
                 block);
        snprintf(qb, sizeof(qb),
                 "transformer_blocks.%u.attn.to_q.lora_B.default.weight",
                 block);
        snprintf(ka, sizeof(ka),
                 "transformer_blocks.%u.attn.to_k.lora_A.default.weight",
                 block);
        snprintf(kb, sizeof(kb),
                 "transformer_blocks.%u.attn.to_k.lora_B.default.weight",
                 block);
        snprintf(va, sizeof(va),
                 "transformer_blocks.%u.attn.to_v.lora_A.default.weight",
                 block);
        snprintf(vb, sizeof(vb),
                 "transformer_blocks.%u.attn.to_v.lora_B.default.weight",
                 block);
        lora_source qkv_sources[3] = {
            { qa, qb, INNER, 0 },
            { ka, kb, INNER, INNER },
            { va, vb, INNER, (uint64_t)INNER * 2 },
        };
        if (!fuse_quantize_and_write(gpu, store, &lora, base_name,
                                     (uint64_t)INNER * 3, HIDDEN, qkv_sources,
                                     3, lora_scale, out, error,
                                     sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_st_free_header(&lora);
            h3_weight_store_free(store);
            return 1;
        }

        char oa[160], ob[160];
        error[0] = '\0';
        snprintf(base_name, sizeof(base_name),
                 "blocks.%u.attn.out_proj.weight", block);
        snprintf(oa, sizeof(oa),
                 "transformer_blocks.%u.attn.to_out.0.lora_A.default.weight",
                 block);
        snprintf(ob, sizeof(ob),
                 "transformer_blocks.%u.attn.to_out.0.lora_B.default.weight",
                 block);
        lora_source out_source[1] = { { oa, ob, HIDDEN, 0 } };
        if (!fuse_quantize_and_write(gpu, store, &lora, base_name, HIDDEN,
                                     INNER, out_source, 1, lora_scale, out,
                                     error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_st_free_header(&lora);
            h3_weight_store_free(store);
            return 1;
        }

        char f1a[160], f1b[160];
        error[0] = '\0';
        snprintf(base_name, sizeof(base_name), "blocks.%u.mlp.fc1.weight",
                 block);
        snprintf(f1a, sizeof(f1a),
                 "transformer_blocks.%u.ff.net.0.proj.lora_A.default.weight",
                 block);
        snprintf(f1b, sizeof(f1b),
                 "transformer_blocks.%u.ff.net.0.proj.lora_B.default.weight",
                 block);
        lora_source fc1_source[1] = { { f1a, f1b, (uint64_t)FFN * 2, 0 } };
        if (!fuse_quantize_and_write(gpu, store, &lora, base_name,
                                     (uint64_t)FFN * 2, HIDDEN, fc1_source, 1,
                                     lora_scale, out, error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_st_free_header(&lora);
            h3_weight_store_free(store);
            return 1;
        }

        char f2a[160], f2b[160];
        error[0] = '\0';
        snprintf(base_name, sizeof(base_name), "blocks.%u.mlp.fc2.weight",
                 block);
        snprintf(f2a, sizeof(f2a),
                 "transformer_blocks.%u.ff.net.2.lora_A.default.weight",
                 block);
        snprintf(f2b, sizeof(f2b),
                 "transformer_blocks.%u.ff.net.2.lora_B.default.weight",
                 block);
        lora_source fc2_source[1] = { { f2a, f2b, HIDDEN, 0 } };
        if (!fuse_quantize_and_write(gpu, store, &lora, base_name, HIDDEN,
                                     FFN, fc2_source, 1, lora_scale, out,
                                     error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            fclose(out);
            h3_gpu_free(gpu);
            h3_st_free_header(&lora);
            h3_weight_store_free(store);
            return 1;
        }

        fprintf(stderr, "h3: lora cache block %2u/%u\n", block + 1,
                DIT_BLOCKS);
    }

    fclose(out);
    h3_gpu_free(gpu);
    h3_st_free_header(&lora);
    h3_weight_store_free(store);
    fprintf(stderr, "h3: wrote LoRA-fused attention cache to %s\n", argv[3]);
    return 0;
}
