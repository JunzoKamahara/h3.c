/* Builds the H3_STARTUP_CACHE: everything the DiT reads from the raw
 * transformer/ checkpoint exactly once per process (per-block norms, the
 * two BF16 token-refiner blocks, condition_proj, rope.inv_freq, and the
 * video/audio patch + final-layer heads) - see h3_dit.c's
 * startup_cache_layout() for the exact record layout and byte offsets,
 * which this tool must match exactly.
 *
 * The point: once this ~1.5 GiB cache exists, the ~66 GiB transformer/
 * directory it was built from only needs a cheap safetensors header scan
 * at startup (h3_weight_store_open), not any of its multi-GB tensor
 * payloads - so it can live on slow/removable/networked storage while only
 * this cache and the (separately built) attention/MLP int8 stream cache
 * need to be on fast local disk.
 *
 * Usage: build_startup_cache <FL2VA/transformer dir> <output cache file>
 */
#include "h3_gpu.h"
#include "h3_weights.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    ROPE_FREQS = 16,
    DIT_BLOCKS = 50,
};

#define CACHE_MAGIC "H3ST"
#define CACHE_VERSION 1u

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t text_dim;
    uint32_t video_patch;
    uint32_t audio_channels;
    uint32_t rope_freqs;
    uint32_t reserved[6];
} cache_header;

static int copy_bf16(h3_weight_store *store, h3_gpu *gpu, const char *name,
                     int ndim, const uint64_t *shape, size_t elements,
                     FILE *out, char *error, size_t error_size) {
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, ndim, shape,
                                                error, error_size);
    if (!tensor) return 0;
    uint16_t *host = malloc(elements * sizeof(*host));
    int ok = host && h3_gpu_tensor_read_bf16(tensor, host, elements) &&
             fwrite(host, sizeof(*host), elements, out) == elements;
    if (!ok && error && error_size && !error[0])
        snprintf(error, error_size, "cannot copy %s: %s", name,
                 host ? strerror(errno) : "out of memory");
    free(host);
    h3_gpu_tensor_free(tensor);
    return ok;
}

static int copy_f32(h3_weight_store *store, h3_gpu *gpu, const char *name,
                    int ndim, const uint64_t *shape, size_t elements,
                    FILE *out, char *error, size_t error_size) {
    h3_gpu_tensor *tensor = h3_weight_load_f32(store, gpu, name, ndim, shape,
                                               error, error_size);
    if (!tensor) return 0;
    float *host = malloc(elements * sizeof(*host));
    int ok = host && h3_gpu_tensor_read_f32(tensor, host, elements) &&
             fwrite(host, sizeof(*host), elements, out) == elements;
    if (!ok && error && error_size && !error[0])
        snprintf(error, error_size, "cannot copy %s: %s", name,
                 host ? strerror(errno) : "out of memory");
    free(host);
    h3_gpu_tensor_free(tensor);
    return ok;
}

#define BF1(name, width) do {                                                \
    uint64_t shape[] = {(width)};                                            \
    error[0] = '\0';                                                         \
    if (!copy_bf16(store, gpu, (name), 1, shape, (width), out,               \
                   error, sizeof(error))) {                                  \
        fprintf(stderr, "h3: %s\n", error); return 1;                        \
    }                                                                        \
} while (0)
#define BF2(name, rows, columns) do {                                        \
    uint64_t shape[] = {(rows), (columns)};                                  \
    error[0] = '\0';                                                         \
    if (!copy_bf16(store, gpu, (name), 2, shape,                             \
                   (size_t)(rows) * (size_t)(columns), out,                  \
                   error, sizeof(error))) {                                  \
        fprintf(stderr, "h3: %s\n", error); return 1;                        \
    }                                                                        \
} while (0)
#define F1(name, width) do {                                                 \
    uint64_t shape[] = {(width)};                                            \
    error[0] = '\0';                                                         \
    if (!copy_f32(store, gpu, (name), 1, shape, (width), out,                \
                  error, sizeof(error))) {                                   \
        fprintf(stderr, "h3: %s\n", error); return 1;                        \
    }                                                                        \
} while (0)
#define F2(name, rows, columns) do {                                         \
    uint64_t shape[] = {(rows), (columns)};                                  \
    error[0] = '\0';                                                         \
    if (!copy_f32(store, gpu, (name), 2, shape,                              \
                  (size_t)(rows) * (size_t)(columns), out,                   \
                  error, sizeof(error))) {                                   \
        fprintf(stderr, "h3: %s\n", error); return 1;                        \
    }                                                                        \
} while (0)

/* Same field order as load_block()/h3_dit.c's refiner record, so the
 * runtime's fixed-offset reads line up. Takes an explicit error buffer
 * (not the BF1/BF2/F1/F2 macros above, which assume a stack array named
 * `error` and take its size via sizeof - here `error` is a pointer
 * parameter, so sizeof(error) would silently give 8 instead of the real
 * buffer size). Returns 1 (matching main()'s early-return convention) on
 * failure, 0 on success. */
static int copy_full_block(h3_weight_store *store, h3_gpu *gpu,
                           const char *prefix, FILE *out,
                           char *error, size_t error_size) {
    char name[160];
    uint64_t shape2[2];
    uint64_t shape1[1];
#define NAME(suffix) (snprintf(name, sizeof(name), "%s%s", prefix, suffix), name)
#define GO(call) do { error[0] = '\0'; if (!(call)) { \
    fprintf(stderr, "h3: %s\n", error); return 1; } } while (0)
    shape1[0] = HIDDEN;
    GO(copy_bf16(store, gpu, NAME("norm1.weight"), 1, shape1, HIDDEN,
                 out, error, error_size));
    GO(copy_bf16(store, gpu, NAME("norm2.weight"), 1, shape1, HIDDEN,
                 out, error, error_size));
    shape2[0] = (uint64_t)INNER * 3; shape2[1] = HIDDEN;
    GO(copy_bf16(store, gpu, NAME("attn.qkv_proj.weight"), 2, shape2,
                 (size_t)INNER * 3 * HIDDEN, out, error, error_size));
    shape1[0] = HEAD_DIM;
    GO(copy_bf16(store, gpu, NAME("attn.q_norm.weight"), 1, shape1, HEAD_DIM,
                 out, error, error_size));
    GO(copy_bf16(store, gpu, NAME("attn.k_norm.weight"), 1, shape1, HEAD_DIM,
                 out, error, error_size));
    shape2[0] = HIDDEN; shape2[1] = INNER;
    GO(copy_bf16(store, gpu, NAME("attn.out_proj.weight"), 2, shape2,
                 (size_t)HIDDEN * INNER, out, error, error_size));
    shape2[0] = (uint64_t)FFN * 2; shape2[1] = HIDDEN;
    GO(copy_bf16(store, gpu, NAME("mlp.fc1.weight"), 2, shape2,
                 (size_t)FFN * 2 * HIDDEN, out, error, error_size));
    shape2[0] = HIDDEN; shape2[1] = FFN;
    GO(copy_bf16(store, gpu, NAME("mlp.fc2.weight"), 2, shape2,
                 (size_t)HIDDEN * FFN, out, error, error_size));
#undef NAME
#undef GO
    return 0;
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

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "h3: cannot open %s: %s\n", argv[2], strerror(errno));
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
    header.text_dim = TEXT_DIM;
    header.video_patch = VIDEO_PATCH;
    header.audio_channels = AUDIO_CHANNELS;
    header.rope_freqs = ROPE_FREQS;
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        fprintf(stderr, "h3: cannot write cache header: %s\n",
                strerror(errno));
        fclose(out);
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 1;
    }

    for (uint32_t block = 0; block < DIT_BLOCKS; block++) {
        char prefix[64], name[160];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", block);
#define NAME(suffix) (snprintf(name, sizeof(name), "%s%s", prefix, suffix), name)
        BF1(NAME("norm1.weight"), HIDDEN);
        BF1(NAME("norm2.weight"), HIDDEN);
        BF1(NAME("attn.q_norm.weight"), HEAD_DIM);
        BF1(NAME("attn.k_norm.weight"), HEAD_DIM);
#undef NAME
        fprintf(stderr, "h3: startup cache block norms %2u/%u\n",
                block + 1, DIT_BLOCKS);
    }

    BF2("condition_proj.weight", HIDDEN, TEXT_DIM);
    BF1("condition_proj.bias", HIDDEN);

    if (copy_full_block(store, gpu, "token_refiner.blocks.0.", out,
                        error, sizeof(error))) return 1;
    fprintf(stderr, "h3: startup cache token refiner block 1/2\n");
    if (copy_full_block(store, gpu, "token_refiner.blocks.1.", out,
                        error, sizeof(error))) return 1;
    fprintf(stderr, "h3: startup cache token refiner block 2/2\n");

    BF1("token_refiner.final_norm.weight", HIDDEN);
    F1("rope.inv_freq", ROPE_FREQS);
    F2("video_patch_proj.weight", HIDDEN, VIDEO_PATCH);
    F1("video_patch_proj.bias", HIDDEN);
    F2("audio_patch_proj.weight", HIDDEN, AUDIO_CHANNELS);
    F1("audio_patch_proj.bias", HIDDEN);
    BF1("final_layer.norm.weight", HIDDEN);
    F2("final_layer.video_out.weight", VIDEO_PATCH, HIDDEN);
    F1("final_layer.video_out.bias", VIDEO_PATCH);
    F2("final_layer.audio_out.weight", AUDIO_CHANNELS, HIDDEN);
    F1("final_layer.audio_out.bias", AUDIO_CHANNELS);

    fclose(out);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    fprintf(stderr, "h3: wrote startup cache to %s\n", argv[2]);
    return 0;
}
