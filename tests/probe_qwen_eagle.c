/* QINT-015h-1a -- CLI + self-test for the EAGLE-3 compatibility probe.
 *
 *   ./h3_qwen_eagle_probe <checkpoint_dir> [<target_text_config.json>]
 *       Probe a real checkpoint. Prints the per-field report; exit code is the
 *       verdict (0 COMPATIBLE, 1 INCOMPATIBLE, 2 PROBE_ERROR).
 *
 *   ./h3_qwen_eagle_probe --selftest
 *       Build four miniature checkpoints in a temp dir and assert the probe's
 *       verdict + reasons. No model, no GPU, no network.
 */

#include "qwen_eagle_probe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *m) {
    fprintf(stderr, "FAIL probe_qwen_eagle: %s\n", m);
    exit(1);
}

/* ---- miniature safetensors writer (header + zeroed data) ---------------- */

typedef struct {
    const char *name;
    const char *dtype; /* "BF16" | "I64" | "I32" | "F32" */
    int ndim;
    uint64_t shape[4];
} mini_tensor;

static uint64_t dtype_bytes(const char *d) {
    if (!strcmp(d, "BF16") || !strcmp(d, "F16")) return 2;
    if (!strcmp(d, "F32") || !strcmp(d, "I32") || !strcmp(d, "U32")) return 4;
    if (!strcmp(d, "I64") || !strcmp(d, "U64") || !strcmp(d, "F64")) return 8;
    if (!strcmp(d, "I8") || !strcmp(d, "U8") || !strcmp(d, "BOOL")) return 1;
    die("mini: unknown dtype");
    return 0;
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot create fixture file");
    fputs(text, f);
    fclose(f);
}

static void write_safetensors(const char *path, const mini_tensor *t, size_t n) {
    /* header JSON */
    char hdr[8192];
    size_t off = 0;
    uint64_t data_cursor = 0;
    off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "{");
    for (size_t i = 0; i < n; i++) {
        uint64_t elems = 1;
        for (int d = 0; d < t[i].ndim; d++) elems *= t[i].shape[d];
        uint64_t bytes = elems * dtype_bytes(t[i].dtype);
        off += (size_t)snprintf(hdr + off, sizeof(hdr) - off,
                                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                                i ? "," : "", t[i].name, t[i].dtype);
        for (int d = 0; d < t[i].ndim; d++)
            off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "%s%llu",
                                    d ? "," : "",
                                    (unsigned long long)t[i].shape[d]);
        off += (size_t)snprintf(hdr + off, sizeof(hdr) - off,
                                "],\"data_offsets\":[%llu,%llu]}",
                                (unsigned long long)data_cursor,
                                (unsigned long long)(data_cursor + bytes));
        data_cursor += bytes;
    }
    off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "}");
    if (off >= sizeof(hdr)) die("mini: header too big");

    FILE *f = fopen(path, "wb");
    if (!f) die("cannot create safetensors fixture");
    uint64_t hlen = off;
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++) prefix[i] = (unsigned char)(hlen >> (8 * i));
    fwrite(prefix, 1, 8, f);
    fwrite(hdr, 1, off, f);
    /* zeroed data section */
    char zero[4096] = {0};
    uint64_t left = data_cursor;
    while (left) {
        size_t chunk = left > sizeof(zero) ? sizeof(zero) : (size_t)left;
        fwrite(zero, 1, chunk, f);
        left -= chunk;
    }
    fclose(f);
}

static void mkfixture(const char *base, const char *name, const char *config,
                      const mini_tensor *t, size_t n) {
    char dir[1024], p[1200];
    snprintf(dir, sizeof(dir), "%s/%s", base, name);
    mkdir(dir, 0755);
    snprintf(p, sizeof(p), "%s/config.json", dir);
    write_file(p, config);
    snprintf(p, sizeof(p), "%s/model.safetensors", dir);
    write_safetensors(p, t, n);
}

static int probe_case(const char *base, const char *name, const char *tgt_cfg,
                      qwen_eagle_verdict want, const char *reason_substr) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/%s", base, name);
    qwen_eagle_target target;
    char err[256] = {0};
    if (tgt_cfg) {
        if (!qwen_eagle_target_from_config(tgt_cfg, &target, err, sizeof(err)))
            die(err);
    } else {
        qwen_eagle_target_default(&target);
    }
    char buf[8192];
    FILE *mem = fmemopen(buf, sizeof(buf), "w");
    if (!mem) die("fmemopen");
    qwen_eagle_verdict got = qwen_eagle_probe(dir, &target, mem, err, sizeof(err));
    fclose(mem);
    printf("---- %s ----\n%s\n", name, buf);
    if (got != want) {
        fprintf(stderr, "  %s: verdict %d, wanted %d\n", name, got, want);
        die("wrong verdict");
    }
    if (reason_substr && !strstr(buf, reason_substr)) {
        fprintf(stderr, "  %s: report missing substring \"%s\"\n", name,
                reason_substr);
        die("missing expected reason");
    }
    return 0;
}

static int selftest(void) {
    char base[] = "/tmp/h3_eagle_probe_XXXXXX";
    if (!mkdtemp(base)) die("mkdtemp");

    char tgt[1200];
    snprintf(tgt, sizeof(tgt), "%s/target_mini.json", base);
    write_file(tgt,
               "{\"text_config\":{\"hidden_size\":8,\"vocab_size\":20,"
               "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
               "\"head_dim\":2,\"intermediate_size\":16,"
               "\"rope_theta\":5000000,"
               "\"rope_scaling\":{\"mrope_section\":[2,1,1],"
               "\"mrope_interleaved\":true}}}");

    /* 1. compatible miniature EAGLE-3 for the mini target. */
    {
        static const mini_tensor t[] = {
            {"fc.weight", "BF16", 2, {8, 24}},
            {"midlayer.self_attn.q_proj.weight", "BF16", 2, {8, 8}},
            {"midlayer.mlp.gate_proj.weight", "BF16", 2, {16, 8}},
            {"lm_head.weight", "BF16", 2, {6, 8}},
            {"d2t", "I64", 1, {6}},
            {"t2d", "I64", 1, {20}},
            {"embed_tokens.weight", "BF16", 2, {20, 8}},
        };
        mkfixture(base, "compat",
                  "{\"architectures\":[\"LlamaForCausalLMEagle3\"],"
                  "\"model_type\":\"llama\",\"hidden_size\":8,"
                  "\"draft_vocab_size\":6,\"vocab_size\":20,"
                  "\"num_hidden_layers\":1,\"num_attention_heads\":4,"
                  "\"num_key_value_heads\":2,\"head_dim\":2,"
                  "\"intermediate_size\":16,\"rope_theta\":5000000,"
                  "\"torch_dtype\":\"bfloat16\","
                  "\"rope_scaling\":{\"mrope_section\":[2,1,1],"
                  "\"mrope_interleaved\":true}}",
                  t, sizeof(t) / sizeof(t[0]));
        probe_case(base, "compat", tgt, QWEN_EAGLE_COMPATIBLE, "RESULT: COMPATIBLE");
    }

    /* 2. hidden-size mismatch (the 8B-EAGLE-vs-32B-target case, in miniature). */
    {
        static const mini_tensor t[] = {
            {"fc.weight", "BF16", 2, {4, 12}},
            {"midlayer.self_attn.q_proj.weight", "BF16", 2, {4, 4}},
            {"lm_head.weight", "BF16", 2, {20, 4}},
            {"t2d", "I64", 1, {20}},
        };
        mkfixture(base, "mismatch_hidden",
                  "{\"architectures\":[\"LlamaForCausalLMEagle3\"],"
                  "\"model_type\":\"llama\",\"hidden_size\":4,"
                  "\"vocab_size\":20,\"num_hidden_layers\":1,"
                  "\"torch_dtype\":\"bfloat16\"}",
                  t, sizeof(t) / sizeof(t[0]));
        probe_case(base, "mismatch_hidden", tgt, QWEN_EAGLE_INCOMPATIBLE,
                   "hidden_size: draft=4 target=8");
    }

    /* 3. reduced draft vocab, no d2t/t2d mapping tensors. */
    {
        static const mini_tensor t[] = {
            {"fc.weight", "BF16", 2, {8, 24}},
            {"midlayer.self_attn.q_proj.weight", "BF16", 2, {8, 8}},
            {"lm_head.weight", "BF16", 2, {6, 8}},
        };
        mkfixture(base, "no_vocab_map",
                  "{\"architectures\":[\"LlamaForCausalLMEagle3\"],"
                  "\"model_type\":\"llama\",\"hidden_size\":8,"
                  "\"draft_vocab_size\":6,\"vocab_size\":20,"
                  "\"num_hidden_layers\":1,\"torch_dtype\":\"bfloat16\"}",
                  t, sizeof(t) / sizeof(t[0]));
        probe_case(base, "no_vocab_map", tgt, QWEN_EAGLE_INCOMPATIBLE,
                   "no d2t mapping tensor");
    }

    /* 4. drafter's own weights are AWQ/GPTQ packed. */
    {
        static const mini_tensor t[] = {
            {"fc.weight", "BF16", 2, {8, 24}},
            {"midlayer.self_attn.q_proj.qweight", "I32", 2, {8, 1}},
            {"midlayer.self_attn.q_proj.scales", "BF16", 2, {1, 8}},
            {"midlayer.self_attn.q_proj.qzeros", "I32", 2, {1, 1}},
            {"lm_head.weight", "BF16", 2, {20, 8}},
        };
        mkfixture(base, "awq_drafter",
                  "{\"architectures\":[\"LlamaForCausalLMEagle3\"],"
                  "\"model_type\":\"llama\",\"hidden_size\":8,"
                  "\"vocab_size\":20,\"draft_vocab_size\":20,"
                  "\"num_hidden_layers\":1,\"torch_dtype\":\"float16\","
                  "\"quantization_config\":{\"quant_method\":\"awq\","
                  "\"bits\":4}}",
                  t, sizeof(t) / sizeof(t[0]));
        probe_case(base, "awq_drafter", tgt, QWEN_EAGLE_INCOMPATIBLE,
                   "drafter weights are quantized");
    }

    /* 5. missing config.json -> PROBE_ERROR, not a verdict. */
    {
        char d[1200];
        snprintf(d, sizeof(d), "%s/empty", base);
        mkdir(d, 0755);
        probe_case(base, "empty", tgt, QWEN_EAGLE_PROBE_ERROR, NULL);
    }

    printf("ok: QINT-015h-1a eagle probe self-test (5 cases)\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <checkpoint_dir> [<target_text_config.json>]\n"
                "       %s --selftest\n",
                argv[0], argv[0]);
        return 2;
    }
    qwen_eagle_target target;
    char err[256] = {0};
    if (argc >= 3) {
        if (!qwen_eagle_target_from_config(argv[2], &target, err, sizeof(err))) {
            fprintf(stderr, "target config: %s\n", err);
            return 2;
        }
    } else {
        qwen_eagle_target_default(&target);
    }
    qwen_eagle_verdict v =
        qwen_eagle_probe(argv[1], &target, stdout, err, sizeof(err));
    if (v == QWEN_EAGLE_PROBE_ERROR) fprintf(stderr, "probe error: %s\n", err);
    return (int)v;
}
