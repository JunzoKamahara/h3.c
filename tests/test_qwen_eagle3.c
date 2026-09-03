/* QINT-015h-1b -- load + shape-validation + reference-forward smoke for the
 * EAGLE-3 draft head. NO coordinator, NO parity claim (that is 1c).
 *
 *   ./h3_qwen_eagle3_test --selftest        # model-free: miniature checkpoint
 *   ./h3_qwen_eagle3_test <checkpoint_dir>  # load a real checkpoint + forward
 */

#include "qwen_eagle3.h"

#include "h3_json.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *m) {
    fprintf(stderr, "FAIL test_qwen_eagle3: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) die(m); }

/* ---- miniature checkpoint writer -------------------------------------- */

typedef struct {
    const char *name;
    const char *dtype; /* BF16 | I64 | BOOL */
    int ndim;
    uint64_t shape[2];
} mten;

static uint64_t dbytes(const char *d) {
    if (!strcmp(d, "BF16")) return 2;
    if (!strcmp(d, "I64")) return 8;
    if (!strcmp(d, "BOOL") || !strcmp(d, "I8")) return 1;
    die("mini: bad dtype");
    return 0;
}

static void wfile(const char *p, const char *s) {
    FILE *f = fopen(p, "wb");
    if (!f) die("cannot write fixture");
    fputs(s, f);
    fclose(f);
}

/* zeroed data, except d2t which we fill with a valid delta map. */
static void wsafetensors(const char *path, const mten *t, size_t n,
                         int draft_vocab) {
    char hdr[8192];
    size_t off = 0;
    uint64_t cur = 0;
    off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "{");
    for (size_t i = 0; i < n; i++) {
        uint64_t el = 1;
        for (int d = 0; d < t[i].ndim; d++) el *= t[i].shape[d];
        uint64_t b = el * dbytes(t[i].dtype);
        off += (size_t)snprintf(hdr + off, sizeof(hdr) - off,
                                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                                i ? "," : "", t[i].name, t[i].dtype);
        for (int d = 0; d < t[i].ndim; d++)
            off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "%s%llu",
                                    d ? "," : "",
                                    (unsigned long long)t[i].shape[d]);
        off += (size_t)snprintf(hdr + off, sizeof(hdr) - off,
                                "],\"data_offsets\":[%llu,%llu]}",
                                (unsigned long long)cur,
                                (unsigned long long)(cur + b));
        cur += b;
    }
    off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "}");
    if (off >= sizeof(hdr)) die("mini: header overflow");

    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write safetensors");
    uint64_t hlen = off;
    unsigned char pfx[8];
    for (int i = 0; i < 8; i++) pfx[i] = (unsigned char)(hlen >> (8 * i));
    fwrite(pfx, 1, 8, f);
    fwrite(hdr, 1, off, f);
    /* data */
    uint64_t written = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t el = 1;
        for (int d = 0; d < t[i].ndim; d++) el *= t[i].shape[d];
        uint64_t b = el * dbytes(t[i].dtype);
        if (!strcmp(t[i].name, "d2t")) {
            for (uint64_t j = 0; j < el; j++) {
                int64_t delta = (int64_t)(j * 2); /* target = j + 2j = 3j */
                fwrite(&delta, 8, 1, f);
            }
        } else {
            char z[512] = {0};
            uint64_t left = b;
            while (left) {
                size_t ch = left > sizeof(z) ? sizeof(z) : (size_t)left;
                fwrite(z, 1, ch, f);
                left -= ch;
            }
        }
        written += b;
    }
    (void)written;
    (void)draft_vocab;
    fclose(f);
}

/* mini EAGLE-3: hidden 8, heads 4/2, head_dim 2 (q_dim 8, kv_dim 4, qkv_in
 * 16 = 2*hidden), intermediate 16, draft vocab 6, target vocab 20, fusion 3. */
static void write_mini(const char *dir, const char *config_json,
                       int with_extra, int drop_norm) {
    mkdir(dir, 0755);
    char p[1200];
    snprintf(p, sizeof(p), "%s/config.json", dir);
    wfile(p, config_json);

    mten t[] = {
        {"fc.weight", "BF16", 2, {8, 24}},
        {"midlayer.input_layernorm.weight", "BF16", 1, {8, 0}},
        {"midlayer.hidden_norm.weight", "BF16", 1, {8, 0}},
        {"midlayer.self_attn.q_proj.weight", "BF16", 2, {8, 16}},
        {"midlayer.self_attn.k_proj.weight", "BF16", 2, {4, 16}},
        {"midlayer.self_attn.v_proj.weight", "BF16", 2, {4, 16}},
        {"midlayer.self_attn.o_proj.weight", "BF16", 2, {8, 8}},
        {"midlayer.post_attention_layernorm.weight", "BF16", 1, {8, 0}},
        {"midlayer.mlp.gate_proj.weight", "BF16", 2, {16, 8}},
        {"midlayer.mlp.up_proj.weight", "BF16", 2, {16, 8}},
        {"midlayer.mlp.down_proj.weight", "BF16", 2, {8, 16}},
        {"norm.weight", "BF16", 1, {8, 0}},
        {"lm_head.weight", "BF16", 2, {6, 8}},
        {"d2t", "I64", 1, {6, 0}},
        {"t2d", "BOOL", 1, {20, 0}},
        {"midlayer.extra.weight", "BF16", 1, {8, 0}}, /* only if with_extra */
    };
    size_t n = sizeof(t) / sizeof(t[0]);
    if (!with_extra) n -= 1;
    if (drop_norm) {
        /* remove "norm.weight" (index 11) by shifting */
        for (size_t i = 11; i + 1 < n; i++) t[i] = t[i + 1];
        n -= 1;
    }
    snprintf(p, sizeof(p), "%s/model.safetensors", dir);
    wsafetensors(p, t, n, 6);
}

static const char *MINI_CFG =
    "{\"architectures\":[\"LlamaForCausalLMEagle3\"],\"model_type\":\"llama\","
    "\"hidden_size\":8,\"draft_vocab_size\":6,\"vocab_size\":20,"
    "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":2,"
    "\"intermediate_size\":16,\"num_hidden_layers\":1,\"rms_norm_eps\":1e-5,"
    "\"rope_parameters\":{\"rope_theta\":5000000.0,\"rope_type\":\"default\"}}";

static int stub_embed(void *ctx, uint32_t token, float *out) {
    (void)ctx;
    for (int i = 0; i < 8; i++)
        out[i] = 0.01f * (float)((token + (uint32_t)i) % 7u) - 0.03f;
    return 1;
}

static int selftest(void) {
    char base[] = "/tmp/h3_eagle3_XXXXXX";
    if (!mkdtemp(base)) die("mkdtemp");
    char dir[1200], err[256];

    /* 1. good miniature checkpoint loads, config is as declared. */
    snprintf(dir, sizeof(dir), "%s/ok", base);
    write_mini(dir, MINI_CFG, 0, 0);
    qwen_eagle3 *e = NULL;
    require(qwen_eagle3_load(dir, &e, err, sizeof(err)), err);
    const qwen_eagle3_config *c = qwen_eagle3_config_of(e);
    require(c->hidden_size == 8 && c->head_dim == 2, "cfg hidden/head_dim");
    require(c->num_attention_heads == 4 && c->num_key_value_heads == 2, "heads");
    require(c->q_dim == 8 && c->kv_dim == 4, "q_dim/kv_dim from head_dim");
    require(c->qkv_in_dim == 16 && c->fusion_in_dim == 24, "qkv/fusion widths");
    require(c->fusion_count == 3, "fusion_count = fusion_in / hidden");
    require(c->draft_vocab_size == 6 && c->target_vocab_size == 20, "vocab");
    require(!c->rope_is_mrope, "llama default rope must not be flagged mrope");
    printf("  loaded mini: hidden=%d head_dim=%d q_dim=%d qkv_in=%d fusion=%dx%d "
           "eps=%g theta=%.0f arch=%s\n",
           c->hidden_size, c->head_dim, c->q_dim, c->qkv_in_dim,
           c->fusion_count, c->hidden_size, (double)c->rms_norm_eps,
           c->rope_theta, c->architecture);

    /* 2. d2t maps draft -> target in range; t2d answers. */
    for (uint32_t d = 0; d < 6; d++) {
        uint32_t tgt = qwen_eagle3_d2t(e, d);
        require(tgt < 20, "d2t out of target range");
        require(tgt == 3u * d, "d2t delta convention (mini uses target=3*draft)");
    }

    /* 3. reference forward runs and yields finite logits. */
    float aux0[8] = {0}, aux1[8] = {0}, aux2[8] = {0};
    for (int i = 0; i < 8; i++) {
        aux0[i] = 0.02f * (float)(i - 4);
        aux1[i] = -0.01f * (float)i;
        aux2[i] = 0.005f;
    }
    const float *aux[3] = {aux0, aux1, aux2};
    float logits[6];
    require(qwen_eagle3_step_ref(e, aux, 3, 7, stub_embed, NULL, NULL, logits, err,
                                 sizeof(err)),
            err);
    for (int i = 0; i < 6; i++)
        require(isfinite(logits[i]), "non-finite draft logit");
    uint32_t am = 0;
    for (int i = 1; i < 6; i++) if (logits[i] > logits[am]) am = (uint32_t)i;
    printf("  forward ok: argmax(draft)=%u -> target %u (t2d_ok=%d)\n", am,
           qwen_eagle3_d2t(e, am), qwen_eagle3_t2d_ok(e, qwen_eagle3_d2t(e, am)));

    qwen_eagle3_free(e);

    /* 4. load -> free repeated (leak / double-free smoke). */
    for (int r = 0; r < 3; r++) {
        qwen_eagle3 *x = NULL;
        require(qwen_eagle3_load(dir, &x, err, sizeof(err)), err);
        qwen_eagle3_free(x);
    }

    /* 5. negatives: extra tensor, missing tensor, 2-layer config. */
    snprintf(dir, sizeof(dir), "%s/extra", base);
    write_mini(dir, MINI_CFG, 1, 0);
    require(!qwen_eagle3_load(dir, &e, err, sizeof(err)),
            "unknown tensor must fail the load");
    require(strstr(err, "unknown tensor") != NULL, "reason should name it");
    printf("  reject extra tensor: %s\n", err);

    snprintf(dir, sizeof(dir), "%s/missing", base);
    write_mini(dir, MINI_CFG, 0, 1);
    require(!qwen_eagle3_load(dir, &e, err, sizeof(err)),
            "missing required tensor must fail");
    require(strstr(err, "norm.weight") != NULL, "reason should name norm.weight");
    printf("  reject missing tensor: %s\n", err);

    snprintf(dir, sizeof(dir), "%s/twolayer", base);
    {
        char cfg2[1024];
        snprintf(cfg2, sizeof(cfg2),
                 "{\"architectures\":[\"LlamaForCausalLMEagle3\"],"
                 "\"model_type\":\"llama\",\"hidden_size\":8,"
                 "\"draft_vocab_size\":6,\"vocab_size\":20,"
                 "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
                 "\"head_dim\":2,\"intermediate_size\":16,"
                 "\"num_hidden_layers\":2,\"rms_norm_eps\":1e-5}");
        write_mini(dir, cfg2, 0, 0);
    }
    require(!qwen_eagle3_load(dir, &e, err, sizeof(err)),
            "a 2-layer config must be rejected");
    printf("  reject 2-layer config: %s\n", err);

    printf("ok: QINT-015h-1b eagle3 load + forward smoke (self-test)\n");
    return 0;
}

static int probe_real(const char *dir) {
    char err[256];
    qwen_eagle3 *e = NULL;
    if (!qwen_eagle3_load(dir, &e, err, sizeof(err))) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    const qwen_eagle3_config *c = qwen_eagle3_config_of(e);
    printf("loaded %s\n", dir);
    printf("  arch=%s model_type=%s\n", c->architecture, c->model_type);
    printf("  hidden=%d  head_dim=%d  heads q/kv=%d/%d  q_dim=%d kv_dim=%d\n",
           c->hidden_size, c->head_dim, c->num_attention_heads,
           c->num_key_value_heads, c->q_dim, c->kv_dim);
    printf("  qkv_in=%d (2*hidden=%d)  fusion=%dx%d=%d  intermediate=%d\n",
           c->qkv_in_dim, 2 * c->hidden_size, c->fusion_count, c->hidden_size,
           c->fusion_in_dim, c->intermediate_size);
    printf("  draft_vocab=%d  target_vocab=%d  rms_eps=%g  rope_theta=%.0f  "
           "mrope=%d\n",
           c->draft_vocab_size, c->target_vocab_size, (double)c->rms_norm_eps,
           c->rope_theta, c->rope_is_mrope);
    printf("  d2t[0..3] -> target %u %u %u %u   d2t[%d-1] -> %u\n",
           qwen_eagle3_d2t(e, 0), qwen_eagle3_d2t(e, 1), qwen_eagle3_d2t(e, 2),
           qwen_eagle3_d2t(e, 3), c->draft_vocab_size,
           qwen_eagle3_d2t(e, (uint32_t)c->draft_vocab_size - 1));
    /* every d2t target must be representable in the draft vocab (t2d). */
    int d2t_roundtrips = 1;
    for (uint32_t d = 0; d < (uint32_t)c->draft_vocab_size; d += 337u)
        if (!qwen_eagle3_t2d_ok(e, qwen_eagle3_d2t(e, d))) d2t_roundtrips = 0;
    printf("  d2t/t2d consistency (sampled): %s\n",
           d2t_roundtrips ? "OK" : "MISMATCH");

    /* forward smoke: zero aux hidden, stub embedding -> finite logits. */
    int H = c->hidden_size;
    float *a0 = calloc((size_t)H, sizeof(float));
    float *a1 = calloc((size_t)H, sizeof(float));
    float *a2 = calloc((size_t)H, sizeof(float));
    float *logits = malloc((size_t)c->draft_vocab_size * sizeof(float));
    const float *aux[8] = {a0, a1, a2, a0, a0, a0, a0, a0};
    if (!qwen_eagle3_step_ref(e, aux, 100u, 0, stub_embed, NULL, NULL, logits, err,
                              sizeof(err))) {
        fprintf(stderr, "forward failed: %s\n", err);
        return 1;
    }
    int finite = 1;
    uint32_t am = 0;
    for (int i = 0; i < c->draft_vocab_size; i++) {
        if (!isfinite(logits[i])) finite = 0;
        if (logits[i] > logits[am]) am = (uint32_t)i;
    }
    printf("  forward: logits finite=%d  argmax=%u -> target %u  t2d_ok=%d\n",
           finite, am, qwen_eagle3_d2t(e, am),
           qwen_eagle3_t2d_ok(e, qwen_eagle3_d2t(e, am)));
    free(a0); free(a1); free(a2); free(logits);
    qwen_eagle3_free(e);
    return finite ? 0 : 1;
}

/* ---- QINT-015h-1c: deterministic fixture + staged C trace ------------- */

/* splitmix64 -> uniform in [-1, 1). */
static double sm64_next(uint64_t *s) {
    *s += 0x9E3779B97F4A7C15ull;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return ((double)(z >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

/* A self-contained parity fixture: 3 aux hidden rows + the token's target
 * embedding + token id + a non-zero position. The embedding is baked in so a
 * Python reference never has to load the 32B target. */
static void gen_fixture(const char *out, int hidden, uint32_t token,
                        int position) {
    FILE *f = fopen(out, "wb");
    if (!f) die("cannot write fixture");
    uint64_t s = 0x9E3779B97F4A7C15ull ^ ((uint64_t)token << 1) ^
                 ((uint64_t)position << 17);
    fprintf(f, "{\n  \"hidden_size\": %d,\n  \"token_id\": %u,\n"
               "  \"position\": %d,\n",
            hidden, token, position);
    const char *labels[3] = {"aux_hidden_low", "aux_hidden_mid",
                             "aux_hidden_high"};
    for (int a = 0; a < 3; a++) {
        fprintf(f, "  \"%s\": [", labels[a]);
        for (int i = 0; i < hidden; i++)
            fprintf(f, "%s%.9g", i ? "," : "", sm64_next(&s));
        fprintf(f, "],\n");
    }
    fprintf(f, "  \"embedding\": [");
    for (int i = 0; i < hidden; i++)
        fprintf(f, "%s%.9g", i ? "," : "", sm64_next(&s));
    fprintf(f, "]\n}\n");
    fclose(f);
    printf("wrote fixture %s (hidden=%d token=%u position=%d)\n", out, hidden,
           token, position);
}

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *b = malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[got] = 0;
    if (n) *n = got;
    return b;
}

static float *json_vec(const h3_json *o, const char *key, int want_n) {
    const h3_json *a = h3_json_object_get(o, key);
    if (!a || !h3_json_is(a, H3_JSON_ARRAY) ||
        (int)h3_json_array_size(a) != want_n)
        return NULL;
    float *v = malloc((size_t)want_n * sizeof(float));
    if (!v) return NULL;
    for (int i = 0; i < want_n; i++)
        v[i] = (float)h3_json_number_or(h3_json_array_at(a, (size_t)i), 0.0);
    return v;
}

typedef struct { const float *emb; int n; } fixed_embed;
static int fixed_embed_fn(void *ctx, uint32_t token, float *out) {
    (void)token;
    fixed_embed *fe = ctx;
    memcpy(out, fe->emb, (size_t)fe->n * sizeof(float));
    return 1;
}

static void wvec(FILE *f, const char *name, const float *v, int n, int first) {
    fprintf(f, "%s\n  \"%s\": [", first ? "" : ",", name);
    for (int i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", (double)v[i]);
    fprintf(f, "]");
}

static int dump_trace(const char *ckpt, const char *fixpath, const char *out) {
    char err[256];
    qwen_eagle3 *e = NULL;
    if (!qwen_eagle3_load(ckpt, &e, err, sizeof(err))) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    const qwen_eagle3_config *c = qwen_eagle3_config_of(e);
    int H = c->hidden_size;

    size_t fn = 0;
    char *ftext = slurp(fixpath, &fn);
    if (!ftext) { fprintf(stderr, "cannot read fixture %s\n", fixpath); return 1; }
    char jerr[256] = {0};
    h3_json *fx = h3_json_parse(ftext, fn, jerr, sizeof(jerr));
    free(ftext);
    if (!fx) { fprintf(stderr, "fixture json: %s\n", jerr); return 1; }
    int fh = (int)h3_json_number_or(h3_json_object_get(fx, "hidden_size"), 0);
    if (fh != H) {
        fprintf(stderr, "fixture hidden_size %d != checkpoint %d\n", fh, H);
        return 1;
    }
    uint32_t token =
        (uint32_t)h3_json_number_or(h3_json_object_get(fx, "token_id"), 0);
    int position =
        (int)h3_json_number_or(h3_json_object_get(fx, "position"), 0);
    float *lo = json_vec(fx, "aux_hidden_low", H);
    float *mi = json_vec(fx, "aux_hidden_mid", H);
    float *hi = json_vec(fx, "aux_hidden_high", H);
    float *emb = json_vec(fx, "embedding", H);
    if (!lo || !mi || !hi || !emb) {
        fprintf(stderr, "fixture is missing an array or it is the wrong length\n");
        return 1;
    }
    const float *aux[3] = {lo, mi, hi};

    qwen_eagle3_trace tr;
    if (!qwen_eagle3_trace_alloc(e, &tr)) { fprintf(stderr, "trace alloc\n"); return 1; }
    float *logits = malloc((size_t)c->draft_vocab_size * sizeof(float));
    fixed_embed fe = {emb, H};
    if (!qwen_eagle3_step_ref(e, aux, token, position, fixed_embed_fn, &fe, &tr,
                              logits, err, sizeof(err))) {
        fprintf(stderr, "forward: %s\n", err);
        return 1;
    }

    /* draft top-5 + d2t target */
    int dv = c->draft_vocab_size;
    int top[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < dv; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > logits[top[k]]) {
                for (int j = 4; j > k; j--) top[j] = top[j - 1];
                top[k] = i;
                break;
            }
        }
    }

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "{\n  \"source\": \"c-reference\",\n  \"token_id\": %u,\n"
               "  \"position\": %d,\n  \"hidden_size\": %d,\n"
               "  \"draft_vocab_size\": %d",
            token, position, H, dv);
    wvec(f, "aux_concat", tr.aux_concat, c->fusion_in_dim, 0);
    wvec(f, "fc_out", tr.fc_out, H, 0);
    wvec(f, "embed_norm", tr.embed_norm, H, 0);
    wvec(f, "hidden_normed", tr.hidden_normed, H, 0);
    wvec(f, "qkv_in", tr.qkv_in, c->qkv_in_dim, 0);
    wvec(f, "q_pre_rope", tr.q_pre_rope, c->q_dim, 0);
    wvec(f, "k_pre_rope", tr.k_pre_rope, c->kv_dim, 0);
    wvec(f, "v", tr.v, c->kv_dim, 0);
    wvec(f, "q_post_rope", tr.q_post_rope, c->q_dim, 0);
    wvec(f, "k_post_rope", tr.k_post_rope, c->kv_dim, 0);
    wvec(f, "attn_heads", tr.attn_heads, c->q_dim, 0);
    wvec(f, "attn_out", tr.attn_out, H, 0);
    wvec(f, "post_attn_norm", tr.post_attn_norm, H, 0);
    wvec(f, "mlp_out", tr.mlp_out, H, 0);
    wvec(f, "final_hidden", tr.final_hidden, H, 0);
    wvec(f, "draft_logits", logits, dv, 0);
    fprintf(f, ",\n  \"draft_top1\": %d,\n  \"draft_top5\": [%d,%d,%d,%d,%d],\n",
            top[0], top[0], top[1], top[2], top[3], top[4]);
    fprintf(f, "  \"target_top1\": %u\n}\n", qwen_eagle3_d2t(e, (uint32_t)top[0]));
    fclose(f);
    printf("wrote C trace %s  (draft top1=%d -> target %u)\n", out, top[0],
           qwen_eagle3_d2t(e, (uint32_t)top[0]));

    free(lo); free(mi); free(hi); free(emb); free(logits);
    qwen_eagle3_trace_free(&tr);
    h3_json_free(fx);
    qwen_eagle3_free(e);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc >= 3 && !strcmp(argv[1], "gen-fixture")) {
        int hidden = argc >= 4 ? atoi(argv[3]) : 5120;
        uint32_t token = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1234u;
        int position = argc >= 6 ? atoi(argv[5]) : 37;
        gen_fixture(argv[2], hidden, token, position);
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "dump"))
        return dump_trace(argv[2], argv[3], argv[4]);
    if (argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  %s --selftest\n"
                "  %s <checkpoint_dir>\n"
                "  %s gen-fixture <out.json> [hidden] [token] [position]\n"
                "  %s dump <checkpoint_dir> <fixture.json> <out_c_trace.json>\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    return probe_real(argv[1]);
}
