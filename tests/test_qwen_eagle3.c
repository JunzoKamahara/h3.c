/* QINT-015h-1b -- load + shape-validation + reference-forward smoke for the
 * EAGLE-3 draft head. NO coordinator, NO parity claim (that is 1c).
 *
 *   ./h3_qwen_eagle3_test --selftest        # model-free: miniature checkpoint
 *   ./h3_qwen_eagle3_test <checkpoint_dir>  # load a real checkpoint + forward
 */

#include "qwen_eagle3.h"

#include "h3_json.h"
#include "qwen_draft.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
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

static uint16_t mk_bf16(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof(b));
    b += 0x7fffu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

/* Records the token-id order it is asked for (QINT-015h-2b alignment check). */
static uint32_t g_rec_seen[8];
static int g_rec_n;
static int g_rec_hidden = 8;
static int rec_embed(void *ctx, uint32_t token, float *out) {
    (void)ctx;
    if (g_rec_n < 8) g_rec_seen[g_rec_n++] = token;
    for (int i = 0; i < g_rec_hidden; i++)
        out[i] = 0.003f * (float)((token + (uint32_t)i) % 5u) - 0.006f;
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

    /* 6. QINT-015h-2b: autoregressive draft chain + backend wiring. */
    snprintf(dir, sizeof(dir), "%s/ok", base);
    require(qwen_eagle3_load(dir, &e, err, sizeof(err)), err);
    {
        const qwen_eagle3_config *cc = qwen_eagle3_config_of(e);
        int Hm = cc->hidden_size;
        float a0[8], a1[8], a2[8];
        for (int i = 0; i < Hm; i++) {
            a0[i] = 0.02f * (float)(i - 4);
            a1[i] = -0.01f * (float)i;
            a2[i] = 0.005f;
        }
        const float *auxr[3] = {a0, a1, a2};

        g_rec_hidden = Hm;
        g_rec_n = 0;

        qwen_eagle3_kv *kv = NULL;
        require(qwen_eagle3_kv_new(e, &kv, err, sizeof(err)), err);
        uint32_t anchor = 3, draft[4], draft2[4];
        int start_pos = 11;
        require(qwen_eagle3_chain(e, kv, auxr, anchor, start_pos, 4, rec_embed,
                                  NULL, draft, err, sizeof(err)),
                err);
        /* first-step alignment: step 0 embeds the ANCHOR, then each step
         * embeds the previous draft token mapped through d2t. */
        require(g_rec_n == 4, "chain must call embed once per step");
        require(g_rec_seen[0] == anchor, "chain step 0 must embed the anchor token");
        require(g_rec_seen[1] == qwen_eagle3_d2t(e, draft[0]),
                "chain step 1 must embed d2t(previous draft argmax)");
        for (int j = 0; j < 4; j++)
            require(draft[j] < (uint32_t)cc->draft_vocab_size, "draft id in range");
        printf("  chain draft ids = %u %u %u %u  -> targets %u %u %u %u\n",
               draft[0], draft[1], draft[2], draft[3],
               qwen_eagle3_d2t(e, draft[0]), qwen_eagle3_d2t(e, draft[1]),
               qwen_eagle3_d2t(e, draft[2]), qwen_eagle3_d2t(e, draft[3]));

        /* deterministic: same inputs -> same chain (reset the KV first). */
        g_rec_n = 0;
        qwen_eagle3_kv_reset(kv);
        require(qwen_eagle3_chain(e, kv, auxr, anchor, start_pos, 4, rec_embed,
                                  NULL, draft2, err, sizeof(err)),
                err);
        require(memcmp(draft, draft2, sizeof(draft)) == 0, "chain not deterministic");
        qwen_eagle3_kv_free(kv);
        qwen_eagle3_free(e);

        /* backend vtable: propose fills target-vocab tokens; no aux -> count 0. */
        char berr[256] = {0};
        qwen_draft_backend *b =
            qwen_draft_eagle_new(dir, rec_embed, NULL, berr, sizeof(berr));
        require(b != NULL, berr);
        uint16_t auxb[3][8];
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < Hm; i++) auxb[a][i] = mk_bf16(auxr[a][i]);
        qwen_draft_context ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.have_anchor = 1;
        ctx.anchor_token = anchor;
        ctx.history_length = 11;
        ctx.n_aux = 3;
        ctx.hidden_size = (size_t)Hm;
        for (int a = 0; a < 3; a++) ctx.aux_hidden[a] = auxb[a];
        qwen_draft_proposal pr;
        require(qwen_draft_propose(b, &ctx, 4, &pr), "propose failed");
        require(pr.count == 4, "eagle backend should propose 4 tokens");
        for (size_t j = 0; j < pr.count; j++)
            require(pr.tokens[j] < 20u, "proposed target token in mini vocab");
        printf("  backend proposal (target vocab) = %u %u %u %u\n", pr.tokens[0],
               pr.tokens[1], pr.tokens[2], pr.tokens[3]);

        ctx.n_aux = 0; /* capture off -> scalar fallback */
        require(qwen_draft_propose(b, &ctx, 4, &pr), "propose failed");
        require(pr.count == 0, "no aux capture -> backend must defer to scalar");
        qwen_draft_destroy(b);
    }

    printf("ok: QINT-015h-1b/2a/2b eagle3 load + forward + chain (self-test)\n");
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

/* A self-contained parity fixture: for each of `T` tokens, 3 aux hidden rows +
 * the token's target embedding + token id + a non-zero position. All baked in
 * so a Python reference never loads the 32B target. Arrays are flat
 * [T*hidden]; positions are 37, 38, ... The T=1 case is the QINT-015h-1c
 * fixture; T>=2 exercises real causal attention (QINT-015h-2a). */
static void gen_fixture(const char *out, int hidden, uint32_t token0,
                        int position0, int T) {
    if (T < 1) T = 1;
    FILE *f = fopen(out, "wb");
    if (!f) die("cannot write fixture");
    uint64_t s = 0x9E3779B97F4A7C15ull ^ ((uint64_t)token0 << 1) ^
                 ((uint64_t)position0 << 17) ^ ((uint64_t)T << 33);
    fprintf(f, "{\n  \"hidden_size\": %d,\n  \"num_tokens\": %d,\n", hidden, T);
    fprintf(f, "  \"token_ids\": [");
    for (int t = 0; t < T; t++) fprintf(f, "%s%u", t ? "," : "", token0 + (uint32_t)t);
    fprintf(f, "],\n  \"positions\": [");
    for (int t = 0; t < T; t++) fprintf(f, "%s%d", t ? "," : "", position0 + t);
    fprintf(f, "],\n");
    const char *labels[3] = {"aux_hidden_low", "aux_hidden_mid",
                             "aux_hidden_high"};
    for (int a = 0; a < 3; a++) {
        fprintf(f, "  \"%s\": [", labels[a]);
        for (int i = 0; i < T * hidden; i++)
            fprintf(f, "%s%.9g", i ? "," : "", sm64_next(&s));
        fprintf(f, "],\n");
    }
    fprintf(f, "  \"embedding\": [");
    for (int i = 0; i < T * hidden; i++)
        fprintf(f, "%s%.9g", i ? "," : "", sm64_next(&s));
    fprintf(f, "]\n}\n");
    fclose(f);
    printf("wrote fixture %s (hidden=%d num_tokens=%d token0=%u position0=%d)\n",
           out, hidden, T, token0, position0);
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

/* Fixture embedding table: token -> its baked-in row (matched by first index). */
typedef struct {
    const uint32_t *tokens;
    const float *rows; /* [T*H] */
    int T, H;
} fix_embed;
static int fix_embed_fn(void *ctx, uint32_t token, float *out) {
    fix_embed *fe = ctx;
    for (int i = 0; i < fe->T; i++)
        if (fe->tokens[i] == token) {
            memcpy(out, fe->rows + (size_t)i * fe->H, (size_t)fe->H * sizeof(float));
            return 1;
        }
    return 0;
}

static void wvec(FILE *f, const char *name, const float *v, int n) {
    fprintf(f, ",\n  \"%s\": [", name);
    for (int i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", (double)v[i]);
    fprintf(f, "]");
}

static void top5_of(const float *logits, int dv, int *top) {
    for (int k = 0; k < 5; k++) top[k] = 0;
    for (int i = 0; i < dv; i++)
        for (int k = 0; k < 5; k++)
            if (logits[i] > logits[top[k]]) {
                for (int j = 4; j > k; j--) top[j] = top[j - 1];
                top[k] = i;
                break;
            }
}

static double cosf_vec(const float *a, const float *b, int n) {
    double da = 0, db = 0, dp = 0;
    for (int i = 0; i < n; i++) { da += (double)a[i]*a[i]; db += (double)b[i]*b[i]; dp += (double)a[i]*b[i]; }
    if (da == 0 || db == 0) return da == db ? 1.0 : 0.0;
    return dp / (sqrt(da) * sqrt(db));
}

static int dump_trace(const char *ckpt, const char *fixpath, const char *out) {
    char err[256];
    qwen_eagle3 *e = NULL;
    if (!qwen_eagle3_load(ckpt, &e, err, sizeof(err))) {
        fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    const qwen_eagle3_config *c = qwen_eagle3_config_of(e);
    int H = c->hidden_size, dv = c->draft_vocab_size;

    size_t fn = 0;
    char *ftext = slurp(fixpath, &fn);
    if (!ftext) { fprintf(stderr, "cannot read fixture %s\n", fixpath); return 1; }
    char jerr[256] = {0};
    h3_json *fx = h3_json_parse(ftext, fn, jerr, sizeof(jerr));
    free(ftext);
    if (!fx) { fprintf(stderr, "fixture json: %s\n", jerr); return 1; }
    if ((int)h3_json_number_or(h3_json_object_get(fx, "hidden_size"), 0) != H) {
        fprintf(stderr, "fixture hidden_size != checkpoint %d\n", H);
        return 1;
    }
    int T = (int)h3_json_number_or(h3_json_object_get(fx, "num_tokens"), 1);
    if (T < 1) T = 1;
    const h3_json *jt = h3_json_object_get(fx, "token_ids");
    const h3_json *jp = h3_json_object_get(fx, "positions");
    if (!jt || (int)h3_json_array_size(jt) != T || !jp ||
        (int)h3_json_array_size(jp) != T) {
        fprintf(stderr, "fixture token_ids/positions must be length %d\n", T);
        return 1;
    }
    uint32_t *tokens = malloc((size_t)T * sizeof(uint32_t));
    int *positions = malloc((size_t)T * sizeof(int));
    for (int i = 0; i < T; i++) {
        tokens[i] = (uint32_t)h3_json_number_or(h3_json_array_at(jt, (size_t)i), 0);
        positions[i] = (int)h3_json_number_or(h3_json_array_at(jp, (size_t)i), 0);
    }
    float *lo = json_vec(fx, "aux_hidden_low", T * H);
    float *mi = json_vec(fx, "aux_hidden_mid", T * H);
    float *hi = json_vec(fx, "aux_hidden_high", T * H);
    float *emb = json_vec(fx, "embedding", T * H);
    if (!lo || !mi || !hi || !emb) {
        fprintf(stderr, "fixture array missing or wrong length (expect %d)\n", T * H);
        return 1;
    }
    /* token-major aux pointers: aux[i*3 + f] */
    const float **aux = malloc((size_t)T * 3 * sizeof(*aux));
    for (int i = 0; i < T; i++) {
        aux[i * 3 + 0] = lo + (size_t)i * H;
        aux[i * 3 + 1] = mi + (size_t)i * H;
        aux[i * 3 + 2] = hi + (size_t)i * H;
    }
    fix_embed fe = {tokens, emb, T, H};

    qwen_eagle3_trace *tr = calloc((size_t)T, sizeof(*tr));
    for (int i = 0; i < T; i++)
        if (!qwen_eagle3_trace_alloc(e, &tr[i])) { fprintf(stderr, "trace alloc\n"); return 1; }
    float *batch_logits = malloc((size_t)T * dv * sizeof(float));
    if (!qwen_eagle3_forward_seq(e, T, aux, tokens, positions, fix_embed_fn, &fe,
                                 tr, batch_logits, err, sizeof(err))) {
        fprintf(stderr, "forward_seq: %s\n", err);
        return 1;
    }

    /* step-wise KV path -- must match the batch path per token. */
    qwen_eagle3_kv *kv = NULL;
    if (!qwen_eagle3_kv_new(e, &kv, err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
    float *kv_logits = malloc((size_t)T * dv * sizeof(float));
    double worst_cos = 1.0;
    for (int i = 0; i < T; i++) {
        if (!qwen_eagle3_kv_step(kv, aux + (size_t)i * 3, tokens[i], positions[i],
                                 fix_embed_fn, &fe, NULL,
                                 kv_logits + (size_t)i * dv, err, sizeof(err))) {
            fprintf(stderr, "kv_step %d: %s\n", i, err);
            return 1;
        }
        double co = cosf_vec(batch_logits + (size_t)i * dv,
                             kv_logits + (size_t)i * dv, dv);
        if (co < worst_cos) worst_cos = co;
    }
    qwen_eagle3_kv_free(kv);
    printf("step-wise KV vs batch: worst per-token logit cosine = %.10f  %s\n",
           worst_cos, worst_cos >= 0.9999999 ? "OK" : "MISMATCH");
    if (worst_cos < 0.9999999) {
        fprintf(stderr, "FAIL: KV-step and forward_seq disagree\n");
        return 1;
    }

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "{\n  \"source\": \"c-reference\",\n  \"num_tokens\": %d,\n"
               "  \"hidden_size\": %d,\n  \"draft_vocab_size\": %d,\n"
               "  \"positions\": [",
            T, H, dv);
    for (int i = 0; i < T; i++) fprintf(f, "%s%d", i ? "," : "", positions[i]);
    fprintf(f, "]");
    for (int i = 0; i < T; i++) {
        char p[24];
        const qwen_eagle3_trace *x = &tr[i];
        const float *L = batch_logits + (size_t)i * dv;
        int top[5];
        top5_of(L, dv, top);
#define W(field, arr, n) do { snprintf(p, sizeof(p), "t%d_%s", i, field); wvec(f, p, (arr), (n)); } while (0)
        W("aux_concat", x->aux_concat, c->fusion_in_dim);
        W("fc_out", x->fc_out, H);
        W("embed_norm", x->embed_norm, H);
        W("hidden_normed", x->hidden_normed, H);
        W("qkv_in", x->qkv_in, c->qkv_in_dim);
        W("q_pre_rope", x->q_pre_rope, c->q_dim);
        W("k_pre_rope", x->k_pre_rope, c->kv_dim);
        W("v", x->v, c->kv_dim);
        W("q_post_rope", x->q_post_rope, c->q_dim);
        W("k_post_rope", x->k_post_rope, c->kv_dim);
        W("attn_heads", x->attn_heads, c->q_dim);
        W("attn_out", x->attn_out, H);
        W("post_attn_norm", x->post_attn_norm, H);
        W("mlp_out", x->mlp_out, H);
        W("final_hidden", x->final_hidden, H);
        W("draft_logits", L, dv);
#undef W
        fprintf(f, ",\n  \"t%d_draft_top1\": %d,\n  \"t%d_draft_top5\": "
                   "[%d,%d,%d,%d,%d],\n  \"t%d_target_top1\": %u",
                i, top[0], i, top[0], top[1], top[2], top[3], top[4], i,
                qwen_eagle3_d2t(e, (uint32_t)top[0]));
    }
    fprintf(f, "\n}\n");
    fclose(f);
    printf("wrote C trace %s  (T=%d)\n", out, T);

    for (int i = 0; i < T; i++) qwen_eagle3_trace_free(&tr[i]);
    free(tr); free(batch_logits); free(kv_logits); free(aux);
    free(lo); free(mi); free(hi); free(emb); free(tokens); free(positions);
    h3_json_free(fx);
    qwen_eagle3_free(e);
    return 0;
}

/* deterministic H-wide embedding stand-in for the `chain` smoke. */
static int hash_embed(void *ctx, uint32_t token, float *out) {
    int H = *(int *)ctx;
    for (int i = 0; i < H; i++)
        out[i] = 0.02f * sinf(0.7f * (float)token + 0.013f * (float)i);
    return 1;
}

/* QINT-015h-2b-0 smoke: run one draft chain on the real EAGLE weights with the
 * fixture's token-0 aux hidden as the frontier. Prints the k draft tokens and
 * the wall time (CPU reference -- 015i decides whether that needs Metal). */
static int run_chain_real(const char *ckpt, const char *fixpath) {
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
    if (!ftext) { fprintf(stderr, "cannot read %s\n", fixpath); return 1; }
    h3_json *fx = h3_json_parse(ftext, fn, err, sizeof(err));
    free(ftext);
    if (!fx) { fprintf(stderr, "fixture: %s\n", err); return 1; }
    float *lo = json_vec(fx, "aux_hidden_low", H);
    float *mi = json_vec(fx, "aux_hidden_mid", H);
    float *hi = json_vec(fx, "aux_hidden_high", H);
    const h3_json *jt = h3_json_object_get(fx, "token_ids");
    const h3_json *jp = h3_json_object_get(fx, "positions");
    uint32_t anchor = jt ? (uint32_t)h3_json_number_or(h3_json_array_at(jt, 0), 7) : 7u;
    int pos = jp ? (int)h3_json_number_or(h3_json_array_at(jp, 0), 37) : 37;
    if (!lo || !mi || !hi) { fprintf(stderr, "fixture aux arrays missing\n"); return 1; }
    const float *aux[3] = {lo, mi, hi};

    qwen_eagle3_kv *kv = NULL;
    require(qwen_eagle3_kv_new(e, &kv, err, sizeof(err)), err);
    int k = 4;
    uint32_t d1[4], d2[4];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    require(qwen_eagle3_chain(e, kv, aux, anchor, pos, k, hash_embed, &H, d1,
                              err, sizeof(err)), err);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 +
                (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    qwen_eagle3_kv_reset(kv);
    require(qwen_eagle3_chain(e, kv, aux, anchor, pos, k, hash_embed, &H, d2,
                              err, sizeof(err)), err);
    require(memcmp(d1, d2, sizeof(d1)) == 0, "chain not deterministic");

    printf("chain: anchor=%u pos=%d  k=%d\n", anchor, pos, k);
    printf("  draft (draft-vocab) = %u %u %u %u\n", d1[0], d1[1], d1[2], d1[3]);
    printf("  draft (target-vocab)= %u %u %u %u\n", qwen_eagle3_d2t(e, d1[0]),
           qwen_eagle3_d2t(e, d1[1]), qwen_eagle3_d2t(e, d1[2]),
           qwen_eagle3_d2t(e, d1[3]));
    printf("  T_draft (CPU reference, %d steps) = %.1f ms  (%.1f ms/step)\n", k,
           ms, ms / k);
    free(lo); free(mi); free(hi);
    h3_json_free(fx);
    qwen_eagle3_kv_free(kv);
    qwen_eagle3_free(e);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc >= 4 && !strcmp(argv[1], "chain"))
        return run_chain_real(argv[2], argv[3]);
    if (argc >= 3 && !strcmp(argv[1], "gen-fixture")) {
        int hidden = argc >= 4 ? atoi(argv[3]) : 5120;
        uint32_t token = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1234u;
        int position = argc >= 6 ? atoi(argv[5]) : 37;
        int ntok = argc >= 7 ? atoi(argv[6]) : 1;
        gen_fixture(argv[2], hidden, token, position, ntok);
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "dump"))
        return dump_trace(argv[2], argv[3], argv[4]);
    if (argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  %s --selftest\n"
                "  %s <checkpoint_dir>\n"
                "  %s gen-fixture <out.json> [hidden] [token] [position] [num_tokens]\n"
                "  %s dump <checkpoint_dir> <fixture.json> <out_c_trace.json>\n"
                "  %s chain <checkpoint_dir> <fixture.json>\n",
                argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    return probe_real(argv[1]);
}
