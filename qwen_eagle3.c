#include "qwen_eagle3.h"

#include "h3_json.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- *
 * QINT-015h-1b: load the EAGLE-3 draft head, validate every tensor, expose a
 * CPU reference forward. No GPU, no KV cache, no coordinator. Weights are
 * converted bf16 -> f32 at load so the reference forward can use Accelerate's
 * cblas_sgemv; ~3.3 GB resident for this checkpoint.
 * ------------------------------------------------------------------------- */

static void set_err(char *e, size_t n, const char *fmt, ...) {
    if (!e || !n) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e, n, fmt, ap);
    va_end(ap);
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* one 2-D bf16 weight [rows, cols], stored row-major as f32 */
typedef struct {
    int rows, cols;
    float *v; /* rows*cols, owned */
} w2d;

struct qwen_eagle3 {
    qwen_eagle3_config cfg;

    w2d fc;        /* [hidden, fusion_in]      */
    w2d q, k, v, o;/* attention projections    */
    w2d gate, up, down;
    w2d lm_head;   /* [draft_vocab, hidden]    */

    float *in_ln;   /* [hidden] */
    float *hidden_norm;
    float *post_ln;
    float *final_norm;

    int64_t *d2t;   /* [draft_vocab] */
    uint8_t *t2d;   /* [target_vocab] */
};

/* ---- config ------------------------------------------------------------- */

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n < 0) { fclose(f); return NULL; }
    char *b = malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = 0;
    if (len) *len = got;
    return b;
}

static int cfg_i(const h3_json *o, const char *key, int fb) {
    const h3_json *v = h3_json_object_get(o, key);
    return (v && h3_json_is(v, H3_JSON_NUMBER)) ? (int)h3_json_number_or(v, fb)
                                                : fb;
}

static int parse_config(const char *dir, qwen_eagle3_config *c, char *err,
                        size_t errn) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) {
        set_err(err, errn, "cannot read %s", path);
        return 0;
    }
    char jerr[256] = {0};
    h3_json *root = h3_json_parse(text, len, jerr, sizeof(jerr));
    free(text);
    if (!root) {
        set_err(err, errn, "config.json: %s", jerr);
        return 0;
    }
    memset(c, 0, sizeof(*c));
    const h3_json *arch = h3_json_object_get(root, "architectures");
    if (arch && h3_json_array_size(arch) > 0) {
        const char *a = h3_json_string_value(h3_json_array_at(arch, 0));
        if (a) snprintf(c->architecture, sizeof(c->architecture), "%s", a);
    }
    const h3_json *mt = h3_json_object_get(root, "model_type");
    if (mt && h3_json_string_value(mt))
        snprintf(c->model_type, sizeof(c->model_type), "%s",
                 h3_json_string_value(mt));

    c->hidden_size = cfg_i(root, "hidden_size", 0);
    c->draft_vocab_size = cfg_i(root, "draft_vocab_size", 0);
    c->target_vocab_size = cfg_i(root, "vocab_size", 0);
    c->num_attention_heads = cfg_i(root, "num_attention_heads", 0);
    c->num_key_value_heads = cfg_i(root, "num_key_value_heads", 0);
    c->head_dim = cfg_i(root, "head_dim", 0); /* explicit -- never derived */
    c->intermediate_size = cfg_i(root, "intermediate_size", 0);
    c->num_hidden_layers = cfg_i(root, "num_hidden_layers", 0);

    const h3_json *eps = h3_json_object_get(root, "rms_norm_eps");
    c->rms_norm_eps =
        eps ? (float)h3_json_number_or(eps, 1e-5f) : 1e-5f;

    /* rope: "rope_parameters":{"rope_theta":..} or a flat "rope_theta". */
    const h3_json *rp = h3_json_object_get(root, "rope_parameters");
    const h3_json *rt = rp ? h3_json_object_get(rp, "rope_theta")
                           : h3_json_object_get(root, "rope_theta");
    c->rope_theta = rt ? h3_json_number_or(rt, 5e6) : 5e6;
    const h3_json *rtype = rp ? h3_json_object_get(rp, "rope_type") : NULL;
    const char *rts = rtype ? h3_json_string_value(rtype) : NULL;
    /* mRoPE only if the config actually asks for it. model_type "llama" with
     * rope_type "default" => plain 1-D rotary. */
    const h3_json *rs = h3_json_object_get(root, "rope_scaling");
    c->rope_is_mrope =
        (rs && h3_json_object_get(rs, "mrope_section") != NULL) ||
        (rts && strstr(rts, "mrope") != NULL);

    h3_json_free(root);

    if (c->hidden_size <= 0 || c->head_dim <= 0 ||
        c->num_attention_heads <= 0 || c->num_key_value_heads <= 0 ||
        c->draft_vocab_size <= 0 || c->target_vocab_size <= 0) {
        set_err(err, errn,
                "config.json missing a required field "
                "(hidden_size/head_dim/heads/vocab)");
        return 0;
    }
    if (c->num_hidden_layers != 1) {
        set_err(err, errn, "expected a 1-layer EAGLE-3 head, config says %d",
                c->num_hidden_layers);
        return 0;
    }
    c->q_dim = c->num_attention_heads * c->head_dim;
    c->kv_dim = c->num_key_value_heads * c->head_dim;
    return 1;
}

/* ---- tensor loading --------------------------------------------------- */

typedef struct {
    const char *name;
    h3_dtype dtype;
    int ndim;
    int64_t shape[2]; /* -1 = "accept whatever, record it" */
    int required;
} expect;

static const h3_st_tensor *find_tensor(const h3_st_header *h, const char *name,
                                       int *used, int count) {
    for (int i = 0; i < count; i++)
        if (!strcmp(h->tensors[i].name, name)) {
            used[i] = 1;
            return &h->tensors[i];
        }
    return NULL;
}

static int shape_ok(const h3_st_tensor *t, const int64_t *want, int ndim) {
    if (t->ndim != ndim) return 0;
    for (int i = 0; i < ndim; i++)
        if (want[i] >= 0 && (int64_t)t->shape[i] != want[i]) return 0;
    return 1;
}

/* read a bf16 [rows,cols] tensor into a freshly malloc'd f32 buffer */
static int load_w2d(const h3_st_header *h, const h3_st_tensor *t, w2d *out,
                    char *err, size_t errn) {
    uint64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    uint16_t *raw = malloc(n * sizeof(uint16_t));
    float *f = malloc(n * sizeof(float));
    if (!raw || !f) {
        free(raw);
        free(f);
        set_err(err, errn, "out of memory loading %s", t->name);
        return 0;
    }
    if (!h3_st_read_data(h, t, raw, n * sizeof(uint16_t), err, errn)) {
        free(raw);
        free(f);
        return 0;
    }
    for (uint64_t i = 0; i < n; i++) f[i] = bf16_to_f32(raw[i]);
    free(raw);
    out->rows = (int)t->shape[0];
    out->cols = t->ndim == 2 ? (int)t->shape[1] : 1;
    out->v = f;
    return 1;
}

static int load_vec(const h3_st_header *h, const h3_st_tensor *t, float **out,
                    char *err, size_t errn) {
    w2d w = {0};
    if (!load_w2d(h, t, &w, err, errn)) return 0;
    *out = w.v;
    return 1;
}

int qwen_eagle3_load(const char *dir, qwen_eagle3 **out, char *error,
                     size_t errn) {
    if (out) *out = NULL;
    if (!dir || !out) {
        set_err(error, errn, "qwen_eagle3_load: bad args");
        return 0;
    }
    qwen_eagle3 *e = calloc(1, sizeof(*e));
    if (!e) {
        set_err(error, errn, "out of memory");
        return 0;
    }
    if (!parse_config(dir, &e->cfg, error, errn)) {
        free(e);
        return 0;
    }
    qwen_eagle3_config *c = &e->cfg;

    char stpath[4096];
    snprintf(stpath, sizeof(stpath), "%s/model.safetensors", dir);
    h3_st_header h;
    if (!h3_st_read_header(stpath, &h, error, errn)) {
        free(e);
        return 0;
    }
    int tc = (int)h.tensor_count;
    int *used = calloc((size_t)(tc > 0 ? tc : 1), sizeof(int));
    if (!used) {
        h3_st_free_header(&h);
        free(e);
        set_err(error, errn, "out of memory");
        return 0;
    }

    int H = c->hidden_size, I = c->intermediate_size;
    int QD = c->q_dim, KVD = c->kv_dim, DV = c->draft_vocab_size,
        TV = c->target_vocab_size;

    /* q/k/v input width and the fusion width are read from the tensors, then
     * cross-checked -- never assumed. */
    const h3_st_tensor *tq =
        find_tensor(&h, "midlayer.self_attn.q_proj.weight", used, tc);
    const h3_st_tensor *tfc = find_tensor(&h, "fc.weight", used, tc);
    if (!tq || tq->ndim != 2 || !tfc || tfc->ndim != 2) {
        set_err(error, errn, "missing fc.weight or q_proj.weight");
        goto fail;
    }
    c->qkv_in_dim = (int)tq->shape[1];
    c->fusion_in_dim = (int)tfc->shape[1];
    c->fusion_count =
        (H > 0 && c->fusion_in_dim % H == 0) ? c->fusion_in_dim / H : 0;

    if ((int)tq->shape[0] != QD) {
        set_err(error, errn, "q_proj rows %d != num_heads*head_dim %d",
                (int)tq->shape[0], QD);
        goto fail;
    }
    if (c->qkv_in_dim != 2 * H) {
        set_err(error, errn,
                "q_proj input %d != 2*hidden %d (EAGLE-3 norms embedding and "
                "fused hidden, then concatenates)",
                c->qkv_in_dim, 2 * H);
        goto fail;
    }
    if (c->fusion_count == 0) {
        set_err(error, errn, "fc.weight input %d is not a multiple of hidden %d",
                c->fusion_in_dim, H);
        goto fail;
    }
    if ((int)tfc->shape[0] != H) {
        set_err(error, errn, "fc.weight rows %d != hidden %d",
                (int)tfc->shape[0], H);
        goto fail;
    }

    int QKV = c->qkv_in_dim, FIN = c->fusion_in_dim;
    struct {
        const char *name;
        int ndim;
        int64_t shp[2];
        void *dst2d;  /* w2d*  */
        void *dstvec; /* float** */
        h3_dtype dt;
    } tbl[] = {
        {"fc.weight", 2, {H, FIN}, &e->fc, NULL, H3_DTYPE_BF16},
        {"midlayer.self_attn.q_proj.weight", 2, {QD, QKV}, &e->q, NULL, H3_DTYPE_BF16},
        {"midlayer.self_attn.k_proj.weight", 2, {KVD, QKV}, &e->k, NULL, H3_DTYPE_BF16},
        {"midlayer.self_attn.v_proj.weight", 2, {KVD, QKV}, &e->v, NULL, H3_DTYPE_BF16},
        {"midlayer.self_attn.o_proj.weight", 2, {H, QD}, &e->o, NULL, H3_DTYPE_BF16},
        {"midlayer.mlp.gate_proj.weight", 2, {I, H}, &e->gate, NULL, H3_DTYPE_BF16},
        {"midlayer.mlp.up_proj.weight", 2, {I, H}, &e->up, NULL, H3_DTYPE_BF16},
        {"midlayer.mlp.down_proj.weight", 2, {H, I}, &e->down, NULL, H3_DTYPE_BF16},
        {"lm_head.weight", 2, {DV, H}, &e->lm_head, NULL, H3_DTYPE_BF16},
        {"midlayer.input_layernorm.weight", 1, {H, 0}, NULL, &e->in_ln, H3_DTYPE_BF16},
        {"midlayer.hidden_norm.weight", 1, {H, 0}, NULL, &e->hidden_norm, H3_DTYPE_BF16},
        {"midlayer.post_attention_layernorm.weight", 1, {H, 0}, NULL, &e->post_ln, H3_DTYPE_BF16},
        {"norm.weight", 1, {H, 0}, NULL, &e->final_norm, H3_DTYPE_BF16},
    };
    for (size_t i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++) {
        const h3_st_tensor *t = find_tensor(&h, tbl[i].name, used, tc);
        if (!t) {
            set_err(error, errn, "required tensor missing: %s", tbl[i].name);
            goto fail;
        }
        if (t->dtype != tbl[i].dt) {
            set_err(error, errn, "%s: dtype %s, expected %s", tbl[i].name,
                    h3_dtype_name(t->dtype), h3_dtype_name(tbl[i].dt));
            goto fail;
        }
        if (!shape_ok(t, tbl[i].shp, tbl[i].ndim)) {
            set_err(error, errn, "%s: unexpected shape", tbl[i].name);
            goto fail;
        }
        int ok = tbl[i].dst2d
                     ? load_w2d(&h, t, tbl[i].dst2d, error, errn)
                     : load_vec(&h, t, tbl[i].dstvec, error, errn);
        if (!ok) goto fail;
    }

    /* d2t / t2d -- integer / bool, not bf16 */
    {
        const h3_st_tensor *td = find_tensor(&h, "d2t", used, tc);
        const h3_st_tensor *tt = find_tensor(&h, "t2d", used, tc);
        if (!td || td->ndim != 1 || (int)td->shape[0] != DV) {
            set_err(error, errn, "d2t missing or not [%d]", DV);
            goto fail;
        }
        if (!tt || tt->ndim != 1 || (int)tt->shape[0] != TV) {
            set_err(error, errn, "t2d missing or not [%d]", TV);
            goto fail;
        }
        e->d2t = malloc((size_t)DV * sizeof(int64_t));
        e->t2d = malloc((size_t)TV);
        if (!e->d2t || !e->t2d) {
            set_err(error, errn, "out of memory (d2t/t2d)");
            goto fail;
        }
        if (td->dtype == H3_DTYPE_I64) {
            if (!h3_st_read_data(&h, td, e->d2t, (size_t)DV * sizeof(int64_t),
                                 error, errn))
                goto fail;
        } else if (td->dtype == H3_DTYPE_I32) {
            int32_t *tmp = malloc((size_t)DV * sizeof(int32_t));
            if (!tmp || !h3_st_read_data(&h, td, tmp,
                                         (size_t)DV * sizeof(int32_t), error,
                                         errn)) {
                free(tmp);
                goto fail;
            }
            for (int i = 0; i < DV; i++) e->d2t[i] = tmp[i];
            free(tmp);
        } else {
            set_err(error, errn, "d2t dtype %s unsupported",
                    h3_dtype_name(td->dtype));
            goto fail;
        }
        /* t2d is BOOL (1 byte) or an int mask. */
        size_t tsz = h3_dtype_size(tt->dtype);
        if (tsz == 1) {
            if (!h3_st_read_data(&h, tt, e->t2d, (size_t)TV, error, errn))
                goto fail;
        } else if (tsz == 4) {
            int32_t *tmp = malloc((size_t)TV * sizeof(int32_t));
            if (!tmp || !h3_st_read_data(&h, tt, tmp,
                                         (size_t)TV * sizeof(int32_t), error,
                                         errn)) {
                free(tmp);
                goto fail;
            }
            for (int i = 0; i < TV; i++) e->t2d[i] = tmp[i] ? 1 : 0;
            free(tmp);
        } else {
            set_err(error, errn, "t2d dtype %s unsupported",
                    h3_dtype_name(tt->dtype));
            goto fail;
        }
    }

    /* every tensor in the file must be one we expected */
    for (int i = 0; i < tc; i++) {
        if (!used[i]) {
            set_err(error, errn, "unknown tensor in checkpoint: %s",
                    h.tensors[i].name);
            goto fail;
        }
    }

    free(used);
    h3_st_free_header(&h);
    *out = e;
    return 1;

fail:
    free(used);
    h3_st_free_header(&h);
    qwen_eagle3_free(e);
    return 0;
}

void qwen_eagle3_free(qwen_eagle3 *e) {
    if (!e) return;
    free(e->fc.v);
    free(e->q.v);
    free(e->k.v);
    free(e->v.v);
    free(e->o.v);
    free(e->gate.v);
    free(e->up.v);
    free(e->down.v);
    free(e->lm_head.v);
    free(e->in_ln);
    free(e->hidden_norm);
    free(e->post_ln);
    free(e->final_norm);
    free(e->d2t);
    free(e->t2d);
    free(e);
}

const qwen_eagle3_config *qwen_eagle3_config_of(const qwen_eagle3 *e) {
    return e ? &e->cfg : NULL;
}

uint32_t qwen_eagle3_d2t(const qwen_eagle3 *e, uint32_t draft_id) {
    if (!e || draft_id >= (uint32_t)e->cfg.draft_vocab_size) return 0;
    /* d2t is a DELTA: target = draft + d2t[draft]. Confirmed against this
     * checkpoint -- { draft + d2t[draft] : all draft } equals exactly the set
     * of target ids whose t2d mask is set (the 32000 kept tokens). */
    int64_t t = (int64_t)draft_id + e->d2t[draft_id];
    if (t < 0) t = 0;
    if (t >= e->cfg.target_vocab_size) t = e->cfg.target_vocab_size - 1;
    return (uint32_t)t;
}

int qwen_eagle3_t2d_ok(const qwen_eagle3 *e, uint32_t target_id) {
    if (!e || target_id >= (uint32_t)e->cfg.target_vocab_size) return 0;
    return e->t2d[target_id] != 0;
}

/* ---- CPU reference forward ------------------------------------------------ */

static void rmsnorm(float *dst, const float *src, const float *w, int n,
                    float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i] * src[i];
    float inv = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) dst[i] = src[i] * inv * w[i];
}

/* dst[r] = sum_c W[r][c] * x[c].  Plain reference; the fast path is 015h-2. */
static void matvec(float *dst, const w2d *w, const float *x) {
    int rows = w->rows, cols = w->cols;
    for (int r = 0; r < rows; r++) {
        const float *wr = w->v + (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
        dst[r] = (float)acc;
    }
}

static void rope_inplace(float *vec, int n_heads, int head_dim, int position,
                         double theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *p = vec + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            double freq = pow(theta, -2.0 * i / head_dim);
            double ang = position * freq;
            float c = (float)cos(ang), s = (float)sin(ang);
            float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;
            p[i + half] = a * s + b * c;
        }
    }
}

int qwen_eagle3_step_ref(const qwen_eagle3 *e, const float *const *aux_hidden,
                         uint32_t prev_token, int position,
                         qwen_eagle3_embed_fn embed, void *embed_ctx,
                         float *out_draft_logits, char *error, size_t errn) {
    if (!e || !aux_hidden || !embed || !out_draft_logits) {
        set_err(error, errn, "qwen_eagle3_step_ref: bad args");
        return 0;
    }
    const qwen_eagle3_config *c = &e->cfg;
    int H = c->hidden_size, I = c->intermediate_size, FIN = c->fusion_in_dim;
    int QD = c->q_dim, KVD = c->kv_dim;
    float eps = e->cfg.rms_norm_eps;

    float *xcat = malloc((size_t)FIN * sizeof(float));
    float *emb = malloc((size_t)H * sizeof(float));
    float *fused = malloc((size_t)H * sizeof(float));
    float *xqkv = malloc((size_t)2 * H * sizeof(float));
    float *q = malloc((size_t)QD * sizeof(float));
    float *kk = malloc((size_t)KVD * sizeof(float));
    float *vv = malloc((size_t)KVD * sizeof(float));
    float *heads = malloc((size_t)QD * sizeof(float));
    float *attn = malloc((size_t)H * sizeof(float));
    float *res = malloc((size_t)H * sizeof(float));
    float *y = malloc((size_t)H * sizeof(float));
    float *g = malloc((size_t)I * sizeof(float));
    float *u = malloc((size_t)I * sizeof(float));
    float *mlp = malloc((size_t)H * sizeof(float));
    if (!xcat || !emb || !fused || !xqkv || !q || !kk || !vv || !heads ||
        !attn || !res || !y || !g || !u || !mlp) {
        set_err(error, errn, "out of memory in eagle step");
        goto done_err;
    }

    /* fusion: concat the fusion_count aux hidden rows, project. */
    for (int f = 0; f < c->fusion_count; f++)
        memcpy(xcat + (size_t)f * H, aux_hidden[f], (size_t)H * sizeof(float));
    matvec(fused, &e->fc, xcat);

    if (!embed(embed_ctx, prev_token, emb)) {
        set_err(error, errn, "target embedding accessor failed for token %u",
                prev_token);
        goto done_err;
    }

    /* EAGLE-3: concat( RMSNorm(embed, input_layernorm),
     *                  RMSNorm(fused, hidden_norm) ) -> q/k/v input. */
    rmsnorm(xqkv, emb, e->in_ln, H, eps);
    rmsnorm(xqkv + H, fused, e->hidden_norm, H, eps);

    matvec(q, &e->q, xqkv);
    matvec(kk, &e->k, xqkv);
    matvec(vv, &e->v, xqkv);

    rope_inplace(q, c->num_attention_heads, c->head_dim, position, c->rope_theta);
    rope_inplace(kk, c->num_key_value_heads, c->head_dim, position, c->rope_theta);

    /* single position -> softmax over one key = 1, so the attention output per
     * query head is that head's (grouped) value. Multi-token draft chains with
     * a real KV cache are QINT-015h-2. */
    int group = c->num_attention_heads / c->num_key_value_heads;
    for (int hh = 0; hh < c->num_attention_heads; hh++) {
        int kv = hh / group;
        memcpy(heads + (size_t)hh * c->head_dim, vv + (size_t)kv * c->head_dim,
               (size_t)c->head_dim * sizeof(float));
    }
    matvec(attn, &e->o, heads);

    for (int i = 0; i < H; i++) res[i] = fused[i] + attn[i]; /* residual = fused */

    rmsnorm(y, res, e->post_ln, H, eps);
    matvec(g, &e->gate, y);
    matvec(u, &e->up, y);
    for (int i = 0; i < I; i++) {
        float x = g[i];
        float silu = x / (1.0f + expf(-x));
        g[i] = silu * u[i];
    }
    matvec(mlp, &e->down, g);
    for (int i = 0; i < H; i++) res[i] += mlp[i];

    rmsnorm(y, res, e->final_norm, H, eps);
    matvec(out_draft_logits, &e->lm_head, y);

    free(xcat); free(emb); free(fused); free(xqkv); free(q); free(kk);
    free(vv); free(heads); free(attn); free(res); free(y); free(g); free(u);
    free(mlp);
    return 1;

done_err:
    free(xcat); free(emb); free(fused); free(xqkv); free(q); free(kk);
    free(vv); free(heads); free(attn); free(res); free(y); free(g); free(u);
    free(mlp);
    return 0;
}
