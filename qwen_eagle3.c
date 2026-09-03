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

void qwen_eagle3_default_aux_layers(int target_num_layers, int *out3) {
    if (!out3) return;
    int n = target_num_layers > 0 ? target_num_layers : 64;
    out3[0] = 1;
    out3[1] = n / 2;
    out3[2] = n - 4;
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

int qwen_eagle3_trace_alloc(const qwen_eagle3 *e, qwen_eagle3_trace *t) {
    if (!e || !t) return 0;
    const qwen_eagle3_config *c = &e->cfg;
    memset(t, 0, sizeof(*t));
    int H = c->hidden_size, QD = c->q_dim, KVD = c->kv_dim;
    struct { float **p; int n; } f[] = {
        {&t->aux_concat, c->fusion_in_dim}, {&t->fc_out, H},
        {&t->embed_norm, H}, {&t->hidden_normed, H}, {&t->qkv_in, c->qkv_in_dim},
        {&t->q_pre_rope, QD}, {&t->k_pre_rope, KVD}, {&t->v, KVD},
        {&t->q_post_rope, QD}, {&t->k_post_rope, KVD}, {&t->attn_heads, QD},
        {&t->attn_out, H}, {&t->post_attn_norm, H}, {&t->mlp_out, H},
        {&t->final_hidden, H},
    };
    for (size_t i = 0; i < sizeof(f) / sizeof(f[0]); i++) {
        *f[i].p = calloc((size_t)f[i].n, sizeof(float));
        if (!*f[i].p) {
            qwen_eagle3_trace_free(t);
            return 0;
        }
    }
    return 1;
}

void qwen_eagle3_trace_free(qwen_eagle3_trace *t) {
    if (!t) return;
    float **p = (float **)t;
    for (size_t i = 0; i < sizeof(*t) / sizeof(float *); i++) free(p[i]);
    memset(t, 0, sizeof(*t));
}

/* GQA causal attention for one query token. q head h attends kv head (h/group)
 * over key/value rows 0..n_keys-1 (the causal prefix -- rows are appended in
 * position order). scale = 1/sqrt(head_dim). */
static void attn_gqa_causal(const qwen_eagle3_config *c, const float *q_rope,
                            const float *k_rows, const float *v_rows, int n_keys,
                            float *out_heads) {
    int hd = c->head_dim, kvd = c->kv_dim;
    int group = c->num_attention_heads / c->num_key_value_heads;
    float scale = (float)(1.0 / sqrt((double)hd));
    float *sc = malloc((size_t)(n_keys > 0 ? n_keys : 1) * sizeof(float));
    for (int h = 0; h < c->num_attention_heads; h++) {
        const float *qh = q_rope + (size_t)h * hd;
        int kv = h / group;
        float mx = -1e30f;
        for (int j = 0; j < n_keys; j++) {
            const float *kj = k_rows + (size_t)j * kvd + (size_t)kv * hd;
            double dot = 0.0;
            for (int d = 0; d < hd; d++) dot += (double)qh[d] * kj[d];
            sc[j] = (float)(dot * scale);
            if (sc[j] > mx) mx = sc[j];
        }
        double den = 0.0;
        for (int j = 0; j < n_keys; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
        float *oh = out_heads + (size_t)h * hd;
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;
        for (int j = 0; j < n_keys; j++) {
            float w = (float)((double)sc[j] / den);
            const float *vj = v_rows + (size_t)j * kvd + (size_t)kv * hd;
            for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
        }
    }
    free(sc);
}

/* Growable K/V cache for the single EAGLE decoder layer. */
struct qwen_eagle3_kv {
    const qwen_eagle3 *e;
    float *k_rows; /* [cap * kv_dim] */
    float *v_rows;
    int n, cap;
};

static int kv_reserve(qwen_eagle3_kv *kv, int kvd, int want) {
    if (want <= kv->cap) return 1;
    int nc = kv->cap ? kv->cap : 8;
    while (nc < want) nc *= 2;
    float *nk = realloc(kv->k_rows, (size_t)nc * kvd * sizeof(float));
    if (!nk) return 0;
    kv->k_rows = nk;
    float *nv = realloc(kv->v_rows, (size_t)nc * kvd * sizeof(float));
    if (!nv) return 0;
    kv->v_rows = nv;
    kv->cap = nc;
    return 1;
}

/* One EAGLE-3 token. The "hidden" input is either fused from the 3 aux hidden
 * (`aux3` non-NULL -- prefill / step 0) or a recurrent hidden supplied
 * directly (`fused_in` non-NULL -- draft steps 1.. , where EAGLE feeds its own
 * previous output). Exactly one of `aux3` / `fused_in` must be set.
 *
 * norm/norm concat -> q/k/v -> RoPE -> append to kv -> GQA causal attention
 * over kv[0..n] -> o_proj -> hidden residual -> SwiGLU MLP -> `out_hidden`
 * (the recurrent hidden, pre-final-norm) -> final norm -> draft lm_head.
 * `emb_row` is the current token's target embedding. Fills `tr` / `out_hidden`
 * when non-NULL. */
static int eagle_one(const qwen_eagle3 *e, qwen_eagle3_kv *kv,
                     const float *const *aux3, const float *fused_in,
                     const float *emb_row, int position, qwen_eagle3_trace *tr,
                     float *out_hidden, float *out_logits, char *err,
                     size_t errn) {
    const qwen_eagle3_config *c = &e->cfg;
    int H = c->hidden_size, I = c->intermediate_size, FIN = c->fusion_in_dim;
    int QD = c->q_dim, KVD = c->kv_dim;
    float eps = c->rms_norm_eps;
#define TR(field, src, n) \
    do { if (tr) memcpy(tr->field, (src), (size_t)(n) * sizeof(float)); } while (0)
    int ok = 0;
    float *xcat = malloc((size_t)FIN * sizeof(float));
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
    if (!xcat || !fused || !xqkv || !q || !kk || !vv || !heads || !attn ||
        !res || !y || !g || !u || !mlp) {
        set_err(err, errn, "out of memory in eagle step");
        goto out;
    }

    if (aux3) {
        for (int f = 0; f < c->fusion_count; f++)
            memcpy(xcat + (size_t)f * H, aux3[f], (size_t)H * sizeof(float));
        TR(aux_concat, xcat, FIN);
        matvec(fused, &e->fc, xcat);
    } else {
        memcpy(fused, fused_in, (size_t)H * sizeof(float)); /* recurrent hidden */
    }
    TR(fc_out, fused, H);

    rmsnorm(xqkv, emb_row, e->in_ln, H, eps);
    rmsnorm(xqkv + H, fused, e->hidden_norm, H, eps);
    TR(embed_norm, xqkv, H);
    TR(hidden_normed, xqkv + H, H);
    TR(qkv_in, xqkv, 2 * H);

    matvec(q, &e->q, xqkv);
    matvec(kk, &e->k, xqkv);
    matvec(vv, &e->v, xqkv);
    TR(q_pre_rope, q, QD);
    TR(k_pre_rope, kk, KVD);
    TR(v, vv, KVD);

    rope_inplace(q, c->num_attention_heads, c->head_dim, position, c->rope_theta);
    rope_inplace(kk, c->num_key_value_heads, c->head_dim, position, c->rope_theta);
    TR(q_post_rope, q, QD);
    TR(k_post_rope, kk, KVD);

    if (!kv_reserve(kv, KVD, kv->n + 1)) {
        set_err(err, errn, "out of memory growing eagle kv cache");
        goto out;
    }
    memcpy(kv->k_rows + (size_t)kv->n * KVD, kk, (size_t)KVD * sizeof(float));
    memcpy(kv->v_rows + (size_t)kv->n * KVD, vv, (size_t)KVD * sizeof(float));
    kv->n++;

    attn_gqa_causal(c, q, kv->k_rows, kv->v_rows, kv->n, heads);
    TR(attn_heads, heads, QD);
    matvec(attn, &e->o, heads);
    TR(attn_out, attn, H);

    for (int i = 0; i < H; i++) res[i] = fused[i] + attn[i]; /* residual = fused */

    rmsnorm(y, res, e->post_ln, H, eps);
    TR(post_attn_norm, y, H);
    matvec(g, &e->gate, y);
    matvec(u, &e->up, y);
    for (int i = 0; i < I; i++) {
        float x = g[i];
        g[i] = (x / (1.0f + expf(-x))) * u[i];
    }
    matvec(mlp, &e->down, g);
    TR(mlp_out, mlp, H);
    for (int i = 0; i < H; i++) res[i] += mlp[i];
    if (out_hidden) memcpy(out_hidden, res, (size_t)H * sizeof(float));

    rmsnorm(y, res, e->final_norm, H, eps);
    TR(final_hidden, y, H);
    matvec(out_logits, &e->lm_head, y);
    ok = 1;
#undef TR
out:
    free(xcat); free(fused); free(xqkv); free(q); free(kk); free(vv);
    free(heads); free(attn); free(res); free(y); free(g); free(u); free(mlp);
    return ok;
}

/* K/V-only: fuse aux3, RMSNorm(emb)|RMSNorm(fused), q/k/v proj, RoPE k, append
 * (k, v) to `kv`. No attention / MLP / lm_head. Used to build the EAGLE draft
 * prefix from target aux -- each prefix row is an independent
 * hidden(x[t]) + Emb(x[t+1]) draft-extend (NOT recurrent), so only its K/V
 * needs to land in the cache. */
static int eagle_kv_append(const qwen_eagle3 *e, qwen_eagle3_kv *kv,
                           const float *const *aux3, const float *emb_row,
                           int position, char *err, size_t errn) {
    const qwen_eagle3_config *c = &e->cfg;
    int H = c->hidden_size, FIN = c->fusion_in_dim, KVD = c->kv_dim;
    float eps = c->rms_norm_eps;
    int ok = 0;
    float *xcat = malloc((size_t)FIN * sizeof(float));
    float *fused = malloc((size_t)H * sizeof(float));
    float *xqkv = malloc((size_t)2 * H * sizeof(float));
    float *kk = malloc((size_t)KVD * sizeof(float));
    float *vv = malloc((size_t)KVD * sizeof(float));
    if (!xcat || !fused || !xqkv || !kk || !vv) {
        set_err(err, errn, "out of memory in eagle prefix append");
        goto out;
    }
    for (int f = 0; f < c->fusion_count; f++)
        memcpy(xcat + (size_t)f * H, aux3[f], (size_t)H * sizeof(float));
    matvec(fused, &e->fc, xcat);
    rmsnorm(xqkv, emb_row, e->in_ln, H, eps);
    rmsnorm(xqkv + H, fused, e->hidden_norm, H, eps);
    matvec(kk, &e->k, xqkv);
    matvec(vv, &e->v, xqkv);
    rope_inplace(kk, c->num_key_value_heads, c->head_dim, position, c->rope_theta);
    if (!kv_reserve(kv, KVD, kv->n + 1)) {
        set_err(err, errn, "out of memory growing eagle kv cache");
        goto out;
    }
    memcpy(kv->k_rows + (size_t)kv->n * KVD, kk, (size_t)KVD * sizeof(float));
    memcpy(kv->v_rows + (size_t)kv->n * KVD, vv, (size_t)KVD * sizeof(float));
    kv->n++;
    ok = 1;
out:
    free(xcat); free(fused); free(xqkv); free(kk); free(vv);
    return ok;
}

int qwen_eagle3_kv_prefix_extend(const qwen_eagle3 *e, qwen_eagle3_kv *kv,
                                 const float *const *aux_all,
                                 const uint32_t *pair_tokens,
                                 int n_pairs, int start_position,
                                 qwen_eagle3_embed_fn embed, void *embed_ctx,
                                 char *error, size_t errn) {
    if (!e || !kv || !aux_all || !pair_tokens || !embed || n_pairs < 0) {
        set_err(error, errn, "qwen_eagle3_kv_prefix_extend: bad args");
        return 0;
    }
    int H = e->cfg.hidden_size;
    float *emb = malloc((size_t)H * sizeof(float));
    if (!emb) {
        set_err(error, errn, "out of memory");
        return 0;
    }
    int ok = 1;
    for (int t = 0; ok && t < n_pairs; t++) {
        const float *a3[3] = {aux_all[0] + (size_t)t * H,
                              aux_all[1] + (size_t)t * H,
                              aux_all[2] + (size_t)t * H};
        if (!embed(embed_ctx, pair_tokens[t], emb)) {
            set_err(error, errn, "embed failed for prefix token %u",
                    pair_tokens[t]);
            ok = 0;
            break;
        }
        ok = eagle_kv_append(e, kv, a3, emb, start_position + t, error, errn);
    }
    free(emb);
    return ok;
}

int qwen_eagle3_forward_seq(const qwen_eagle3 *e, int T,
                            const float *const *aux_hidden,
                            const uint32_t *tokens, const int *positions,
                            qwen_eagle3_embed_fn embed, void *embed_ctx,
                            qwen_eagle3_trace *traces, float *out_logits,
                            char *error, size_t errn) {
    if (!e || T < 1 || !aux_hidden || !tokens || !positions || !embed ||
        !out_logits) {
        set_err(error, errn, "qwen_eagle3_forward_seq: bad args");
        return 0;
    }
    const qwen_eagle3_config *c = &e->cfg;
    int fc = c->fusion_count, dv = c->draft_vocab_size, H = c->hidden_size;
    qwen_eagle3_kv kv = {e, NULL, NULL, 0, 0};
    float *emb = malloc((size_t)H * sizeof(float));
    int ok = emb != NULL;
    for (int i = 0; ok && i < T; i++) {
        if (!embed(embed_ctx, tokens[i], emb)) {
            set_err(error, errn, "embedding accessor failed for token %u",
                    tokens[i]);
            ok = 0;
            break;
        }
        ok = eagle_one(e, &kv, aux_hidden + (size_t)i * fc, NULL, emb,
                       positions[i], traces ? traces + i : NULL, NULL,
                       out_logits + (size_t)i * dv, error, errn);
    }
    free(emb);
    free(kv.k_rows);
    free(kv.v_rows);
    return ok;
}

/* 1-token convenience wrapper -- identical maths to forward_seq with T=1
 * (the QINT-015h-1c parity path). */
int qwen_eagle3_step_ref(const qwen_eagle3 *e, const float *const *aux_hidden,
                         uint32_t prev_token, int position,
                         qwen_eagle3_embed_fn embed, void *embed_ctx,
                         qwen_eagle3_trace *tr, float *out_draft_logits,
                         char *error, size_t errn) {
    return qwen_eagle3_forward_seq(e, 1, aux_hidden, &prev_token, &position,
                                   embed, embed_ctx, tr, out_draft_logits, error,
                                   errn);
}

int qwen_eagle3_kv_new(const qwen_eagle3 *e, qwen_eagle3_kv **out, char *error,
                       size_t errn) {
    if (!e || !out) {
        set_err(error, errn, "qwen_eagle3_kv_new: bad args");
        return 0;
    }
    qwen_eagle3_kv *kv = calloc(1, sizeof(*kv));
    if (!kv) {
        set_err(error, errn, "out of memory");
        return 0;
    }
    kv->e = e;
    *out = kv;
    return 1;
}

int qwen_eagle3_kv_step(qwen_eagle3_kv *kv, const float *const *aux_hidden,
                        uint32_t token, int position, qwen_eagle3_embed_fn embed,
                        void *embed_ctx, qwen_eagle3_trace *trace,
                        float *out_draft_logits, char *error, size_t errn) {
    if (!kv || !aux_hidden || !embed || !out_draft_logits) {
        set_err(error, errn, "qwen_eagle3_kv_step: bad args");
        return 0;
    }
    float *emb = malloc((size_t)kv->e->cfg.hidden_size * sizeof(float));
    if (!emb) {
        set_err(error, errn, "out of memory");
        return 0;
    }
    int ok = embed(embed_ctx, token, emb);
    if (!ok)
        set_err(error, errn, "embedding accessor failed for token %u", token);
    else
        ok = eagle_one(kv->e, kv, aux_hidden, NULL, emb, position, trace, NULL,
                       out_draft_logits, error, errn);
    free(emb);
    return ok;
}

void qwen_eagle3_kv_free(qwen_eagle3_kv *kv) {
    if (!kv) return;
    free(kv->k_rows);
    free(kv->v_rows);
    free(kv);
}

void qwen_eagle3_kv_reset(qwen_eagle3_kv *kv) {
    if (kv) kv->n = 0;
}

int qwen_eagle3_kv_len(const qwen_eagle3_kv *kv) { return kv ? kv->n : 0; }

/* Drop rows past `keep` (>= 0). Used to roll a draft KV back to the accepted
 * target prefix after a verify. */
void qwen_eagle3_kv_truncate(qwen_eagle3_kv *kv, int keep) {
    if (kv && keep >= 0 && keep < kv->n) kv->n = keep;
}

/* QINT-015h-2b: an autoregressive draft chain. Step 0 fuses `aux3` (the target
 * residual at the frontier position) with Emb(`anchor_token`) at
 * `start_position` -- the "hidden(t) + Emb(token t+1)" 1-token shift. Each
 * later step feeds EAGLE's OWN previous output hidden (recurrent -- the 3 aux
 * are NOT re-fused) with Emb(previous draft token). `kv` carries the draft
 * layer's K/V across steps (caller resets / catches it up). Fills
 * `out_draft_ids` [k] with draft-vocab argmax ids (caller maps via d2t). */
int qwen_eagle3_chain(const qwen_eagle3 *e, qwen_eagle3_kv *kv,
                      const float *const *aux3, uint32_t anchor_token,
                      int start_position, int k, qwen_eagle3_embed_fn embed,
                      void *embed_ctx, uint32_t *out_draft_ids, char *error,
                      size_t errn) {
    if (!e || !kv || !aux3 || !embed || !out_draft_ids || k < 1) {
        set_err(error, errn, "qwen_eagle3_chain: bad args");
        return 0;
    }
    const qwen_eagle3_config *c = &e->cfg;
    int H = c->hidden_size, dv = c->draft_vocab_size;
    float *emb = malloc((size_t)H * sizeof(float));
    float *hidden = malloc((size_t)H * sizeof(float));
    float *logits = malloc((size_t)dv * sizeof(float));
    int ok = emb && hidden && logits;
    uint32_t cur = anchor_token;
    for (int j = 0; ok && j < k; j++) {
        if (!embed(embed_ctx, cur, emb)) {
            set_err(error, errn, "embedding accessor failed for token %u", cur);
            ok = 0;
            break;
        }
        ok = eagle_one(e, kv, j == 0 ? aux3 : NULL, j == 0 ? NULL : hidden, emb,
                       start_position + j, NULL, hidden, logits, error, errn);
        if (!ok) break;
        int am = 0;
        for (int i = 1; i < dv; i++)
            if (logits[i] > logits[am]) am = i;
        out_draft_ids[j] = (uint32_t)am;
        /* the next step's Emb() takes a *target*-vocab token id. */
        cur = qwen_eagle3_d2t(e, (uint32_t)am);
    }
    free(emb);
    free(hidden);
    free(logits);
    return ok;
}
