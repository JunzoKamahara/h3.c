#include "qwen_eagle_probe.h"

#include "h3_json.h"
#include "h3_safetensors.h"
#include "qwen_layers.h" /* QWEN_LM_* target constants */

#include <dirent.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- *
 * QINT-015h-1a: read config.json + the safetensors header(s), decide whether
 * an EAGLE-3 draft checkpoint can be wired to the current target. No tensor
 * data, no GPU.
 * ------------------------------------------------------------------------- */

#define MAX_INCOMPAT 32

static void set_error(char *error, size_t size, const char *msg) {
    if (error && size) snprintf(error, size, "%s", msg);
}

static char *read_whole_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (len_out) *len_out = got;
    return buf;
}

/* config.json helpers: EAGLE configs vary, so every lookup tries a few
 * spellings and a few nesting levels. */
static const h3_json *dig(const h3_json *root, const char *const *path) {
    const h3_json *cur = root;
    for (size_t i = 0; path[i] && cur; i++) cur = h3_json_object_get(cur, path[i]);
    return cur;
}

/* Search `roots` (NULL-terminated) for the first that has `key`. */
static const h3_json *find_in(const h3_json *const *roots, const char *key) {
    for (size_t i = 0; roots[i]; i++) {
        const h3_json *v = h3_json_object_get(roots[i], key);
        if (v) return v;
    }
    return NULL;
}

static int cfg_int(const h3_json *const *roots, const char *const *keys,
                   int fallback, int *found) {
    for (size_t k = 0; keys[k]; k++) {
        const h3_json *v = find_in(roots, keys[k]);
        if (v && h3_json_is(v, H3_JSON_NUMBER)) {
            if (found) *found = 1;
            return (int)h3_json_number_or(v, fallback);
        }
    }
    if (found) *found = 0;
    return fallback;
}

static double cfg_num(const h3_json *const *roots, const char *const *keys,
                      double fallback, int *found) {
    for (size_t k = 0; keys[k]; k++) {
        const h3_json *v = find_in(roots, keys[k]);
        if (v && h3_json_is(v, H3_JSON_NUMBER)) {
            if (found) *found = 1;
            return h3_json_number_or(v, fallback);
        }
    }
    if (found) *found = 0;
    return fallback;
}

static const char *cfg_str(const h3_json *const *roots, const char *const *keys) {
    for (size_t k = 0; keys[k]; k++) {
        const h3_json *v = find_in(roots, keys[k]);
        if (v && h3_json_is(v, H3_JSON_STRING)) return h3_json_string_value(v);
    }
    return NULL;
}

/* ---- target -------------------------------------------------------------- */

void qwen_eagle_target_default(qwen_eagle_target *t) {
    if (!t) return;
    t->family = "Qwen3-VL-32B-Instruct (H3 backbone defaults)";
    t->hidden_size = QWEN_LM_HIDDEN;
    t->vocab_size = QWEN_LM_VOCAB;
    t->num_attention_heads = QWEN_LM_QUERY_HEADS;
    t->num_key_value_heads = QWEN_LM_KV_HEADS;
    t->head_dim = QWEN_LM_HEAD_DIM;
    t->intermediate_size = QWEN_LM_INTERMEDIATE;
    t->rope_theta = (double)QWEN_LM_ROPE_THETA;
    t->mrope_section[0] = 24;
    t->mrope_section[1] = 20;
    t->mrope_section[2] = 20;
    t->mrope_interleaved = 1;
}

int qwen_eagle_target_from_config(const char *config_path,
                                  qwen_eagle_target *t, char *error,
                                  size_t error_size) {
    if (!config_path || !t) {
        set_error(error, error_size, "qwen_eagle_target_from_config: bad args");
        return 0;
    }
    qwen_eagle_target_default(t);
    size_t len = 0;
    char *text = read_whole_file(config_path, &len);
    if (!text) {
        set_error(error, error_size, "cannot read target config.json");
        return 0;
    }
    char jerr[256] = {0};
    h3_json *root = h3_json_parse(text, len, jerr, sizeof(jerr));
    free(text);
    if (!root) {
        set_error(error, error_size, jerr[0] ? jerr : "target config parse");
        return 0;
    }
    const h3_json *tc = h3_json_object_get(root, "text_config");
    const h3_json *roots[] = {tc ? tc : root, root, NULL};
    static const char *k_hidden[] = {"hidden_size", NULL};
    static const char *k_vocab[] = {"vocab_size", NULL};
    static const char *k_nah[] = {"num_attention_heads", NULL};
    static const char *k_nkv[] = {"num_key_value_heads", NULL};
    static const char *k_hd[] = {"head_dim", NULL};
    static const char *k_int[] = {"intermediate_size", NULL};
    static const char *k_theta[] = {"rope_theta", NULL};
    int f;
    t->hidden_size = cfg_int(roots, k_hidden, t->hidden_size, &f);
    t->vocab_size = cfg_int(roots, k_vocab, t->vocab_size, &f);
    t->num_attention_heads = cfg_int(roots, k_nah, t->num_attention_heads, &f);
    t->num_key_value_heads = cfg_int(roots, k_nkv, t->num_key_value_heads, &f);
    t->head_dim = cfg_int(roots, k_hd, t->head_dim, &f);
    t->intermediate_size = cfg_int(roots, k_int, t->intermediate_size, &f);
    t->rope_theta = cfg_num(roots, k_theta, t->rope_theta, &f);
    const h3_json *rs = dig(roots[0], (const char *[]){"rope_scaling", NULL});
    if (!rs) rs = dig(root, (const char *[]){"rope_scaling", NULL});
    if (rs) {
        const h3_json *ms = h3_json_object_get(rs, "mrope_section");
        if (ms && h3_json_is(ms, H3_JSON_ARRAY) && h3_json_array_size(ms) >= 3)
            for (int i = 0; i < 3; i++)
                t->mrope_section[i] =
                    (int)h3_json_number_or(h3_json_array_at(ms, (size_t)i), 0);
        const h3_json *il = h3_json_object_get(rs, "mrope_interleaved");
        if (il) t->mrope_interleaved = h3_json_bool_or(il, t->mrope_interleaved);
    }
    t->family = "Qwen3-VL target (from config.json)";
    h3_json_free(root);
    return 1;
}

/* ---- tensor map -------------------------------------------------------- */

typedef struct {
    char *name;
    h3_dtype dtype;
    int ndim;
    uint64_t shape[8];
} tm_entry;

typedef struct {
    tm_entry *v;
    size_t n, cap;
} tensor_map;

static void tm_free(tensor_map *m) {
    for (size_t i = 0; i < m->n; i++) free(m->v[i].name);
    free(m->v);
    m->v = NULL;
    m->n = m->cap = 0;
}

static int tm_push(tensor_map *m, const h3_st_tensor *t) {
    if (m->n == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 64;
        tm_entry *nv = realloc(m->v, nc * sizeof(*nv));
        if (!nv) return 0;
        m->v = nv;
        m->cap = nc;
    }
    tm_entry *e = &m->v[m->n];
    e->name = strdup(t->name);
    if (!e->name) return 0;
    e->dtype = t->dtype;
    e->ndim = t->ndim;
    for (int i = 0; i < 8; i++) e->shape[i] = t->shape[i];
    m->n++;
    return 1;
}

static int has_suffix(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && !strcmp(s + ls - lf, suf);
}
static int has_sub(const char *s, const char *sub) { return strstr(s, sub) != NULL; }

/* First tensor whose name ends with `suffix` (EAGLE prefixes vary:
 * `fc.weight`, `midlayer.fc.weight`, `model.fc.weight`). */
static const tm_entry *tm_by_suffix(const tensor_map *m, const char *suffix) {
    for (size_t i = 0; i < m->n; i++)
        if (has_suffix(m->v[i].name, suffix)) return &m->v[i];
    return NULL;
}
static const tm_entry *tm_by_exact(const tensor_map *m, const char *name) {
    for (size_t i = 0; i < m->n; i++)
        if (!strcmp(m->v[i].name, name)) return &m->v[i];
    return NULL;
}

static int load_tensor_map(const char *dir, tensor_map *m, char *error,
                           size_t error_size) {
    DIR *d = opendir(dir);
    if (!d) {
        set_error(error, error_size, "cannot open checkpoint directory");
        return 0;
    }
    struct dirent *e;
    int files = 0;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' || !has_suffix(e->d_name, ".safetensors"))
            continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        h3_st_header h;
        char herr[256] = {0};
        if (!h3_st_read_header(path, &h, herr, sizeof(herr))) {
            closedir(d);
            snprintf(error, error_size, "safetensors header: %s", herr);
            return 0;
        }
        for (size_t i = 0; i < h.tensor_count; i++)
            if (!tm_push(m, &h.tensors[i])) {
                h3_st_free_header(&h);
                closedir(d);
                set_error(error, error_size, "out of memory");
                return 0;
            }
        h3_st_free_header(&h);
        files++;
    }
    closedir(d);
    if (!files) {
        set_error(error, error_size, "no .safetensors files in directory");
        return 0;
    }
    return 1;
}

/* A weight tensor is "not plain float" if its dtype is integer/bool or if the
 * checkpoint carries GPTQ/AWQ-style packing companions. */
static int dtype_is_float(h3_dtype d) {
    return d == H3_DTYPE_F16 || d == H3_DTYPE_BF16 || d == H3_DTYPE_F32 ||
           d == H3_DTYPE_F64;
}

/* ---- the probe ------------------------------------------------------------ */

typedef struct {
    const char *items[MAX_INCOMPAT];
    char store[MAX_INCOMPAT][160];
    int n;
} reasons;

static void add_reason(reasons *r, const char *fmt, ...) {
    if (r->n >= MAX_INCOMPAT) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->store[r->n], sizeof(r->store[r->n]), fmt, ap);
    va_end(ap);
    r->items[r->n] = r->store[r->n];
    r->n++;
}

static const char *ok_bad(int ok) { return ok ? "OK" : "MISMATCH"; }

qwen_eagle_verdict qwen_eagle_probe(const char *dir,
                                    const qwen_eagle_target *target, FILE *out,
                                    char *error, size_t error_size) {
    if (!dir || !target || !out) {
        set_error(error, error_size, "qwen_eagle_probe: bad args");
        return QWEN_EAGLE_PROBE_ERROR;
    }

    char cfg_path[4096];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", dir);
    size_t clen = 0;
    char *ctext = read_whole_file(cfg_path, &clen);
    if (!ctext) {
        set_error(error, error_size, "no config.json in checkpoint directory");
        return QWEN_EAGLE_PROBE_ERROR;
    }
    char jerr[256] = {0};
    h3_json *root = h3_json_parse(ctext, clen, jerr, sizeof(jerr));
    free(ctext);
    if (!root) {
        set_error(error, error_size, jerr[0] ? jerr : "config.json parse error");
        return QWEN_EAGLE_PROBE_ERROR;
    }

    tensor_map tm = {0};
    char terr[256] = {0};
    int have_tensors = load_tensor_map(dir, &tm, terr, sizeof(terr));

    /* draft config: try the outer object, a nested "model" / "draft_model",
     * and an "eagle_config" side-car. */
    const h3_json *model = h3_json_object_get(root, "model");
    const h3_json *draft = h3_json_object_get(root, "draft_model");
    const h3_json *eagle = h3_json_object_get(root, "eagle_config");
    const h3_json *roots[] = {root, model, draft, eagle, NULL};

    static const char *k_arch[] = {"architectures", NULL};
    static const char *k_mt[] = {"model_type", NULL};
    static const char *k_hidden[] = {"hidden_size", "d_model", NULL};
    static const char *k_tvocab[] = {"vocab_size", "target_vocab_size", NULL};
    static const char *k_dvocab[] = {"draft_vocab_size", "draft_vocab",
                                     "vocab_size_draft", NULL};
    static const char *k_layers[] = {"num_hidden_layers", "num_layers",
                                     "n_layer", NULL};
    static const char *k_nah[] = {"num_attention_heads", "n_head", NULL};
    static const char *k_nkv[] = {"num_key_value_heads", "num_kv_heads", NULL};
    static const char *k_hd[] = {"head_dim", NULL};
    static const char *k_int[] = {"intermediate_size", "ffn_dim", NULL};
    static const char *k_theta[] = {"rope_theta", "rotary_emb_base", NULL};
    static const char *k_dtype[] = {"torch_dtype", "dtype", NULL};
    static const char *k_aux[] = {"num_hidden_states_to_capture",
                                  "num_aux_hidden_states", "fusion_num_hidden",
                                  NULL};

    int hf, dvf, tvf, lf, nahf, nkvf, hdf, intf, thf, auxf;
    int d_hidden = cfg_int(roots, k_hidden, 0, &hf);
    int d_tvocab = cfg_int(roots, k_tvocab, 0, &tvf);
    int d_dvocab = cfg_int(roots, k_dvocab, 0, &dvf);
    int d_layers = cfg_int(roots, k_layers, 0, &lf);
    int d_nah = cfg_int(roots, k_nah, 0, &nahf);
    int d_nkv = cfg_int(roots, k_nkv, 0, &nkvf);
    int d_hd = cfg_int(roots, k_hd, 0, &hdf);
    int d_int = cfg_int(roots, k_int, 0, &intf);
    double d_theta = cfg_num(roots, k_theta, 0.0, &thf);
    int d_aux_cfg = cfg_int(roots, k_aux, 0, &auxf);
    const char *d_mt = cfg_str(roots, k_mt);
    const char *d_dtype = cfg_str(roots, k_dtype);

    const char *d_arch = NULL;
    for (size_t i = 0; roots[i]; i++) {
        const h3_json *a = h3_json_object_get(roots[i], "architectures");
        if (a && h3_json_is(a, H3_JSON_ARRAY) && h3_json_array_size(a) > 0) {
            d_arch = h3_json_string_value(h3_json_array_at(a, 0));
            break;
        }
    }
    (void)k_arch;

    /* rope_scaling / mrope on the draft side. */
    int d_mrope[3] = {0, 0, 0};
    int d_mrope_il = -1, have_mrope = 0;
    for (size_t i = 0; roots[i]; i++) {
        const h3_json *rs = h3_json_object_get(roots[i], "rope_scaling");
        if (!rs) continue;
        const h3_json *ms = h3_json_object_get(rs, "mrope_section");
        if (ms && h3_json_is(ms, H3_JSON_ARRAY) && h3_json_array_size(ms) >= 3) {
            for (int j = 0; j < 3; j++)
                d_mrope[j] =
                    (int)h3_json_number_or(h3_json_array_at(ms, (size_t)j), 0);
            have_mrope = 1;
        }
        const h3_json *il = h3_json_object_get(rs, "mrope_interleaved");
        if (il) d_mrope_il = h3_json_bool_or(il, 0);
        break;
    }

    /* ---- tensor-derived facts ---- */
    const tm_entry *fc = have_tensors ? tm_by_suffix(&tm, "fc.weight") : NULL;
    const tm_entry *d2t = NULL, *t2d = NULL, *lm_head = NULL, *embed = NULL;
    int draft_layer_max = -1, quant_companions = 0, int_weights = 0;
    h3_dtype weight_dtype = H3_DTYPE_UNKNOWN;
    if (have_tensors) {
        d2t = tm_by_suffix(&tm, "d2t");
        if (!d2t) d2t = tm_by_exact(&tm, "d2t");
        t2d = tm_by_suffix(&tm, "t2d");
        if (!t2d) t2d = tm_by_exact(&tm, "t2d");
        lm_head = tm_by_suffix(&tm, "lm_head.weight");
        embed = tm_by_suffix(&tm, "embed_tokens.weight");
        for (size_t i = 0; i < tm.n; i++) {
            const char *nm = tm.v[i].name;
            if (has_suffix(nm, "qweight") || has_suffix(nm, ".scales") ||
                has_suffix(nm, ".qzeros") || has_suffix(nm, ".g_idx"))
                quant_companions++;
            if (has_suffix(nm, ".weight") &&
                (has_sub(nm, "proj") || has_sub(nm, "mlp") ||
                 has_sub(nm, "fc")) &&
                !dtype_is_float(tm.v[i].dtype))
                int_weights++;
            if (has_suffix(nm, "proj.weight") &&
                weight_dtype == H3_DTYPE_UNKNOWN && dtype_is_float(tm.v[i].dtype))
                weight_dtype = tm.v[i].dtype;
            /* midlayer.<k>... or layers.<k>... -> draft decoder depth */
            const char *ml = strstr(nm, "midlayer.");
            const char *ly = strstr(nm, "layers.");
            const char *p = ml ? ml + 9 : (ly ? ly + 7 : NULL);
            if (p) {
                int k = atoi(p);
                if (k > draft_layer_max) draft_layer_max = k;
            }
        }
    }
    int fc_fusion_count = 0, fc_out_ok = 0;
    if (fc && fc->ndim == 2) {
        fc_out_ok = ((int)fc->shape[0] == target->hidden_size);
        if (target->hidden_size > 0 &&
            fc->shape[1] % (uint64_t)target->hidden_size == 0)
            fc_fusion_count = (int)(fc->shape[1] / (uint64_t)target->hidden_size);
    }
    int draft_layer_count =
        (draft_layer_max >= 0) ? draft_layer_max + 1 : (lf ? d_layers : 0);

    /* ---- essential compatibility checks ---- */
    reasons R = {0};

    /* Positive identification: this must actually BE an EAGLE-3 draft head,
     * not (say) the full target model or an unrelated checkpoint. */
    int arch_says_eagle = 0;
    {
        char low[128] = {0};
        const char *s = d_arch ? d_arch : (d_mt ? d_mt : "");
        for (size_t i = 0; s[i] && i < sizeof(low) - 1; i++)
            low[i] = (char)((s[i] >= 'A' && s[i] <= 'Z') ? s[i] + 32 : s[i]);
        arch_says_eagle = strstr(low, "eagle") != NULL;
    }
    int looks_like_draft =
        (fc != NULL) || arch_says_eagle ||
        (draft_layer_count >= 1 && draft_layer_count <= 2 &&
         (d2t != NULL || (dvf && d_dvocab > 0)));
    if (!looks_like_draft)
        add_reason(&R,
                   "not an EAGLE-3 draft head: no fc fusion tensor, %d decoder "
                   "layer(s), architecture '%s'",
                   draft_layer_count, d_arch ? d_arch : (d_mt ? d_mt : "?"));
    else if (draft_layer_count > 2)
        add_reason(&R, "draft decoder has %d layers (EAGLE-3 head is 1, rarely 2)",
                   draft_layer_count);
    if (looks_like_draft && !fc)
        add_reason(&R, "no fc fusion tensor found -- cannot fuse the target's "
                       "aux hidden states");

    int hidden_ok = hf && d_hidden == target->hidden_size;
    if (hf && !hidden_ok)
        add_reason(&R, "hidden_size: draft=%d target=%d", d_hidden,
                   target->hidden_size);
    else if (!hf && fc && !fc_out_ok)
        add_reason(&R, "fc.weight rows=%llu != target hidden_size %d",
                   (unsigned long long)fc->shape[0], target->hidden_size);

    if (fc && !fc_out_ok)
        add_reason(&R, "fc fusion output %llu != target hidden_size %d",
                   (unsigned long long)fc->shape[0], target->hidden_size);
    if (fc && fc_fusion_count == 0)
        add_reason(&R,
                   "fc fusion input %llu is not a multiple of target hidden %d",
                   (unsigned long long)fc->shape[1], target->hidden_size);

    /* vocab mapping: needed whenever the draft LM head is a reduced vocab. */
    int reduced_vocab = (dvf && d_dvocab > 0 && d_dvocab != target->vocab_size) ||
                        (lm_head && lm_head->ndim == 2 &&
                         (int)lm_head->shape[0] != target->vocab_size);
    if (tvf && d_tvocab != target->vocab_size)
        add_reason(&R, "target vocab_size in config: draft=%d target=%d",
                   d_tvocab, target->vocab_size);
    if (have_tensors && reduced_vocab && !d2t)
        add_reason(&R, "reduced draft vocab but no d2t mapping tensor");
    if (have_tensors && reduced_vocab && !t2d)
        add_reason(&R, "reduced draft vocab but no t2d mapping tensor");

    /* drafter quantization: the 015h-1b C loader handles plain bf16/fp16
     * only. GPTQ/AWQ packing on the drafter's own weights is out of scope. */
    int drafter_quantized = quant_companions > 0 || int_weights > 0;
    const h3_json *qc = NULL;
    for (size_t i = 0; roots[i]; i++) {
        qc = h3_json_object_get(roots[i], "quantization_config");
        if (qc) break;
    }
    const char *quant_method = NULL;
    if (qc) {
        const h3_json *qm = h3_json_object_get(qc, "quant_method");
        quant_method = qm ? h3_json_string_value(qm) : "(unspecified)";
    }
    if (have_tensors && drafter_quantized)
        add_reason(&R,
                   "drafter weights are quantized (%d packing tensors, %d "
                   "int-typed weights) -- C loader is bf16/fp16 only",
                   quant_companions, int_weights);

    if (!have_tensors)
        add_reason(&R, "safetensors header unreadable: %s", terr);

    /* ---- report ---- */
    fprintf(out, "EAGLE3 compatibility probe -- %s\n\n", dir);
    fprintf(out, "Target:\n");
    fprintf(out, "  family                  %s\n", target->family);
    fprintf(out, "  hidden_size             %d\n", target->hidden_size);
    fprintf(out, "  vocab_size              %d\n", target->vocab_size);
    fprintf(out, "  heads (q/kv)            %d / %d   head_dim %d\n",
            target->num_attention_heads, target->num_key_value_heads,
            target->head_dim);
    fprintf(out, "  intermediate_size       %d\n", target->intermediate_size);
    fprintf(out, "  rope_theta              %.0f\n", target->rope_theta);
    fprintf(out, "  mrope_section           [%d,%d,%d]  interleaved %d\n",
            target->mrope_section[0], target->mrope_section[1],
            target->mrope_section[2], target->mrope_interleaved);

    fprintf(out, "\nDraft:\n");
    fprintf(out, "  architectures[0]        %s\n", d_arch ? d_arch : "(absent)");
    fprintf(out, "  model_type              %s\n", d_mt ? d_mt : "(absent)");
    if (hf)
        fprintf(out, "  hidden_size             %d          %s\n", d_hidden,
                ok_bad(hidden_ok));
    else
        fprintf(out, "  hidden_size             (absent in config)\n");
    if (dvf)
        fprintf(out, "  draft_vocab_size        %d\n", d_dvocab);
    if (tvf)
        fprintf(out, "  vocab_size (target)     %d          %s\n", d_tvocab,
                ok_bad(d_tvocab == target->vocab_size));
    if (nahf || nkvf || hdf)
        fprintf(out,
                "  heads (q/kv)            %d / %d   head_dim %d   (draft "
                "attention is self-contained; informational)\n",
                d_nah, d_nkv, d_hd);
    if (intf)
        fprintf(out, "  intermediate_size       %d\n", d_int);
    fprintf(out, "  decoder layers          %d%s\n", draft_layer_count,
            draft_layer_count == 1 ? "           (EAGLE-3 single layer)" : "");
    if (auxf)
        fprintf(out, "  aux hidden (config)     %d\n", d_aux_cfg);
    if (thf)
        fprintf(out, "  rope_theta              %.0f          (loader param, "
                     "not a gate)\n",
                d_theta);
    if (have_mrope)
        fprintf(out,
                "  mrope_section           [%d,%d,%d]  interleaved %d   (loader "
                "param)\n",
                d_mrope[0], d_mrope[1], d_mrope[2], d_mrope_il);
    fprintf(out, "  config dtype            %s\n",
            d_dtype ? d_dtype : "(absent)");
    fprintf(out, "  weight_dtype (tensors)  %s\n",
            weight_dtype == H3_DTYPE_UNKNOWN ? "(no float proj weight found)"
                                             : h3_dtype_name(weight_dtype));
    fprintf(out, "  quantization            %s%s\n",
            quant_method ? quant_method : "none",
            drafter_quantized ? "  (drafter tensors packed)" : "");

    fprintf(out, "\nFusion (fc.weight):\n");
    if (fc)
        fprintf(out,
                "  shape                   [%llu, %llu]   -> out %s, fuses %d x "
                "%d\n",
                (unsigned long long)fc->shape[0],
                (unsigned long long)fc->shape[1], ok_bad(fc_out_ok),
                fc_fusion_count, target->hidden_size);
    else
        fprintf(out, "  fc.weight               (not found in safetensors)\n");

    fprintf(out, "\nVocabulary mapping:\n");
    if (lm_head)
        fprintf(out, "  lm_head.weight          [%llu, %llu]\n",
                (unsigned long long)lm_head->shape[0],
                (unsigned long long)lm_head->shape[1]);
    fprintf(out, "  d2t                     %s\n",
            d2t ? "present" : "ABSENT");
    if (d2t)
        fprintf(out, "                          shape [%llu]\n",
                (unsigned long long)d2t->shape[0]);
    fprintf(out, "  t2d                     %s\n",
            t2d ? "present" : "ABSENT");
    if (t2d)
        fprintf(out, "                          shape [%llu]\n",
                (unsigned long long)t2d->shape[0]);
    fprintf(out, "  embed_tokens.weight     %s\n",
            embed ? "present (draft owns embeddings)"
                  : "absent (shares target embeddings)");

    if (have_tensors)
        fprintf(out, "\nSafetensors: %zu tensors across the directory.\n", tm.n);

    fprintf(out, "\n");
    qwen_eagle_verdict verdict;
    if (!have_tensors && !root) {
        verdict = QWEN_EAGLE_PROBE_ERROR;
    } else if (R.n == 0) {
        fprintf(out, "RESULT: COMPATIBLE\n");
        fprintf(out, "  -> QINT-015h-1b (tensor loader) may be built for this "
                     "checkpoint.\n");
        verdict = QWEN_EAGLE_COMPATIBLE;
    } else {
        fprintf(out, "RESULT: INCOMPATIBLE\n");
        fprintf(out, "reason:\n");
        for (int i = 0; i < R.n; i++)
            fprintf(out, "  - %s\n", R.items[i]);
        fprintf(out, "  -> do NOT build 015h-1b for this checkpoint; reselect.\n");
        verdict = QWEN_EAGLE_INCOMPATIBLE;
    }

    tm_free(&tm);
    h3_json_free(root);
    return verdict;
}
