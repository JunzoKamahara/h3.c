/* QEXP-001 (non-blocking): how sensitive is the H3 DiT to the ~14 % layer-49
 * hidden drift of the Mixed-W4/BF16 chat decode path?
 *
 *   --emit BF16_FILE MIXED_FILE
 *       Writes the layer-49 conditioning two ways for a fixed prompt:
 *         BF16_FILE  = qwen_session_forward_to_layer(50)  (canonical, H3 path)
 *         MIXED_FILE = Mixed-W4/BF16 chat decode, captured via H3_QWEN_DUMP_L49
 *       Both are raw BF16 [n_tokens, 5120].
 *   --run BF16_FILE MIXED_FILE
 *       Loads a T2VA DiT with each conditioning, denoises the SAME seeded
 *       noise, and compares the resulting video / audio latents.
 *
 * `make qexp-001` runs both. NOT a default-gate blocker (a quantised state
 * never reaches the DiT in normal operation).
 */

#include "h3_dit.h"
#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "qwen_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HID 5120u
#define STEPS 16
#define SEED 42ULL

static const char *PROMPT = "A red fox walking through snow";

static void fail(const char *m) {
    fprintf(stderr, "FAIL tests/test_qwen_l49_h3_sensitivity.c: %s\n", m);
    exit(1);
}
static void require(int c, const char *m) { if (!c) fail(m); }

static float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static char *path_join(const char *a, const char *b) {
    size_t n = strlen(a) + strlen(b) + 2;
    char *r = malloc(n);
    if (!r) fail("alloc");
    snprintf(r, n, "%s/%s", a, b);
    return r;
}

static void progress(const char *phase, int c, int t, void *o) {
    (void)o;
    if (t > 0 && (c == t || c % 8 == 0))
        fprintf(stderr, "  %s %d/%d\n", phase, c, t);
}

static size_t tokenize(const char *root, uint32_t **ids_out) {
    char error[512];
    char *tp = path_join(root, "FL2VA/tokenizer/tokenizer.json");
    h3_tokenizer *tok = h3_tokenizer_load(tp, error, sizeof(error));
    if (!tok) fail(error);
    size_t n = 0;
    if (!h3_tokenizer_encode(tok, PROMPT, 1, ids_out, &n, error, sizeof(error)))
        fail(error);
    h3_tokenizer_free(tok);
    free(tp);
    require(n >= 3, "prompt too short");
    return n;
}

static void emit(const char *root, const char *bf16_file,
                 const char *mixed_file) {
    /* QEXP-003: honour a pre-set H3_QWEN_Q4* config (e.g. mixed + BF16
     * down/gate/up); default to plain `mixed` when nothing is set. */
    if (!getenv("H3_QWEN_Q4")) setenv("H3_QWEN_Q4", "mixed", 1);
    remove(mixed_file);
    setenv("H3_QWEN_DUMP_L49", mixed_file, 1);

    char error[512];
    uint32_t *ids = NULL;
    size_t n = tokenize(root, &ids);

    char *wp = path_join(root, "FL2VA/text_encoder");
    qwen_engine *engine = NULL;
    if (!qwen_engine_open(&engine, wp, "h3_shaders.metal", error, sizeof(error)))
        fail(error);
    qwen_session *session = NULL;
    if (!qwen_session_create(&session, engine, error, sizeof(error)))
        fail(error);
    if (!qwen_session_set_resident(session, 1, error, sizeof(error)))
        fail(error);

    /* Canonical BF16 layer-49 (the H3 path -- unaffected by H3_QWEN_Q4). */
    qwen_input in = {0};
    in.token_ids = ids;
    in.token_count = n;
    qwen_intermediate_state st = {0};
    if (!qwen_session_forward_to_layer(session, &in, 50, &st, NULL, NULL, error,
                                       sizeof(error)))
        fail(error);
    require(st.tokens == n && st.hidden_size == HID, "layer-49 shape");
    FILE *bf = fopen(bf16_file, "wb");
    require(bf && fwrite(st.values, sizeof(uint16_t), n * HID, bf) == n * HID,
            "cannot write BF16 conditioning");
    fclose(bf);
    qwen_intermediate_state_free(&st);

    /* Mixed-W4/BF16 chat decode, token by token -> H3_QWEN_DUMP_L49 appends. */
    require(qwen_session_rewind(session, 0, error, sizeof(error)), error);
    for (size_t i = 0; i < n; i++)
        if (!qwen_session_eval(session, &ids[i], 1, error, sizeof(error)))
            fail(error);

    qwen_session_free(session);
    qwen_engine_close(engine);
    h3_tokenizer_ids_free(ids);
    free(wp);

    FILE *mf = fopen(mixed_file, "rb");
    require(mf != NULL, "mixed conditioning was not written");
    fseek(mf, 0, SEEK_END);
    long got = ftell(mf);
    fclose(mf);
    require((size_t)got == n * HID * sizeof(uint16_t),
            "mixed conditioning size mismatch");
    printf("emit: prompt \"%s\" -> %zu tokens; wrote %s + %s\n", PROMPT, n,
           bf16_file, mixed_file);
}

static uint16_t *read_cond(const char *file, size_t n) {
    FILE *f = fopen(file, "rb");
    require(f != NULL, "cannot open conditioning file");
    uint16_t *v = malloc(n * HID * sizeof(*v));
    require(v && fread(v, sizeof(*v), n * HID, f) == n * HID,
            "conditioning size mismatch");
    fclose(f);
    return v;
}

static void denoise_with(const char *root, uint16_t *cond, size_t n,
                         float *video_out, float *audio_out, size_t *nv_out,
                         size_t *na_out) {
    char error[512];
    char *wp = path_join(root, "FL2VA/transformer");

    h3_text_embedding text = {0};
    text.tokens = n;
    text.width = HID;
    text.values = cond;

    h3_layout_spec spec = {(int)n, 2, 2, 2, 8, 5, NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) fail(error);
    h3_sigma_schedule sigmas;
    require(h3_schedule_build(STEPS, &sigmas), "cannot build sampler schedule");

    h3_dit *dit = h3_dit_load_t2va(
        wp, "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 1 /* ssd_streaming */, 1.0f,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        progress, NULL, error, sizeof(error));
    if (!dit) fail(error);

    size_t nv = h3_dit_video_elements(dit), na = h3_dit_audio_elements(dit);
    float *video = malloc(nv * sizeof(float));
    float *audio = malloc(na * sizeof(float));
    require(video && audio, "latent alloc");

    h3_rng rng;
    h3_rng_seed(&rng, SEED);
    h3_rng_fill_normal(&rng, video, nv);
    h3_rng_fill_normal(&rng, audio, na);

    if (!h3_dit_denoise(dit, video, audio, progress, NULL, error, sizeof(error)))
        fail(error);

    memcpy(video_out, video, nv * sizeof(float));
    memcpy(audio_out, audio, na * sizeof(float));
    *nv_out = nv;
    *na_out = na;

    free(video);
    free(audio);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(wp);
}

static void compare(const char *label, const float *a, const float *b,
                    size_t n) {
    double se = 0, sa = 0, dot = 0, na = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        se += d * d;
        sa += (double)b[i] * b[i];
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    double rel = sqrt(se / (sa > 1e-30 ? sa : 1e-30));
    double cos = dot / (sqrt(sa) * sqrt(na) + 1e-30);
    printf("  %-14s rel_l2=%.4f  cos=%.5f  max_abs=%.4f  (n=%zu)\n", label, rel,
           cos, mx, n);
}

int main(int argc, char **argv) {
    const char *root = "MiniMax-H3";
    if (argc >= 4 && !strcmp(argv[1], "--emit")) {
        emit(root, argv[2], argv[3]);
        return 0;
    }
    /* --perturb REL IN OUT : write OUT = IN + N(0, REL * per-element rms),
     * re-rounded to bf16. Calibrates "SSIM cost of a REL-magnitude layer-49
     * perturbation" independent of quantisation. */
    if (argc >= 5 && !strcmp(argv[1], "--perturb")) {
        double rel = atof(argv[2]);
        uint32_t *ids = NULL;
        size_t n = tokenize(root, &ids);
        h3_tokenizer_ids_free(ids);
        uint16_t *c = read_cond(argv[3], n);
        double ss = 0;
        for (size_t i = 0; i < n * HID; i++) {
            double v = bf16_to_f32(c[i]);
            ss += v * v;
        }
        float sigma = (float)(rel * sqrt(ss / (double)(n * HID)));
        uint64_t st = 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < n * HID; i++) {
            /* box-muller-ish from xorshift */
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            float u = ((st >> 11) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            float u2 = ((st >> 11) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
            float g = (u + u2 + ((float)((st >> 5) & 0xFFF) / 4096.0f - 0.5f)) *
                      1.7320508f;
            c[i] = f32_to_bf16(bf16_to_f32(c[i]) + sigma * g);
        }
        FILE *o = fopen(argv[4], "wb");
        require(o && fwrite(c, sizeof(uint16_t), n * HID, o) == n * HID,
                "perturb write");
        fclose(o);
        free(c);
        printf("perturb: rel=%.3f sigma=%.4g -> %s\n", rel, sigma, argv[4]);
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "--run")) {
        uint32_t *ids = NULL;
        size_t n = tokenize(root, &ids);
        h3_tokenizer_ids_free(ids);
        uint16_t *cb = read_cond(argv[2], n);
        uint16_t *cm = read_cond(argv[3], n);

        double drift = 0, dcos = 0, dsb = 0;
        for (size_t i = 0; i < n * HID; i++) {
            double bv = bf16_to_f32(cb[i]), mv = bf16_to_f32(cm[i]);
            drift += (mv - bv) * (mv - bv);
            dsb += bv * bv;
            dcos += mv * bv;
        }
        printf("conditioning drift (mixed vs bf16): rel_l2=%.4f  cos=%.5f\n",
               sqrt(drift / dsb), dcos / dsb);

        static float vb[1 << 18], ab[1 << 18], vm[1 << 18], am[1 << 18];
        size_t nv = 0, na = 0, nv2 = 0, na2 = 0;
        printf("### DiT denoise with BF16 conditioning\n");
        denoise_with(root, cb, n, vb, ab, &nv, &na);
        printf("### DiT denoise with Mixed-W4 conditioning\n");
        denoise_with(root, cm, n, vm, am, &nv2, &na2);
        require(nv == nv2 && na == na2, "latent geometry differs between runs");

        printf("H3 output sensitivity (Mixed-W4 cond vs BF16 cond, seed %llu, "
               "%d steps):\n", (unsigned long long)SEED, STEPS);
        compare("video latent", vm, vb, nv);
        compare("audio latent", am, ab, na);
        puts("ok: QEXP-001 layer-49 -> H3 sensitivity");

        free(cb);
        free(cm);
        return 0;
    }
    fail("usage: --emit BF16_FILE MIXED_FILE | --run BF16_FILE MIXED_FILE");
}
