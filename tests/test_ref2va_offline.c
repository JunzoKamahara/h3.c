/* P10-REF2VA-00: minimal offline sanity check for Ref2VA (reference-input)
 * generation.
 *
 * The DiT already accepts image/video/audio reference conditioning
 * (h3_dit_load_conditioned) and the CLI-level h3_generate() wires the whole
 * path end to end (reference -> VAE encode -> patchify -> Ref2VA text
 * conditioning -> conditioned transformer). But nothing in this repo had
 * ever run that path against a real reference and checked the result: the
 * only other caller of h3_dit_load_conditioned (tests/bench_dit.c) feeds it
 * all-zero condition rows for timing only.
 *
 * This generates two clips with the IDENTICAL prompt/seed/size/steps and
 * only the reference IMAGE swapped (solid red vs. solid blue), through the
 * real Ref2VA transformer checkpoint, and requires the two outputs to
 * differ by a real margin. If the reference conditioning rows were silently
 * dropped or misrouted, the two clips would come out identical.
 *
 * This is a wiring check, not a quality or parity check: there is no local
 * reference implementation to diff against, so it does not (and cannot)
 * prove the conditioning is being applied *correctly* -- only that it is
 * being applied at all.
 *
 * Slow (streams the 62 GB Ref2VA transformer checkpoint twice): not part of
 * `make test`.
 *
 *   ./h3_ref2va_offline_test MiniMax-H3
 */

#include "h3.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_ref2va_offline.c: %s\n", message);
    exit(1);
}
static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

#define REF2VA_SIZE 256
#define REF2VA_FRAMES 22 /* h3_generate()'s floor for one full 22-frame chunk */

static char *join(const char *dir, const char *name) {
    size_t n = strlen(dir) + strlen(name) + 2;
    char *path = malloc(n);
    if (path) snprintf(path, n, "%s/%s", dir, name);
    return path;
}

static void write_solid_reference(const char *path, uint8_t r, uint8_t g,
                                  uint8_t b) {
    size_t count = (size_t)REF2VA_SIZE * REF2VA_SIZE;
    uint8_t *pixels = malloc(count * 3);
    if (!pixels) fail("out of memory building a synthetic reference image");
    for (size_t index = 0; index < count; index++) {
        pixels[index * 3 + 0] = r;
        pixels[index * 3 + 1] = g;
        pixels[index * 3 + 2] = b;
    }
    char error[256];
    if (!h3_ffmpeg_write_png_rgb24(path, pixels, REF2VA_SIZE, REF2VA_SIZE,
                                   error, sizeof(error)))
        fail(error);
    free(pixels);
}

static void run_ref2va(h3_ctx *ctx, const char *reference_image_path,
                       const char *output_path) {
    h3_reference reference = {H3_REFERENCE_IMAGE, reference_image_path, NULL,
                              0};
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = REF2VA_SIZE;
    params.height = REF2VA_SIZE;
    params.frames = REF2VA_FRAMES;
    params.steps = 2;         /* minimum allowed -- wiring check, not quality */
    params.ssd_streaming = 1; /* matches the P8 job path; no 62 GB resident spike */
    params.references = &reference;
    params.reference_count = 1;
    params.output_path = output_path;

    h3_result *result = h3_generate(ctx, "A calm still scene.", &params);
    require(result != NULL, h3_last_error(ctx));
    h3_result_free(result);
}

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] : "MiniMax-H3";

    h3_ctx *ctx = h3_load_dir(model_dir);
    require(ctx != NULL, "h3_load_dir");

    const h3_model_info *model = h3_model(ctx);
    if (!model->ref2va_transformer.files) {
        fprintf(stderr,
                "skip: Ref2VA transformer checkpoint is not installed\n");
        h3_free(ctx);
        return 0;
    }

    char template[] = "/tmp/h3-ref2va-XXXXXX";
    char *scratch = mkdtemp(template);
    require(scratch != NULL, "mkdtemp scratch dir");

    char *ref_a = join(scratch, "ref-a.png");
    char *ref_b = join(scratch, "ref-b.png");
    char *out_a = join(scratch, "out-a.mp4");
    char *out_b = join(scratch, "out-b.mp4");
    require(ref_a && ref_b && out_a && out_b, "out of memory building paths");

    write_solid_reference(ref_a, 220, 40, 40);  /* solid red */
    write_solid_reference(ref_b, 40, 60, 220);  /* solid blue */

    fprintf(stderr, "ref2va-offline: generating clip A (red reference)\n");
    run_ref2va(ctx, ref_a, out_a);
    fprintf(stderr, "ref2va-offline: generating clip B (blue reference)\n");
    run_ref2va(ctx, ref_b, out_b);

    h3_free(ctx);
    ctx = NULL;

    float *pixels_a = NULL, *pixels_b = NULL;
    int frames_a = 0, frames_b = 0;
    char error[256];
    require(h3_ffmpeg_read_video_f32(out_a, REF2VA_SIZE, REF2VA_SIZE,
                                     REF2VA_FRAMES, &pixels_a, &frames_a,
                                     error, sizeof(error)),
           error);
    require(h3_ffmpeg_read_video_f32(out_b, REF2VA_SIZE, REF2VA_SIZE,
                                     REF2VA_FRAMES, &pixels_b, &frames_b,
                                     error, sizeof(error)),
           error);
    require(frames_a == frames_b,
           "the two clips decoded to different frame counts");

    size_t total = (size_t)3 * (size_t)frames_a * REF2VA_SIZE * REF2VA_SIZE;
    double sum_abs_diff = 0.0, mean_a = 0.0;
    for (size_t index = 0; index < total; index++) {
        require(isfinite(pixels_a[index]) && isfinite(pixels_b[index]),
               "generated output contains a non-finite pixel");
        sum_abs_diff += fabs((double)pixels_a[index] - (double)pixels_b[index]);
        mean_a += pixels_a[index];
    }
    mean_a /= (double)total;
    double sum_var_a = 0.0;
    for (size_t index = 0; index < total; index++) {
        double d = (double)pixels_a[index] - mean_a;
        sum_var_a += d * d;
    }
    double mean_abs_diff = sum_abs_diff / (double)total;
    double variance_a = sum_var_a / (double)total;

    printf("ref2va-offline: pixel variance (clip A)     = %.6f\n", variance_a);
    printf("ref2va-offline: mean abs diff (A vs B)       = %.6f\n",
          mean_abs_diff);

    require(variance_a > 1e-4, "clip A is degenerate (near-constant output)");
    require(mean_abs_diff > 1e-3,
           "swapping the reference image produced no measurable change in "
           "the output -- Ref2VA conditioning does not appear to reach the "
           "transformer");

    free(pixels_a);
    free(pixels_b);
    unlink(ref_a); unlink(ref_b); unlink(out_a); unlink(out_b);
    rmdir(scratch);
    free(ref_a); free(ref_b); free(out_a); free(out_b);

    printf("ok: Ref2VA reference conditioning measurably changes generated "
          "output\n");
    return 0;
}
