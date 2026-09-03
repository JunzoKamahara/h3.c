#ifndef QWEN_EAGLE_PROBE_H
#define QWEN_EAGLE_PROBE_H

/* QINT-015h-1a -- static compatibility probe for an EAGLE-3 draft checkpoint.
 *
 * Reads ONLY <checkpoint_dir>/config.json and the *.safetensors header(s)
 * (the JSON tensor map -- never any tensor data). Nothing touches the GPU,
 * nothing is allocated per weight. It answers one question: can this
 * checkpoint act as the speculative-decoding draft head for the current
 * Qwen3-VL-32B target (hidden 5120, vocab 151936, 64/8 heads, mRoPE
 * [24,20,20] theta 5e6)?
 *
 * The verdict decides whether QINT-015h-1b (the real tensor loader) may be
 * built for this checkpoint. Performance is NOT considered here.
 */

#include <stddef.h>
#include <stdio.h>

/* Architecture the draft head must match. qwen_eagle_target_default() fills
 * this from the H3 Qwen backbone constants (qwen_layers.h); a caller may also
 * load it from a target text_config.json with qwen_eagle_target_from_config(). */
typedef struct {
    const char *family;        /* label only, e.g. "Qwen3-VL-32B-Instruct" */
    int hidden_size;
    int vocab_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int intermediate_size;
    double rope_theta;
    int mrope_section[3];
    int mrope_interleaved;
} qwen_eagle_target;

void qwen_eagle_target_default(qwen_eagle_target *target);

/* Populate `target` from a Qwen3-VL config.json (accepts either the outer
 * config with a "text_config" object, or a bare text_config). Returns 1 on
 * success. `family` points into static storage. */
int qwen_eagle_target_from_config(const char *config_path,
                                  qwen_eagle_target *target, char *error,
                                  size_t error_size);

typedef enum {
    QWEN_EAGLE_COMPATIBLE = 0,
    QWEN_EAGLE_INCOMPATIBLE = 1,
    QWEN_EAGLE_PROBE_ERROR = 2 /* checkpoint unreadable -- not a verdict */
} qwen_eagle_verdict;

/* Probe `checkpoint_dir`. Writes a per-field human-readable report to `out`
 * (a FILE*), listing EVERY incompatibility, not just the first. Returns the
 * verdict. `error` is filled only for QWEN_EAGLE_PROBE_ERROR. */
qwen_eagle_verdict qwen_eagle_probe(const char *checkpoint_dir,
                                    const qwen_eagle_target *target, FILE *out,
                                    char *error, size_t error_size);

#endif
