#ifndef QWEN_EAGLE3_H
#define QWEN_EAGLE3_H

/* QINT-015h-1b -- load an EAGLE-3 draft-head checkpoint into the C runtime.
 *
 * Scope: read config.json, read every tensor from model.safetensors, validate
 * names and shapes, and expose a *reference* CPU forward. This step does NOT
 * connect to the speculative coordinator and does NOT claim the forward is
 * numerically correct against the reference (Python / SGLang) implementation
 * -- that is QINT-015h-1c.
 *
 * The draft SHARES the target's token-embedding table (the checkpoint has no
 * embed_tokens.weight); a caller passes an accessor, never a copy.
 *
 * Checkpoint layout (mattbucci/Qwen3-VL-32B-AWQ-EAGLE3, 15 tensors):
 *   fc.weight                              [5120, 15360]   fuse 3x5120 -> 5120
 *   midlayer.input_layernorm.weight        [5120]          norm(embedding)
 *   midlayer.hidden_norm.weight            [5120]          norm(fused hidden)
 *   midlayer.self_attn.q_proj.weight       [4096, 10240]   10240 = 2*hidden
 *   midlayer.self_attn.k_proj.weight       [1024, 10240]
 *   midlayer.self_attn.v_proj.weight       [1024, 10240]
 *   midlayer.self_attn.o_proj.weight       [5120, 4096]
 *   midlayer.post_attention_layernorm.weight [5120]
 *   midlayer.mlp.gate_proj.weight          [32768, 5120]
 *   midlayer.mlp.up_proj.weight            [32768, 5120]
 *   midlayer.mlp.down_proj.weight          [5120, 32768]
 *   norm.weight                            [5120]          final RMSNorm
 *   lm_head.weight                         [32000, 5120]   draft vocab
 *   d2t                                    i64  [32000]    draft id -> target
 *   t2d                                    bool [151936]   target-in-draft mask
 * No biases anywhere. head_dim is explicit (128); 32*128 = 4096 != hidden.
 */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char architecture[64];  /* "LlamaForCausalLMEagle3" */
    char model_type[32];    /* "llama"                  */
    int hidden_size;        /* 5120 -- must match the target */
    int draft_vocab_size;   /* 32000 */
    int target_vocab_size;  /* 151936 */
    int num_attention_heads;/* 32 */
    int num_key_value_heads;/* 8  */
    int head_dim;           /* 128 -- authoritative, NOT hidden/heads */
    int q_dim;              /* num_attention_heads * head_dim = 4096 */
    int kv_dim;             /* num_key_value_heads * head_dim = 1024 */
    int qkv_in_dim;         /* q_proj.weight columns = 10240 (2*hidden here) */
    int intermediate_size;  /* 32768 */
    int num_hidden_layers;  /* 1 */
    int fusion_in_dim;      /* fc.weight columns = 15360 */
    int fusion_count;       /* fusion_in_dim / hidden_size = 3 */
    float rms_norm_eps;     /* 1e-5 (the draft's own eps, not the target's) */
    double rope_theta;      /* 5e6 */
    int rope_is_mrope;      /* 0 -- model_type llama => plain 1D rotary.
                             * 1c must confirm against the reference impl. */
} qwen_eagle3_config;

typedef struct qwen_eagle3 qwen_eagle3;

/* Write the `hidden_size` f32 target-embedding row for `token` into `out`.
 * Return 1 on success. The draft never owns this table. */
typedef int (*qwen_eagle3_embed_fn)(void *ctx, uint32_t token, float *out);

/* Load <checkpoint_dir>/config.json + model.safetensors. Fails (with a reason)
 * on an unknown tensor, a missing required tensor, a shape mismatch, or a
 * non-bf16 weight. */
int qwen_eagle3_load(const char *checkpoint_dir, qwen_eagle3 **out, char *error,
                     size_t error_size);
void qwen_eagle3_free(qwen_eagle3 *eagle);

const qwen_eagle3_config *qwen_eagle3_config_of(const qwen_eagle3 *eagle);

/* Vocabulary mapping. d2t is stored as a delta: target_id = draft_id + d2t. */
uint32_t qwen_eagle3_d2t(const qwen_eagle3 *eagle, uint32_t draft_id);
int qwen_eagle3_t2d_ok(const qwen_eagle3 *eagle, uint32_t target_id);

/* QINT-015h-1c: every intermediate of one reference step, for staged parity
 * against SpecForge / SGLang `LlamaForCausalLMEagle3`. All buffers are f32 and
 * sized from the config by qwen_eagle3_trace_alloc(). A 1-token step has a
 * degenerate softmax, so `q_*_rope` / `k_*_rope` must be compared directly --
 * the final logits alone do not verify RoPE or the Q/K projections. */
typedef struct {
    float *aux_concat;    /* [fusion_in_dim]  the 3 aux hidden, concatenated */
    float *fc_out;        /* [hidden]         fc.weight @ aux_concat         */
    float *embed_norm;    /* [hidden]         RMSNorm(embedding, input_ln)   */
    float *hidden_normed; /* [hidden]         RMSNorm(fc_out, hidden_norm)   */
    float *qkv_in;        /* [qkv_in_dim]     concat(embed_norm, hidden_normed) */
    float *q_pre_rope;    /* [q_dim]  */
    float *k_pre_rope;    /* [kv_dim] */
    float *v;             /* [kv_dim] */
    float *q_post_rope;   /* [q_dim]  */
    float *k_post_rope;   /* [kv_dim] */
    float *attn_heads;    /* [q_dim]          per-head attention result       */
    float *attn_out;      /* [hidden]         o_proj @ attn_heads             */
    float *post_attn_norm;/* [hidden]         RMSNorm(fc_out+attn_out, post)  */
    float *mlp_out;       /* [hidden]         SwiGLU MLP output               */
    float *final_hidden;  /* [hidden]         RMSNorm(residual, norm)         */
} qwen_eagle3_trace;

int qwen_eagle3_trace_alloc(const qwen_eagle3 *eagle, qwen_eagle3_trace *trace);
void qwen_eagle3_trace_free(qwen_eagle3_trace *trace);

/* One standalone EAGLE-3 step, CPU reference. `aux_hidden` is `fusion_count`
 * pointers to `hidden_size` f32 (the target's low/mid/high residual at the
 * current position); `prev_token` is the last committed token; `position` is
 * the absolute RoPE position. Fills `out_draft_logits` (`draft_vocab_size`
 * f32); also fills `trace` when non-NULL. Single position, no KV cache.
 *
 * NOT parity-verified -- QINT-015h-1c compares the trace stage by stage
 * against SpecForge / SGLang. */
int qwen_eagle3_step_ref(const qwen_eagle3 *eagle, const float *const *aux_hidden,
                         uint32_t prev_token, int position,
                         qwen_eagle3_embed_fn embed, void *embed_ctx,
                         qwen_eagle3_trace *trace, float *out_draft_logits,
                         char *error, size_t error_size);

#endif
