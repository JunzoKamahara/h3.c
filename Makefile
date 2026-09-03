CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	qwen_engine.c qwen_layers.c qwen_policy.c qwen_q4.c qwen_lm.c qwen_kv.c qwen_chat.c qwen_tools.c qwen_stream.c \
	qwen_spec.c qwen_draft_oracle.c qwen_draft_ngram.c qwen_eagle_probe.c qwen_eagle3.c qwen_draft_eagle.c \
	h3_json.c h3_http.c qwen_server.c \
	h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test parity real-parity phase0-parity phase1-parity phase2-parity \
	phase3-check phase4-check phase5-check phase6-check stream-check phase7-check bench-chat resident-check q4-check q4-decode-check quant-eval quant-calib quant-eval-awq quant-ablate l49-drift qexp-001 qexp-001b qexp-001b-control qexp-001b-save qexp-002 qexp-002-save qexp-003 qint-009 qint-011 qint-010 \
	spec-pending-oracle-check spec-pending-reject-check spec-pending-boundary-check spec-pending-eos-check spec-pending-vlm-check spec-pending-parity spec-batch-rewind-check spec-kernel-check spec-verify-parity spec-bench spec-stage-bench spec-chain-drift-check spec-chain-drift-gate-check spec-aux-capture-check spec-eagle-probe-check spec-eagle3-load-check spec-eagle-live-check spec-check phase7-vlm-check clean

all: h3 h3_serve libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_serve: h3_serve_main.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_intermediate_test: tests/test_qwen_intermediate.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_lm_test: tests/test_qwen_lm.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_kv_test: tests/test_qwen_kv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_chat_test: tests/test_qwen_chat.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_server_test: tests/test_qwen_server.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_tools_test: tests/test_qwen_tools.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_responses_test: tests/test_qwen_responses.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_stream_test: tests/test_qwen_stream.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_vlm_test: tests/test_qwen_vlm.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_bench: tests/bench_qwen.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_matmul_bench: tests/bench_qwen_matmul.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_spec_bench: tests/bench_qwen_spec.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_verify_stage_bench: tests/bench_qwen_verify_stage.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QINT-015d-3: verify-M vs scalar-M timing (draft quality removed). Mixed
# target, warm-up first, per-trial rewind untimed. Two context lengths.
spec-bench: h3_qwen_spec_bench
	H3_QWEN_Q4=mixed ./h3_qwen_spec_bench MiniMax-H3

# QINT-015e-1: which projection carries the verify-M fixed cost (no model).
spec-stage-bench: h3_qwen_verify_stage_bench
	./h3_qwen_verify_stage_bench

h3_qwen_resident_test: tests/test_qwen_resident.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_q4_test: tests/test_qwen_q4.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_quant_eval: tests/test_qwen_quant_eval.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_l49_drift: tests/test_qwen_l49_drift.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_qwen_spec_test: tests/test_qwen_spec.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QINT-015d -- speculative decoding. Everything runs against the production
# Mixed-W4/BF16 target. The coordinator emits the target's greedy argmax
# sequence; a divergence from a particular greedy run is allowed only at a
# decode near-tie (parity_check rebuilds the logits at the divergence).
#
# QINT-015d-2 -- pending-anchor coordinator (1 target batch forward / cycle):
spec-pending-oracle-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-oracle
spec-pending-reject-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-reject
spec-pending-boundary-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-boundary
spec-pending-eos-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-eos
spec-pending-vlm-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-vlm
spec-pending-parity: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test pending-parity
# QINT-015d-0: append-block-then-rewind transaction, text + non-zero mRoPE.
spec-batch-rewind-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test batch-rewind
# QINT-015d-1: scalar decode vs the batched verifier, per row + margin
# histogram.
spec-verify-parity: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test verify-parity
# QINT-015e-0: chained verify_block vs teacher-forced scalar -- the numerical
# baseline to re-run after any 015e kernel change.
spec-chain-drift-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test chain-drift
spec-chain-drift-gate-check: h3_qwen_spec_test
	./h3_qwen_spec_test chain-drift-gate
# QINT-015h-2b-1: live EAGLE draft wiring -- real {1,32,60} aux + anchor +
# resident-embedding partial-row accessor; backend == direct chain at L-1.
# EAGLE_CKPT (default ~/models/mattbucci-eagle3) can override.
spec-eagle-live-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test eagle-live $(EAGLE_CKPT)
# QINT-015h-1a: static EAGLE-3 checkpoint compatibility probe (no model, no GPU).
h3_qwen_eagle_probe: tests/probe_qwen_eagle.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)
spec-eagle-probe-check: h3_qwen_eagle_probe
	./h3_qwen_eagle_probe --selftest
# QINT-015h-1b: load an EAGLE-3 checkpoint into the C runtime + reference-forward
# smoke. No GPU, no coordinator, no parity claim (that is 1c).
h3_qwen_eagle3_test: tests/test_qwen_eagle3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)
# --selftest now also covers QINT-015h-2a (KV / causal) and -2b (draft chain
# + qwen_draft_eagle backend, first-step alignment).
spec-eagle3-load-check: h3_qwen_eagle3_test
	./h3_qwen_eagle3_test --selftest
# QINT-015h-2b-0: run one real EAGLE draft chain (needs the checkpoint).
spec-eagle3-chain-smoke: h3_qwen_eagle3_test
	./h3_qwen_eagle3_test gen-fixture /tmp/eagle3_fix1.json 5120 1234 37 1
	./h3_qwen_eagle3_test chain $(EAGLE_CKPT) /tmp/eagle3_fix1.json
# QINT-015h-1c / -2a: deterministic fixtures + staged C reference traces
# (1-token = 1c, 2-token causal + step-wise-KV-vs-batch invariant = 2a). Then
# run scripts/eagle3_reference.py (numpy, or SpecForge/SGLang with hooks) and
# scripts/eagle3_compare.py on the same fixtures. EAGLE_CKPT overrides.
EAGLE_CKPT ?= $(HOME)/models/mattbucci-eagle3
spec-eagle3-parity-trace: h3_qwen_eagle3_test
	./h3_qwen_eagle3_test gen-fixture /tmp/eagle3_fix1.json 5120 1234 37 1
	./h3_qwen_eagle3_test dump $(EAGLE_CKPT) /tmp/eagle3_fix1.json /tmp/eagle3_c1.json
	./h3_qwen_eagle3_test gen-fixture /tmp/eagle3_fix2.json 5120 1234 37 2
	./h3_qwen_eagle3_test dump $(EAGLE_CKPT) /tmp/eagle3_fix2.json /tmp/eagle3_c2.json
	@echo "next, per fixture N in {1,2}:"
	@echo "  python3 scripts/eagle3_reference.py $(EAGLE_CKPT) /tmp/eagle3_fixN.json /tmp/eagle3_refN.json"
	@echo "  python3 scripts/eagle3_compare.py   /tmp/eagle3_cN.json /tmp/eagle3_refN.json"
spec-aux-capture-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test aux-capture
spec-check: h3_qwen_spec_test
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test coordinator
	H3_QWEN_Q4=mixed ./h3_qwen_spec_test lowlevel

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_qwen_intermediate_test h3_qwen_lm_test h3_qwen_kv_test \
	h3_qwen_chat_test h3_qwen_server_test h3_qwen_tools_test \
	h3_qwen_responses_test h3_qwen_stream_test h3_qwen_vlm_test \
	h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test h3_qwen_q4_test

	./h3_tests
	./h3_qwen_q4_test
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors; then \
		./h3_qwen_intermediate_test MiniMax-H3; \
		./h3_qwen_lm_test MiniMax-H3; \
		./h3_qwen_kv_test MiniMax-H3; \
		./h3_qwen_chat_test MiniMax-H3; \
		./h3_qwen_server_test MiniMax-H3; \
		./h3_qwen_tools_test MiniMax-H3; \
		./h3_qwen_responses_test MiniMax-H3; \
		./h3_qwen_stream_test; \
		./h3_qwen_vlm_test MiniMax-H3; \
	else \
		echo "skip: released Qwen text-encoder weights are not installed"; \
	fi
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

# Phase 0 release-blocking check: the Qwen layer-49 intermediate state reached
# through qwen_session_get_h3_conditioning() must stay bit-for-bit identical to
# the legacy h3_text_encode_bf16() / h3_text_encode_multimodal_bf16() output.
phase0-parity: h3_qwen_intermediate_test
	./h3_qwen_intermediate_test MiniMax-H3

# Phase 1 check: forward_full() must decompose exactly into
# continue_from_intermediate(get_h3_conditioning()), be run-to-run
# deterministic, and leave the layer-49 boundary untouched.
phase1-parity: h3_qwen_lm_test
	./h3_qwen_lm_test MiniMax-H3

# Phase 2 check: the KV-cache session (prefill, incremental decode, chunked
# prefill, rewind, multi-turn) must stay bit-for-bit with the Phase 1 full
# forward and be deterministic.
phase2-parity: h3_qwen_kv_test
	./h3_qwen_kv_test MiniMax-H3

# Phase 3 check: ChatML rendering / tokenization for system/user/assistant/tool
# and one end-to-end templated chat turn through the KV session.
phase3-check: h3_qwen_chat_test
	./h3_qwen_chat_test MiniMax-H3

# Phase 4 check: JSON parser, HTTP loopback for /v1/models, and
# /v1/chat/completions (buffered + SSE) through the runtime.
phase4-check: h3_qwen_server_test
	./h3_qwen_server_test MiniMax-H3

# Phase 5 check: tool-call markup parse/render and a function-calling round
# trip through /v1/chat/completions.
phase5-check: h3_qwen_tools_test
	./h3_qwen_tools_test MiniMax-H3

# Phase 6 check: POST /v1/responses (buffered, array input, tools, streaming).
phase6-check: h3_qwen_responses_test
	./h3_qwen_responses_test MiniMax-H3

# Incremental tool-call streamer unit test (no model).
stream-check: h3_qwen_stream_test
	./h3_qwen_stream_test

# Phase 7 check: the multimodal layer-49 state is shared by H3 and the Chat tail.
phase7-check: h3_qwen_vlm_test
	./h3_qwen_vlm_test MiniMax-H3

# Chat-engine throughput probe (prefill + incremental decode tok/s).
bench-chat: h3_qwen_bench
	./h3_qwen_bench MiniMax-H3 8

# Per-shape decode-matmul microbenchmark (bf16 vs int8), no model weights.
bench-matmul: h3_qwen_matmul_bench
	./h3_qwen_matmul_bench

# Approach B: a resident-weights session must decode bit-for-bit like a
# streaming one, and much faster. Holds two sessions at once (~65 GB+); not
# part of `make test`.
resident-check: h3_qwen_resident_test
	./h3_qwen_resident_test MiniMax-H3 4

# INT4 decode GEMV (chat-speedup step #2): kernel correctness + quantisation
# error on random matrices. No model weights, safe for `make test`.
q4-check: h3_qwen_q4_test
	./h3_qwen_q4_test

# QINT-015d-1: W4 decode-batch kernel (rows 2..5) vs the scalar q4 GEMV,
# row-for-row. No model weights. Part of `./h3_qwen_q4_test`.
spec-kernel-check: h3_qwen_q4_test
	./h3_qwen_q4_test

# End-to-end INT4 decode against the BF16 streaming path (argmax + bounded
# logit error + speedup). Opt-in feature; needs the resident + streaming
# sessions at once like resident-check.
q4-decode-check: h3_qwen_resident_test
	H3_QWEN_Q4=1 ./h3_qwen_resident_test MiniMax-H3 8

# Quantization quality eval (QINT-008..012 baseline): teacher-forced logit
# comparison over a fixed prompt set. Two resident loads (BF16 then W4A16),
# ~3 min. Not part of `make test`.
quant-eval: h3_qwen_quant_eval
	H3_QWEN_Q4=0 ./h3_qwen_quant_eval MiniMax-H3 --emit-ref quant_bf16_ref.f32
	H3_QWEN_Q4=1 ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32

# AWQ calibration capture (QINT-006): mean |x| per projection input over a
# disjoint calibration prompt set -> quant_calib.awqc.
quant-calib: h3_qwen_quant_eval
	H3_QWEN_Q4=0 ./h3_qwen_quant_eval MiniMax-H3 --emit-calib quant_calib.awqc

# AWQ quality eval (QINT-008/009): same comparison as quant-eval but the
# W4A16 path uses AWQ scaling from quant_calib.awqc.
quant-eval-awq: h3_qwen_quant_eval quant_calib.awqc
	test -f quant_bf16_ref.f32 || \
	  H3_QWEN_Q4=0 ./h3_qwen_quant_eval MiniMax-H3 --emit-ref quant_bf16_ref.f32
	H3_QWEN_Q4=1 H3_QWEN_Q4_AWQ=quant_calib.awqc \
	  ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32

quant_calib.awqc:
	$(MAKE) quant-calib

# Layer-49 hidden drift (QINT-012): decode-path layer-49 residual vs the BF16
# canonical (forward_to_layer). Two runs -- BF16 decode (kernel drift only)
# and Mixed-W4/BF16 (kernel + quant).
l49-drift: h3_qwen_l49_drift
	@echo "### BF16 decode path (kernel drift only)"
	H3_QWEN_Q4=0 ./h3_qwen_l49_drift MiniMax-H3
	@echo "### Mixed-W4/BF16 decode path"
	H3_QWEN_Q4=mixed ./h3_qwen_l49_drift MiniMax-H3

# QEXP-003 (non-blocking): cheap K_M-style conditioning-quant levers on the
# chat 0..49, measured by layer-49 drift only (no DiT -- fast). C1 keeps
# down_proj BF16 (the pathological-channel source); C2 also keeps gate/up
# BF16. See docs/quant-eval-baseline.md "QEXP-003". For the DiT perceptual
# leg, emit conditioning with test_qwen_l49_h3_sensitivity and feed qexp-001b.
qexp-003: h3_qwen_l49_drift
	@echo "### C0  mixed (q/o/gate/up/down W4, K/V BF16)"
	H3_QWEN_Q4=mixed ./h3_qwen_l49_drift MiniMax-H3
	@echo "### C1  mixed + BF16 down_proj"
	H3_QWEN_Q4=mixed H3_QWEN_Q4_BF16_PROJ=down ./h3_qwen_l49_drift MiniMax-H3
	@echo "### C2  mixed + BF16 down_proj + gate/up  (only q/o W4)"
	H3_QWEN_Q4=mixed H3_QWEN_Q4_BF16_PROJ=down,gateup ./h3_qwen_l49_drift MiniMax-H3

# Tensor/layer ablation (QINT-016): re-run the eval with parts of the resident
# INT4 set forced back to BF16, to localise which projections cause the argmax
# flips. Each variant is a fresh process (one resident load, ~50 s).
quant-ablate: h3_qwen_quant_eval
	test -f quant_bf16_ref.f32 || \
	  H3_QWEN_Q4=0 ./h3_qwen_quant_eval MiniMax-H3 --emit-ref quant_bf16_ref.f32
	@echo "### A: all W4 (lm_head W4)"
	H3_QWEN_Q4=1 H3_QWEN_Q4_HEAD=1 ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32
	@echo "### B: W4 + BF16 lm_head"
	H3_QWEN_Q4=1 ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32
	@echo "### C: W4 + BF16 layers 56-63"
	H3_QWEN_Q4=1 H3_QWEN_Q4_BF16_LAYERS=56-63 ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32
	@echo "### D: W4 + BF16 layers 50-63"
	H3_QWEN_Q4=1 H3_QWEN_Q4_BF16_LAYERS=50-63 ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32
	@echo "### E: W4 + BF16 K/V"
	H3_QWEN_Q4=1 H3_QWEN_Q4_BF16_PROJ=kv ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32
	@echo "### F: W4 + BF16 down_proj"
	H3_QWEN_Q4=1 H3_QWEN_Q4_BF16_PROJ=down ./h3_qwen_quant_eval MiniMax-H3 --compare quant_bf16_ref.f32

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_serve h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_qwen_intermediate_test h3_qwen_lm_test \
		h3_qwen_kv_test h3_qwen_chat_test h3_qwen_server_test h3_qwen_tools_test h3_qwen_responses_test h3_qwen_stream_test h3_qwen_vlm_test h3_qwen_bench \
		h3_qwen_resident_test h3_qwen_q4_test h3_qwen_quant_eval \
		h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 \
	libh3.a *.o *.d tests/*.o tests/*.d

h3_qwen_l49_h3_sensitivity: tests/test_qwen_l49_h3_sensitivity.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QEXP-001 (non-blocking): H3 DiT sensitivity to the Mixed-W4/BF16 layer-49
# drift. Two conditionings -> two full DiT denoise runs -> latent comparison.
# Loads the 62 GB FL2VA transformer twice (SSD-streamed). Long.
qexp-001: h3_qwen_l49_h3_sensitivity
	./h3_qwen_l49_h3_sensitivity --emit qexp_cond_bf16.bin qexp_cond_mixed.bin
	./h3_qwen_l49_h3_sensitivity --run  qexp_cond_bf16.bin qexp_cond_mixed.bin

h3_qwen_l49_h3_perceptual: tests/test_qwen_l49_h3_perceptual.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QEXP-001b (non-blocking): perceptual sensitivity -- a real (256x256) H3
# generation from BF16 vs Mixed-W4 conditioning, VAE-decoded, compared with
# SSIM / PSNR / audio SNR. Loads transformer + video VAE + audio VAE twice.
# Very long (~20-40 min). Needs `make qexp-001` first for the conditionings.
qexp-001b: h3_qwen_l49_h3_perceptual
	./h3_qwen_l49_h3_perceptual --run qexp_cond_bf16.bin qexp_cond_mixed.bin

# Control for qexp-001b: same conditioning twice, to prove the DiT+VAE
# pipeline is deterministic (should be SSIM/corr = 1.0).
qexp-001b-control: h3_qwen_l49_h3_perceptual
	./h3_qwen_l49_h3_perceptual --control qexp_cond_bf16.bin

# Same as qexp-001b but also writes qexp_bf16.mp4 / qexp_mixed.mp4 /
# qexp_sidebyside.mp4 (left=BF16 cond, right=Mixed-W4 cond) for eyeballing.
qexp-001b-save: h3_qwen_l49_h3_perceptual
	./h3_qwen_l49_h3_perceptual --save qexp_cond_bf16.bin qexp_cond_mixed.bin

h3_qexp002_shared_prefix: tests/test_qexp002_shared_prefix.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QEXP-002 (non-blocking): --quality-mode shared layers-0..49 prefix for a
# combined Chat + H3 request. Verifies bit-for-bit Chat parity + shows the
# saved 0..49 forward. Loads text encoder + transformer + VAEs.
qexp-002: h3_qexp002_shared_prefix
	./h3_qexp002_shared_prefix
qexp-002-save: h3_qexp002_shared_prefix
	./h3_qexp002_shared_prefix --save

h3_qwen_ja_generation: tests/test_qwen_ja_generation.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QINT-009: light Japanese task-level check, BF16 vs Mixed-W4/BF16 decode.
# 10 plain JA chat turns, greedy. Mechanical gates (non-empty / valid UTF-8 /
# no runaway repetition / sane length) are asserted; the printed pairs are
# for eyeballing meaning + fluency. Two resident loads + a compare.
qint-009: h3_qwen_ja_generation
	H3_QWEN_Q4=0     ./h3_qwen_ja_generation --emit qint009_bf16.txt
	H3_QWEN_Q4=mixed ./h3_qwen_ja_generation --emit qint009_mixed.txt
	./h3_qwen_ja_generation --compare qint009_bf16.txt qint009_mixed.txt

h3_qwen_tool_parity: tests/test_qwen_tool_parity.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QINT-011: tool-calling parity, BF16 vs Mixed-W4/BF16 decode. Two resident
# loads (~1-2 min each) + a compare. Not part of `make test`.
qint-011: h3_qwen_tool_parity
	H3_QWEN_Q4=0     ./h3_qwen_tool_parity --emit qint011_bf16.txt
	H3_QWEN_Q4=mixed ./h3_qwen_tool_parity --emit qint011_mixed.txt
	./h3_qwen_tool_parity --compare qint011_bf16.txt qint011_mixed.txt

h3_qwen_vlm_image_test: tests/test_qwen_vlm_image.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# P7-004: end-to-end VLM chat from a real (ffmpeg-synthesised) image.
phase7-vlm-check: h3_qwen_vlm_image_test
	./h3_qwen_vlm_image_test MiniMax-H3

h3_qwen_vlm_parity: tests/test_qwen_vlm_parity.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# QINT-010: VLM answer parity, BF16 vs Mixed-W4/BF16 decode. Two resident-ish
# loads + a compare. Not part of `make test`.
qint-010: h3_qwen_vlm_parity
	H3_QWEN_Q4=0     ./h3_qwen_vlm_parity --emit qint010_bf16.txt
	H3_QWEN_Q4=mixed ./h3_qwen_vlm_parity --emit qint010_mixed.txt
	./h3_qwen_vlm_parity --compare qint010_bf16.txt qint010_mixed.txt
