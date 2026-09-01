#ifndef QWEN_SERVER_H
#define QWEN_SERVER_H

/* Phase 4 -- OpenAI-compatible HTTP server.
 *
 * Dependency direction (spec section 38): this module sits above the runtime.
 * It depends on qwen_engine.h / h3_tokenizer.h / h3_http.h / h3_json.h; the
 * Qwen engine knows nothing about HTTP or OpenAI JSON.
 *
 * Endpoints:
 *   GET  /v1/models
 *   POST /v1/chat/completions   (stream = true|false)
 *
 * Decoding is greedy (Phase 2 sampler); tool calls and /v1/responses are later
 * phases. */

#include <stddef.h>
#include <stdint.h>

typedef struct qwen_server qwen_server;

/* `weight_directory` is the Qwen text-encoder directory; `tokenizer_path` the
 * tokenizer.json; `model_id` is what /v1/models and responses report.
 * `resident` != 0 pins all 64 decoder layers in Unified Memory (~62 GB) once
 * at startup so every request decodes without re-streaming them (Approach B);
 * the environment variable H3_QWEN_RESIDENT=1 does the same. */
int qwen_server_create(qwen_server **out, const char *weight_directory,
                       const char *tokenizer_path,
                       const char *shader_source_path, const char *model_id,
                       int resident, char *error, size_t error_size);
void qwen_server_free(qwen_server *server);

/* Bind host:port (port 0 = OS-assigned) and serve until qwen_server_stop().
 * The bound port is reported through *bound_port when non-NULL. */
int qwen_server_run(qwen_server *server, const char *host, uint16_t port,
                    uint16_t *bound_port, char *error, size_t error_size);
void qwen_server_stop(qwen_server *server);

#endif
