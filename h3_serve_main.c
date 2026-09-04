/* h3-runtime OpenAI-compatible server entry point (Phase 4).
 *
 *   h3_serve --model MiniMax-H3 [--port 8080] [--host 127.0.0.1]
 *            [--shaders h3_shaders.metal] [--model-id minimax-h3]
 *
 * --model is the release root; the Qwen text encoder and tokenizer are taken
 * from MODEL/FL2VA/text_encoder and MODEL/FL2VA/tokenizer/tokenizer.json.
 */

#include "qwen_server.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* QINT-014: Chat/VLM decode precision mode.
 *
 *   default        Mixed-W4/BF16   ~5 tok/s, ~30 GB resident   (quality-qualified)
 *   --fast         Pure W4A16      ~6 tok/s                     (opt-in, lower quality)
 *   --quality      BF16            reference quality
 *
 * --allow-remote-images lets the /v1 endpoints fetch http(s) image URLs
 * (data: URIs always work). Off by default; a coarse internal-host guard
 * applies even when on.
 *
 * All three keep H3 media conditioning on the canonical BF16 path: the H3
 * text encoder (h3_text_encoder.c) is a separate code path that never reads
 * H3_QWEN_Q4, and h3_conditioning_accepts() rejects a non-BF16 chat state at
 * the bridge. --fast is "fast Chat decode", never "fast conditioning".
 *
 * Precedence:  explicit --fast/--quality  >  H3_QWEN_Q4 in the environment  >
 * built-in default (mixed). We implement it by writing H3_QWEN_Q4 here: the
 * flags overwrite it, the default only fills it in when unset. */
static void apply_decode_mode(int fast, int quality) {
    if (quality) {
        setenv("H3_QWEN_Q4", "0", 1);
        fprintf(stderr, "decode mode: --quality (BF16)\n");
    } else if (fast) {
        setenv("H3_QWEN_Q4", "1", 1);
        fprintf(stderr, "decode mode: --fast (Pure W4A16)\n");
    } else if (setenv("H3_QWEN_Q4", "mixed", 0) == 0 &&
               !strcmp(getenv("H3_QWEN_Q4"), "mixed")) {
        fprintf(stderr, "decode mode: default (Mixed-W4/BF16)\n");
    } else {
        fprintf(stderr, "decode mode: H3_QWEN_Q4=%s (from environment)\n",
                getenv("H3_QWEN_Q4"));
    }
}

static qwen_server *g_server = NULL;

static void on_signal(int signal_number) {
    (void)signal_number;
    if (g_server) qwen_server_stop(g_server);
}

static char *join_path(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (result) snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

int main(int argc, char **argv) {
    const char *model_root = NULL;
    const char *host = "127.0.0.1";
    const char *shaders = "h3_shaders.metal";
    const char *model_id = "minimax-h3";
    long port = 8080;
    int stream_weights = 0;
    int fast = 0, quality = 0;

    for (int index = 1; index < argc; index++) {
        const char *arg = argv[index];
        const char *value = index + 1 < argc ? argv[index + 1] : NULL;
        if (!strcmp(arg, "--model") && value) model_root = argv[++index];
        else if (!strcmp(arg, "--host") && value) host = argv[++index];
        else if (!strcmp(arg, "--shaders") && value) shaders = argv[++index];
        else if (!strcmp(arg, "--model-id") && value) model_id = argv[++index];
        else if (!strcmp(arg, "--stream")) stream_weights = 1;
        else if (!strcmp(arg, "--resident")) { /* now the default */ }
        else if (!strcmp(arg, "--fast")) fast = 1;
        else if (!strcmp(arg, "--quality")) quality = 1;
        else if (!strcmp(arg, "--allow-remote-images"))
            setenv("H3_ALLOW_REMOTE_IMAGES", "1", 1);
        else if (!strcmp(arg, "--port") && value) port = strtol(argv[++index],
                                                                NULL, 10);
        else {
            fprintf(stderr,
                    "usage: %s --model ROOT [--port N] [--host H] "
                    "[--shaders PATH] [--model-id ID] [--stream] "
                    "[--fast | --quality] [--allow-remote-images]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!model_root || port < 0 || port > 65535) {
        fprintf(stderr, "%s: --model is required and --port must be 0..65535\n",
                argv[0]);
        return 2;
    }
    if (fast && quality) {
        fprintf(stderr, "%s: --fast and --quality are mutually exclusive\n",
                argv[0]);
        return 2;
    }
    apply_decode_mode(fast, quality);

    char *weights = join_path(model_root, "FL2VA/text_encoder");
    char *tokenizer = join_path(model_root, "FL2VA/tokenizer/tokenizer.json");
    if (!weights || !tokenizer) {
        fprintf(stderr, "%s: out of memory\n", argv[0]);
        return 1;
    }

    char error[512];
    if (!stream_weights)
        fprintf(stderr, "loading resident decoder weights (~62 GB; --stream to skip)...\n");
    if (!qwen_server_create(&g_server, weights, tokenizer, shaders, model_id,
                            stream_weights, error, sizeof(error))) {
        fprintf(stderr, "%s: %s\n", argv[0], error);
        free(weights);
        free(tokenizer);
        return 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    uint16_t bound = 0;
    fprintf(stderr, "h3-runtime serving %s on http://%s:%ld ...\n", model_id,
            host, port);
    int ok = qwen_server_run(g_server, host, (uint16_t)port, &bound, error,
                             sizeof(error));
    if (bound && bound != (uint16_t)port)
        fprintf(stderr, "(bound port %u)\n", bound);
    if (!ok) fprintf(stderr, "%s: %s\n", argv[0], error);

    qwen_server_free(g_server);
    g_server = NULL;
    free(weights);
    free(tokenizer);
    return ok ? 0 : 1;
}
