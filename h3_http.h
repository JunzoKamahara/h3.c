#ifndef H3_HTTP_H
#define H3_HTTP_H

/* Minimal blocking HTTP/1.1 server for the Phase 4 OpenAI-compatible endpoints.
 *
 * One request per connection (always `Connection: close`), single-threaded
 * accept loop, bounded header and body sizes. Enough for a local inference
 * server; it is not a hardened public web server. */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *method;      /* "GET", "POST", ... (uppercased) */
    const char *path;        /* request target, query string included */
    const char *body;        /* NUL-terminated; may be "" */
    size_t body_length;
    const char *content_type; /* request Content-Type header, or NULL */
} h3_http_request;

typedef struct h3_http_responder h3_http_responder;

/* Buffered reply: sends status line, Content-Type, Content-Length and body,
 * then closes. `body` may be NULL when `body_length` is 0. */
void h3_http_send(h3_http_responder *responder, int status,
                  const char *content_type, const void *body,
                  size_t body_length);

/* Streaming reply (for SSE): sends the status line and headers with no
 * Content-Length, then each h3_http_write() flushes bytes to the socket.
 * Finish with h3_http_finish(). */
void h3_http_begin_stream(h3_http_responder *responder, int status,
                          const char *content_type);
int h3_http_write(h3_http_responder *responder, const void *data, size_t n);
void h3_http_finish(h3_http_responder *responder);

typedef void (*h3_http_handler)(const h3_http_request *request,
                                h3_http_responder *responder, void *user);

typedef struct h3_http_server h3_http_server;

/* Bind and listen on host:port. port 0 lets the OS choose; read it back with
 * h3_http_server_port(). host NULL means 127.0.0.1. */
h3_http_server *h3_http_listen(const char *host, uint16_t port, char *error,
                              size_t error_size);
uint16_t h3_http_server_port(const h3_http_server *server);

/* Accept and handle connections until h3_http_stop() is called from a handler
 * or another thread. Returns 0 on clean stop, non-zero on a fatal accept
 * error. */
int h3_http_run(h3_http_server *server, h3_http_handler handler, void *user);
void h3_http_stop(h3_http_server *server);
void h3_http_close(h3_http_server *server);

#endif
