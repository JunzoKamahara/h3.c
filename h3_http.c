#include "h3_http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

enum {
    HTTP_MAX_HEADERS = 64 * 1024,
    HTTP_MAX_BODY = 8 * 1024 * 1024
};

struct h3_http_server {
    int listen_fd;
    uint16_t port;
    atomic_int stop;
};

struct h3_http_responder {
    int fd;
    int started;
    int streaming;
    int failed;
};

static int write_all(int fd, const void *data, size_t length) {
    const char *cursor = data;
    while (length) {
        ssize_t written = write(fd, cursor, length);
        if (written < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        cursor += written;
        length -= (size_t)written;
    }
    return 1;
}

static const char *status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

void h3_http_send(h3_http_responder *responder, int status,
                  const char *content_type, const void *body,
                  size_t body_length) {
    if (!responder || responder->started) return;
    responder->started = 1;
    if (!content_type) content_type = "application/octet-stream";
    char header[512];
    int header_length = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text(status), content_type, body_length);
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) {
        responder->failed = 1;
        return;
    }
    if (!write_all(responder->fd, header, (size_t)header_length) ||
        (body_length && !write_all(responder->fd, body, body_length)))
        responder->failed = 1;
}

void h3_http_begin_stream(h3_http_responder *responder, int status,
                          const char *content_type) {
    if (!responder || responder->started) return;
    responder->started = 1;
    responder->streaming = 1;
    if (!content_type) content_type = "text/event-stream";
    char header[512];
    int header_length = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text(status), content_type);
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) {
        responder->failed = 1;
        return;
    }
    if (!write_all(responder->fd, header, (size_t)header_length))
        responder->failed = 1;
}

int h3_http_write(h3_http_responder *responder, const void *data, size_t n) {
    if (!responder || !responder->streaming || responder->failed) return 0;
    if (!n) return 1;
    if (!write_all(responder->fd, data, n)) {
        responder->failed = 1;
        return 0;
    }
    return 1;
}

void h3_http_finish(h3_http_responder *responder) {
    (void)responder; /* connection is closed by the accept loop */
}

h3_http_server *h3_http_listen(const char *host, uint16_t port, char *error,
                               size_t error_size) {
#define HTTP_FAIL(msg) do {                                                    \
    if (error && error_size) snprintf(error, error_size, "%s: %s", msg,        \
                                      strerror(errno));                         \
    if (fd >= 0) close(fd);                                                     \
    free(server);                                                              \
    return NULL;                                                               \
} while (0)
    int fd = -1;
    h3_http_server *server = calloc(1, sizeof(*server));
    if (!server) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) HTTP_FAIL("cannot create socket");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr =
        host ? inet_addr(host) : htonl(INADDR_LOOPBACK);
    if (address.sin_addr.s_addr == INADDR_NONE)
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        HTTP_FAIL("cannot bind");
    if (listen(fd, 16) < 0) HTTP_FAIL("cannot listen");

    socklen_t length = sizeof(address);
    if (getsockname(fd, (struct sockaddr *)&address, &length) == 0)
        server->port = ntohs(address.sin_port);
    else
        server->port = port;
    server->listen_fd = fd;
    atomic_init(&server->stop, 0);
    return server;
#undef HTTP_FAIL
}

uint16_t h3_http_server_port(const h3_http_server *server) {
    return server ? server->port : 0;
}

void h3_http_stop(h3_http_server *server) {
    if (server) atomic_store(&server->stop, 1);
}

void h3_http_close(h3_http_server *server) {
    if (!server) return;
    if (server->listen_fd >= 0) close(server->listen_fd);
    free(server);
}

/* Read the full request (headers + Content-Length body) into `buffer`.
 * Returns the byte count, or -1 on error / oversize. */
static long read_request(int fd, char **buffer_out) {
    size_t capacity = 8192;
    size_t length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) return -1;
    size_t header_end = 0;
    long content_length = -1;

    for (;;) {
        if (length + 1 >= capacity) {
            if (capacity >= HTTP_MAX_BODY + HTTP_MAX_HEADERS) break;
            size_t grown = capacity * 2;
            char *bigger = realloc(buffer, grown);
            if (!bigger) {
                free(buffer);
                return -1;
            }
            buffer = bigger;
            capacity = grown;
        }
        ssize_t got = read(fd, buffer + length, capacity - length - 1);
        if (got < 0) {
            if (errno == EINTR) continue;
            free(buffer);
            return -1;
        }
        if (got == 0) break;
        length += (size_t)got;
        buffer[length] = '\0';

        if (!header_end) {
            char *marker = strstr(buffer, "\r\n\r\n");
            if (marker) {
                header_end = (size_t)(marker - buffer) + 4;
                if (header_end > HTTP_MAX_HEADERS) {
                    free(buffer);
                    return -1;
                }
                /* case-insensitive scan for Content-Length in the headers */
                for (char *line = buffer; line < marker;) {
                    char *eol = strstr(line, "\r\n");
                    if (!eol || eol > marker) break;
                    if ((size_t)(eol - line) > 15 &&
                        !strncasecmp(line, "Content-Length:", 15)) {
                        content_length = strtol(line + 15, NULL, 10);
                    }
                    line = eol + 2;
                }
            }
        }
        if (header_end) {
            if (content_length < 0) content_length = 0;
            if (content_length > HTTP_MAX_BODY) {
                free(buffer);
                return -1;
            }
            if (length >= header_end + (size_t)content_length) break;
        }
    }
    if (!header_end) {
        free(buffer);
        return -1;
    }
    *buffer_out = buffer;
    return (long)length;
}

static void handle_connection(int fd, h3_http_handler handler, void *user) {
    char *raw = NULL;
    long total = read_request(fd, &raw);
    h3_http_responder responder = {fd, 0, 0, 0};
    if (total < 0) {
        h3_http_send(&responder, 400, "application/json",
                     "{\"error\":{\"message\":\"malformed request\"}}", 41);
        free(raw);
        return;
    }

    char *header_end = strstr(raw, "\r\n\r\n");
    *header_end = '\0';
    char *body = header_end + 4;
    size_t body_length = (size_t)(raw + total - body);
    body[body_length] = '\0';

    char *line_end = strstr(raw, "\r\n");
    if (line_end) *line_end = '\0';
    char *method = raw;
    char *path = strchr(raw, ' ');
    h3_http_request request = {0};
    if (path) {
        *path++ = '\0';
        char *version = strchr(path, ' ');
        if (version) *version = '\0';
        request.method = method;
        request.path = path;
    }
    request.body = body;
    request.body_length = body_length;

    const char *content_type = NULL;
    for (char *line = line_end ? line_end + 2 : raw; line && *line;) {
        char *eol = strstr(line, "\r\n");
        if (eol) *eol = '\0';
        if (!strncasecmp(line, "Content-Type:", 13)) {
            content_type = line + 13;
            while (*content_type == ' ') content_type++;
        }
        if (!eol) break;
        line = eol + 2;
    }
    request.content_type = content_type;

    if (!request.method || !request.path) {
        h3_http_send(&responder, 400, "application/json",
                     "{\"error\":{\"message\":\"bad request line\"}}", 39);
    } else {
        handler(&request, &responder, user);
        if (!responder.started)
            h3_http_send(&responder, 500, "application/json",
                         "{\"error\":{\"message\":\"no response\"}}", 34);
    }
    free(raw);
}

int h3_http_run(h3_http_server *server, h3_http_handler handler, void *user) {
    if (!server || !handler) return 1;
    while (!atomic_load(&server->stop)) {
        struct pollfd waiter = {server->listen_fd, POLLIN, 0};
        int ready = poll(&waiter, 1, 500);
        if (ready <= 0) {
            if (ready < 0 && errno != EINTR) return 1;
            continue; /* timeout or signal: re-check the stop flag */
        }
        struct sockaddr_in peer;
        socklen_t peer_length = sizeof(peer);
        int fd = accept(server->listen_fd, (struct sockaddr *)&peer,
                        &peer_length);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (atomic_load(&server->stop)) break;
            return 1;
        }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        handle_connection(fd, handler, user);
        close(fd);
    }
    return 0;
}
