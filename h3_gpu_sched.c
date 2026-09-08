#include "h3_gpu_sched.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static int g_enabled = 1;
/* Cap on one yield-point wait. A chat GPU op that has the device to itself
 * finishes fast (a decode token ~0.3 s) and releases the diffusion thread via
 * the condvar well before the cap; the cap only bounds pathological cases so
 * diffusion is never starved. 1 s balances chat responsiveness against video
 * slowdown -- P8-MEM-01 measured p50/p95 inter-token back to native at 1 s,
 * with ~14 % video slowdown; a 6 s cap did not improve things further. */
static long g_max_ms = 1000;
static unsigned g_every = 1;

static unsigned g_waiters;            /* chat GPU work units in flight */
static unsigned long g_yield_count;   /* diffusion actually parked */
static double g_yield_seconds;

static void configure(void) {
    const char *disable = getenv("H3_GPU_SCHED");
    g_enabled = !(disable && !strcmp(disable, "0"));
    const char *max_ms = getenv("H3_GPU_SCHED_MAX_MS");
    if (max_ms) {
        long v = atol(max_ms);
        if (v >= 0 && v <= 10000) g_max_ms = v;
    }
    const char *every = getenv("H3_GPU_SCHED_EVERY");
    if (every) {
        long v = atol(every);
        if (v >= 1 && v <= 1000) g_every = (unsigned)v;
    }
}

int h3_gpu_sched_enabled(void) {
    pthread_once(&g_once, configure);
    return g_enabled;
}

void h3_gpu_sched_set_enabled(int on) {
    pthread_once(&g_once, configure);
    pthread_mutex_lock(&g_mu);
    g_enabled = on ? 1 : 0;
    if (!g_enabled && g_waiters) {
        g_waiters = 0;
        pthread_cond_broadcast(&g_cv);
    }
    pthread_mutex_unlock(&g_mu);
}

void h3_gpu_sched_chat_enter(void) {
    pthread_once(&g_once, configure);
    if (!g_enabled) return;
    pthread_mutex_lock(&g_mu);
    g_waiters++;
    pthread_mutex_unlock(&g_mu);
}

void h3_gpu_sched_chat_leave(void) {
    pthread_once(&g_once, configure);
    if (!g_enabled) return;
    pthread_mutex_lock(&g_mu);
    if (g_waiters) g_waiters--;
    if (!g_waiters) pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

void h3_gpu_sched_diffusion_yield_point(unsigned block_index) {
    pthread_once(&g_once, configure);
    if (!g_enabled) return;
    if (g_every > 1 && (block_index % g_every) != 0) return;

    pthread_mutex_lock(&g_mu);
    if (g_waiters) {
        double started = now_seconds();
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += g_max_ms / 1000;
        deadline.tv_nsec += (g_max_ms % 1000) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec += 1;
            deadline.tv_nsec -= 1000000000L;
        }
        while (g_waiters) {
            if (pthread_cond_timedwait(&g_cv, &g_mu, &deadline) == ETIMEDOUT)
                break;
        }
        g_yield_count++;
        g_yield_seconds += now_seconds() - started;
    }
    pthread_mutex_unlock(&g_mu);
}

unsigned long h3_gpu_sched_yield_count(void) {
    pthread_mutex_lock(&g_mu);
    unsigned long v = g_yield_count;
    pthread_mutex_unlock(&g_mu);
    return v;
}

double h3_gpu_sched_yield_seconds(void) {
    pthread_mutex_lock(&g_mu);
    double v = g_yield_seconds;
    pthread_mutex_unlock(&g_mu);
    return v;
}

void h3_gpu_sched_reset_stats(void) {
    pthread_mutex_lock(&g_mu);
    g_yield_count = 0;
    g_yield_seconds = 0.0;
    pthread_mutex_unlock(&g_mu);
}
