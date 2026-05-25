/**************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Thin delegate wrapper for rpd_tracer multi-node profiling.
 *
 * Loads librpd_tracer.so into this process (becoming the per-node delegate
 * for clock sync and log aggregation), then forks and execs the user command
 * with LD_PRELOAD so child processes are traced. The delegate outlives the
 * child, providing service continuity for workers.
 *
 * Usage: rpdrun [--exit-delay SECONDS] command [args...]
 **************************************************************************/
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t g_child = -1;

static void forward_signal(int sig) {
    if (g_child > 0)
        kill(g_child, sig);
}

int main(int argc, char **argv) {
    int exit_delay = 0;
    int argi = 1;

    if (argi < argc && strcmp(argv[argi], "--exit-delay") == 0) {
        if (argi + 1 < argc) {
            exit_delay = atoi(argv[argi + 1]);
            argi += 2;
        }
    }

    if (argi >= argc) {
        fprintf(stderr, "Usage: rpdrun [--exit-delay SECONDS] command [args...]\n");
        return 1;
    }

    /* Save user's settings */
    const char *saved_autostart = getenv("RPDT_AUTOSTART");
    const char *saved_delayinit = getenv("RPDT_DELAYINIT");
    char autostart_buf[16] = {};
    char delayinit_buf[16] = {};
    if (saved_autostart)
        strncpy(autostart_buf, saved_autostart, sizeof(autostart_buf) - 1);
    if (saved_delayinit)
        strncpy(delayinit_buf, saved_delayinit, sizeof(delayinit_buf) - 1);

    /* Delegate mode: create singleton, start services, don't trace */
    setenv("RPDT_AUTOSTART", "0", 1);
    setenv("RPDT_DELAYINIT", "0", 1);
    setenv("RPDT_QUIET", "1", 1);

    if (dlopen("librpd_tracer.so", RTLD_NOW) == NULL) {
        fprintf(stderr, "rpdrun: failed to load librpd_tracer.so: %s\n", dlerror());
        return 1;
    }

    /* Restore user's settings for child processes */
    if (saved_autostart)
        setenv("RPDT_AUTOSTART", autostart_buf, 1);
    else
        unsetenv("RPDT_AUTOSTART");
    if (saved_delayinit)
        setenv("RPDT_DELAYINIT", delayinit_buf, 1);
    else
        unsetenv("RPDT_DELAYINIT");

    setenv("LD_PRELOAD", "librpd_tracer.so", 1);

    /* Forward signals to child */
    signal(SIGINT, forward_signal);
    signal(SIGTERM, forward_signal);

    g_child = fork();
    if (g_child < 0) {
        perror("rpdrun: fork");
        return 1;
    }

    if (g_child == 0) {
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        execvp(argv[argi], &argv[argi]);
        perror("rpdrun: exec");
        return 1;
    }

    int status = 0;
    waitpid(g_child, &status, 0);

    if (exit_delay > 0) {
        fprintf(stderr, "rpdrun: waiting %ds for remote nodes to flush\n", exit_delay);
        sleep(exit_delay);
    }

    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
