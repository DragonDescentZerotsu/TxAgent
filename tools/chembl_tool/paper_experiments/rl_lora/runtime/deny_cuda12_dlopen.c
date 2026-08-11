#define _GNU_SOURCE

#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

typedef void *(*dlopen_fn)(const char *, int);

/*
 * nvidia-cudnn-frontend 1.23 probes both libcudart.so.12 and .13 and throws
 * when both are installed.  The NeMo worker is a CUDA 13 environment, while
 * node002 also exposes a system CUDA 12.8 runtime through ldconfig.  Reject
 * only the unversioned CUDA 12 probe; all other dlopen calls pass through.
 */
void *dlopen(const char *filename, int flags) {
    dlopen_fn real_dlopen =
        (dlopen_fn)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5");

    if (real_dlopen == NULL) {
        return NULL;
    }
    if (filename != NULL && strcmp(filename, "libcudart.so.12") == 0) {
        /* Use the real loader so dlerror() receives a normal failure string. */
        return real_dlopen(
            "/__txagent_cuda13_only__/libcudart.so.12", flags);
    }
    return real_dlopen(filename, flags);
}
