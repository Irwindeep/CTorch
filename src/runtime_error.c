#include <stdarg.h>
#include <stdio.h>

#include "ctorch.h"
#include <ctorch/error.h>

void ct_runtime_error_impl(int status_code, const char *status_msg,
                           const char *file_path, const char *func_name,
                           const char *error_msg_fmt, ...) {
    fprintf(stderr, "[Error]: Status %d [%s]\n", status_code, status_msg);
    fprintf(stderr, "  - [Message]:  ");

    va_list args;
    va_start(args, error_msg_fmt);
    vfprintf(stderr, error_msg_fmt, args);
    va_end(args);

    fprintf(stderr, "\n");

    fprintf(stderr, "  - [Location]: %s (%s)\n", func_name, file_path);
    ct_exit(status_code);
}
