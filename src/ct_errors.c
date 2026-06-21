#include <stdarg.h>
#include <stdio.h>

#include "ct_errors.h"
#include "ctorch.h"

void __runtime_error_impl(int error_code, const char *code_name,
                          const char *file_path, const char *func_name,
                          const char *format, ...) {
    fprintf(stderr, "[Error]: Status %d [%s]\n", error_code, code_name);
    fprintf(stderr, "  - [Message]:  ");

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");

    fprintf(stderr, "  - [Location]: %s (%s)\n", func_name, file_path);
    ct_exit(error_code);
}
