#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ctorch/error.h"

CT_Error ct_error_state;

#define MAX_BUFFER_SIZE 256
#define ERROR_BUFF_SIZE 1024

static int num_digits(size_t num) {
    int count = 0;
    while (num > 0) {
        num /= 10;
        count++;
    }

    return count;
}

void ct_runtime_error(int status_code, const char *error_msg_fmt, ...) {
    size_t lineno    = ct_error_state.line < 3 ? 1 : ct_error_state.line - 2;
    int    num_lines = (int)(ct_error_state.line - lineno + 1);
    int    nums_len  = num_digits(ct_error_state.line);

    char error_buff[ERROR_BUFF_SIZE];

    va_list args;
    va_start(args, error_msg_fmt);
    vsnprintf(error_buff, ERROR_BUFF_SIZE, error_msg_fmt, args);
    va_end(args);

    FILE *fptr = fopen(ct_error_state.file, "r");
    if (fptr == NULL) {
        fprintf(stderr, "%s:%zu: [Runtime Error | %d] %s\n",
                ct_error_state.file, ct_error_state.line, status_code,
                error_buff);
        fprintf(stderr, "Unable to read %s\n", ct_error_state.file);
        exit(status_code);
    }

    char   buff[MAX_BUFFER_SIZE];
    size_t line = 0;
    while (fgets(buff, MAX_BUFFER_SIZE, fptr)) {
        if (line == lineno - 1)
            break;

        size_t buff_len = strlen(buff);
        if (buff[buff_len - 1] == '\n')
            line++;
    }

    char lines[3][MAX_BUFFER_SIZE];
    for (int i = 0; i < num_lines; i++) {
        strcpy(lines[i], buff);
        if (i != num_lines - 1)
            fgets(buff, MAX_BUFFER_SIZE, fptr);
    }

    int column          = -1;
    int callee_func_len = (int)strlen(ct_error_state.callee_func),
        buff_len        = (int)strlen(buff);

    for (int i = 0; i <= buff_len - callee_func_len; i++) {
        if (memcmp(buff + i, ct_error_state.callee_func,
                   (size_t)callee_func_len) == 0) {
            column = i + 1;
            break;
        }
    }

    if (column > 0) {
        fprintf(stderr, "%s:%zu:%d: [Runtime Error | %d] %s\n",
                ct_error_state.file, ct_error_state.line, column, status_code,
                error_buff);
        for (int i = 0; i < num_lines; i++)
            fprintf(stderr, "  %*zu | %s", nums_len, lineno + (size_t)i,
                    lines[i]);

        fprintf(stderr, "%*s", 5 + column + nums_len, "^");
        for (int i = 0; i < callee_func_len - 1; i++) {
            fputc('^', stderr);
        }
        fputc('\n', stderr);
    } else {
        fprintf(stderr, "%s:%zu: [Runtime Error | %d] %s\n",
                ct_error_state.file, ct_error_state.line, status_code,
                error_buff);
        fprintf(stderr,
                "File %s changed since compilation, cannot reliably provide "
                "error context\n",
                ct_error_state.file);
    }

    fclose(fptr);
    exit(status_code);
}
