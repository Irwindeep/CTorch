#include "error_codes.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    ErrorCode code;
    const char *msg;
} ErrorInfo;

static const ErrorInfo error_table[] = {
    /* array related error codes 10<x> */
    {ARRAY_INIT_FAILURE, "ARRAY_INIT_FAILURE"},
    {INVALID_IDX, "INVALID_IDX"},
    {NON_BROADCASTABLE_ARRAYS, "NON_BROADCASTABLE_ARRAYS"},
    {SHAPE_MISMATCH, "SHAPE_MISMATCH"},
    {INVALID_ARRAY, "INVALID_ARRAY"},
    {INVALID_DTYPE, "INVALID_DTYPE"},
    {REPEATED_ARRAY_DIMS, "REPEATED_ARRAY_DIMS"},
    {INVALID_DIM, "INVALID_DIM"},

    /* tensor related error codes 20<x> */
    {TENSOR_INIT_FAILURE, "TENSOR_INIT_FAILURE"},
    {INVALID_GRAD, "INVALID_GRAD"},
    {DEPENDENCY_ARR_INIT_FAILURE, "DEPENDENCY_ARR_INIT_FAILURE"},
    {ENV_INIT_FAILURE, "ENV_INIT_FAILURE"},
    {ENV_PUSH_FAILURE, "ENV_PUSH_FAILURE"},

    /* autograd related error codes 30<x> */
    {BACKWARD_FN_INIT_FAILURE, "BACKWARD_FN_INIT_FAILURE"},
    {NEXT_FNS_INIT_FAILURE, "NEXT_FNS_INIT_FAILURE"},
    {GRAD_INIT_FAILURE, "GRAD_INIT_FAILURE"},
    {INVALID_BACKWARD_PASS, "INVALID_BACKWARD_PASS"},
    {INVALID_NUM_INPUTS_OUTPUTS, "INVALID_NUM_INPUTS_OUTPUTS"},

    /* random related error codes 40<x> */
    {PRNG_INIT_FAILURE, "PRNG_INIT_FAILURE"},
    {INVALID_LOW_HIGH, "INVALID_LOW_HIGH"},
};

const char *error_code_to_string(ErrorCode code) {
    for (size_t i = 0; i < sizeof(error_table) / sizeof(error_table[0]); ++i) {
        if (error_table[i].code == code)
            return error_table[i].msg;
    }
    return "Unknown error";
}

void print_error(const Error *err) {
    fprintf(stderr,
            "\e[1;31mError Code %d [%s]\e[0m\n"
            "\t\e[1;31mMessage \e[0m: \e[1;36m%s\033[0m\n"
            "\t\e[1;31mLocation\e[0m: \e[1;36m%s:%d\033[0m\n",
            err->code, error_code_to_string(err->code), err->message,
            err->file ? err->file : "(unknown)", err->line);
}

void exit_with_error(const Error *err) {
    print_error(err);
    exit((int)err->code);
}
