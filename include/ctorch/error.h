/**
 * @file error.h
 * @brief Error Handling for CTorch
 *
 * The `ct_runtime_error` function is an interface for internal CTorch functions
 * to give runtime error diagnostics alongwith some context of the user code.
 *
 * # Example
 *
 * ```c
 * // main.c
 * #include "ctorch/ctorch.h"
 *
 * int main() {
 *     CT_Tensor *tensor = ct_randn(ct_shape(10, 6), CT_DTYPE_FLOAT,
 *                              CT_NO_GRAD, NULL);
 *     // gives error, cannot multiply matrices with shape 10x6 and 10x6
 *     CT_Tensor *out = ct_matmul(tensor, tensor);
 *     return 0;
 * }
 * ```
 *
 * ```console
 * $ ./main
 * main.c:7:21: [Runtime Error | 104] Shape Mismatch: k1 (6) != k2 (10) in
 * (..., m, k1), (..., k2, n) -> (..., m, n)
 *   5 |                              CT_NO_GRAD, NULL);
 *   6 |     // gives error, cannot multiply matrices with shape 10x6 and 10x6
 *   7 |     CT_Tensor *out = ct_matmul(tensor, tensor);
 *                            ^^^^^^^^^
 * ```
 */

#ifndef CTORCH_ERROR_H
#define CTORCH_ERROR_H

#include <stddef.h>

typedef struct ct_error {
    size_t line;
    char  *file;
    char  *callee_func;
} CT_Error;

extern CT_Error ct_error_state;

#if defined(__GNUC__) || defined(__clang__)
void ct_runtime_error(int status_code, const char *error_msg_fmt, ...)
    __attribute__((format(printf, 2, 3)));
#else
void ct_runtime_error(int status_code, const char *error_msg_fmt, ...);
#endif

#endif /* end of include guard: CTORCH_ERROR_H */
