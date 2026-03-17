#include "array_binops.h"

#define ARRAY_BINOP_KERNEL(T, NAME, OP)                                        \
    void NAME(const T *A, const T *B, T *C, const size_t *sA,                  \
              const size_t *sB, const size_t *sC, int ndim, size_t total_size, \
              const size_t *shapeC) {                                          \
        if (total_size == 0)                                                   \
            return;                                                            \
                                                                               \
        if (ndim == 0) {                                                       \
            C[0] = OP(A[0], B[0], T);                                          \
            return;                                                            \
        }                                                                      \
                                                                               \
        size_t inner = shapeC[ndim - 1];                                       \
        size_t outer = total_size / inner;                                     \
        _Pragma("omp parallel for schedule(static)") for (size_t i = 0;        \
                                                          i < outer; i++) {    \
            size_t tmp = i;                                                    \
            size_t offsetA = 0, offsetB = 0, offsetC = 0;                      \
            for (int d = ndim - 2; d >= 0; d--) {                              \
                size_t idx = tmp % shapeC[d];                                  \
                tmp /= shapeC[d];                                              \
                offsetA += idx * sA[d];                                        \
                offsetB += idx * sB[d];                                        \
                offsetC += idx * sC[d];                                        \
            }                                                                  \
            const T *a = A + offsetA;                                          \
            const T *b = B + offsetB;                                          \
            T *c = C + offsetC;                                                \
                                                                               \
            size_t sa = sA[ndim - 1];                                          \
            size_t sb = sB[ndim - 1];                                          \
            size_t sc = sC[ndim - 1];                                          \
                                                                               \
            for (size_t j = 0; j < inner; j++) {                               \
                c[j * sc] = OP(a[j * sa], b[j * sb], T);                       \
            }                                                                  \
        }                                                                      \
    }

ARRAY_BINOP_KERNEL(int, _array_add_i, OP_ADD)
ARRAY_BINOP_KERNEL(float, _array_add_f, OP_ADD)
ARRAY_BINOP_KERNEL(double, _array_add_d, OP_ADD)
ARRAY_BINOP_KERNEL(long int, _array_add_l, OP_ADD)

ARRAY_BINOP_KERNEL(int, _array_sub_i, OP_SUB)
ARRAY_BINOP_KERNEL(float, _array_sub_f, OP_SUB)
ARRAY_BINOP_KERNEL(double, _array_sub_d, OP_SUB)
ARRAY_BINOP_KERNEL(long int, _array_sub_l, OP_SUB)

ARRAY_BINOP_KERNEL(int, _array_mul_i, OP_MUL)
ARRAY_BINOP_KERNEL(float, _array_mul_f, OP_MUL)
ARRAY_BINOP_KERNEL(double, _array_mul_d, OP_MUL)
ARRAY_BINOP_KERNEL(long int, _array_mul_l, OP_MUL)

ARRAY_BINOP_KERNEL(int, _array_div_i, OP_DIV)
ARRAY_BINOP_KERNEL(float, _array_div_f, OP_DIV)
ARRAY_BINOP_KERNEL(double, _array_div_d, OP_DIV)
ARRAY_BINOP_KERNEL(long int, _array_div_l, OP_DIV)

ARRAY_BINOP_KERNEL(int, _array_max_i, OP_MAX)
ARRAY_BINOP_KERNEL(float, _array_max_f, OP_MAX)
ARRAY_BINOP_KERNEL(double, _array_max_d, OP_MAX)
ARRAY_BINOP_KERNEL(long int, _array_max_l, OP_MAX)

ARRAY_BINOP_KERNEL(int, _array_min_i, OP_MIN)
ARRAY_BINOP_KERNEL(float, _array_min_f, OP_MIN)
ARRAY_BINOP_KERNEL(double, _array_min_d, OP_MIN)
ARRAY_BINOP_KERNEL(long int, _array_min_l, OP_MIN)

ARRAY_BINOP_KERNEL(int, _array_gt_i, OP_GT)
ARRAY_BINOP_KERNEL(float, _array_gt_f, OP_GT)
ARRAY_BINOP_KERNEL(double, _array_gt_d, OP_GT)
ARRAY_BINOP_KERNEL(long int, _array_gt_l, OP_GT)

ARRAY_BINOP_KERNEL(int, _array_ge_i, OP_GE)
ARRAY_BINOP_KERNEL(float, _array_ge_f, OP_GE)
ARRAY_BINOP_KERNEL(double, _array_ge_d, OP_GE)
ARRAY_BINOP_KERNEL(long int, _array_ge_l, OP_GE)

ARRAY_BINOP_KERNEL(int, _array_lt_i, OP_LT)
ARRAY_BINOP_KERNEL(float, _array_lt_f, OP_LT)
ARRAY_BINOP_KERNEL(double, _array_lt_d, OP_LT)
ARRAY_BINOP_KERNEL(long int, _array_lt_l, OP_LT)

ARRAY_BINOP_KERNEL(int, _array_le_i, OP_LE)
ARRAY_BINOP_KERNEL(float, _array_le_f, OP_LE)
ARRAY_BINOP_KERNEL(double, _array_le_d, OP_LE)
ARRAY_BINOP_KERNEL(long int, _array_le_l, OP_LE)

ARRAY_BINOP_KERNEL(int, _array_eq_i, OP_EQ)
ARRAY_BINOP_KERNEL(float, _array_eq_f, OP_EQ)
ARRAY_BINOP_KERNEL(double, _array_eq_d, OP_EQ)
ARRAY_BINOP_KERNEL(long int, _array_eq_l, OP_EQ)
