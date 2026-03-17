#ifndef KERNEL_ARRAY_BINOPS_H
#define KERNEL_ARRAY_BINOPS_H

#include "array.h"
#include "ctorch.h"

#include <stddef.h>

typedef void (*dispatcher)(DType dtype, const ndArray *arr1,
                           const ndArray *arr2, ndArray *result,
                           const size_t *b_strides1, const size_t *b_strides2,
                           const size_t *strides, int ndim, size_t total_size,
                           const size_t *shape);

ndArray *array_binop_kernel(ndArray *arr1, ndArray *arr2, dispatcher dispach);

#define OP_ADD(a, b, T) ((a) + (b))
#define OP_SUB(a, b, T) ((a) - (b))
#define OP_MUL(a, b, T) ((a) * (b))
#define OP_DIV(a, b, T) ((a) / (b))
#define OP_MAX(a, b, T) ((a) > (b) ? (a) : (b))
#define OP_MIN(a, b, T) ((a) < (b) ? (a) : (b))

#define OP_GT(a, b, T) ((a) > (b)) ? (T)1 : (T)0;
#define OP_GE(a, b, T) ((a) >= (b)) ? (T)1 : (T)0;
#define OP_LT(a, b, T) ((a) < (b)) ? (T)1 : (T)0;
#define OP_LE(a, b, T) ((a) <= (b)) ? (T)1 : (T)0;
#define OP_EQ(a, b, T) ((a) == (b)) ? (T)1 : (T)0;

#define DEFINE_KERNEL(T, NAME)                                                 \
    void NAME(const T *A, const T *B, T *C, const size_t *sA,                  \
              const size_t *sB, const size_t *sC, int ndim, size_t total_size, \
              const size_t *shapeC);

DEFINE_KERNEL(int, _array_add_i)
DEFINE_KERNEL(float, _array_add_f)
DEFINE_KERNEL(double, _array_add_d)
DEFINE_KERNEL(long int, _array_add_l)

DEFINE_KERNEL(int, _array_sub_i)
DEFINE_KERNEL(float, _array_sub_f)
DEFINE_KERNEL(double, _array_sub_d)
DEFINE_KERNEL(long int, _array_sub_l)

DEFINE_KERNEL(int, _array_mul_i)
DEFINE_KERNEL(float, _array_mul_f)
DEFINE_KERNEL(double, _array_mul_d)
DEFINE_KERNEL(long int, _array_mul_l)

DEFINE_KERNEL(int, _array_div_i)
DEFINE_KERNEL(float, _array_div_f)
DEFINE_KERNEL(double, _array_div_d)
DEFINE_KERNEL(long int, _array_div_l)

DEFINE_KERNEL(int, _array_max_i)
DEFINE_KERNEL(float, _array_max_f)
DEFINE_KERNEL(double, _array_max_d)
DEFINE_KERNEL(long int, _array_max_l)

DEFINE_KERNEL(int, _array_min_i)
DEFINE_KERNEL(float, _array_min_f)
DEFINE_KERNEL(double, _array_min_d)
DEFINE_KERNEL(long int, _array_min_l)

DEFINE_KERNEL(int, _array_gt_i)
DEFINE_KERNEL(float, _array_gt_f)
DEFINE_KERNEL(double, _array_gt_d)
DEFINE_KERNEL(long int, _array_gt_l)

DEFINE_KERNEL(int, _array_ge_i)
DEFINE_KERNEL(float, _array_ge_f)
DEFINE_KERNEL(double, _array_ge_d)
DEFINE_KERNEL(long int, _array_ge_l)

DEFINE_KERNEL(int, _array_lt_i)
DEFINE_KERNEL(float, _array_lt_f)
DEFINE_KERNEL(double, _array_lt_d)
DEFINE_KERNEL(long int, _array_lt_l)

DEFINE_KERNEL(int, _array_le_i)
DEFINE_KERNEL(float, _array_le_f)
DEFINE_KERNEL(double, _array_le_d)
DEFINE_KERNEL(long int, _array_le_l)

DEFINE_KERNEL(int, _array_eq_i)
DEFINE_KERNEL(float, _array_eq_f)
DEFINE_KERNEL(double, _array_eq_d)
DEFINE_KERNEL(long int, _array_eq_l)

#define DEFINE_DISPATCH_FUNC(name, kernel)                                     \
    void dispatch_##name(DType dtype, const ndArray *a, const ndArray *b,      \
                         ndArray *c, const size_t *s1, const size_t *s2,       \
                         const size_t *sc, int n, size_t ts,                   \
                         const size_t *sh);

DEFINE_DISPATCH_FUNC(add, _array_add)
DEFINE_DISPATCH_FUNC(sub, _array_sub)
DEFINE_DISPATCH_FUNC(mul, _array_mul)
DEFINE_DISPATCH_FUNC(div, _array_div)
DEFINE_DISPATCH_FUNC(max, _array_max)
DEFINE_DISPATCH_FUNC(min, _array_min)

DEFINE_DISPATCH_FUNC(gt, _array_gt)
DEFINE_DISPATCH_FUNC(ge, _array_ge)
DEFINE_DISPATCH_FUNC(lt, _array_lt)
DEFINE_DISPATCH_FUNC(le, _array_le)
DEFINE_DISPATCH_FUNC(eq, _array_eq)

#endif // !KERNEL_ARRAY_BINOPS_H
