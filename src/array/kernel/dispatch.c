#include "array.h"
#include "array_binops.h"

#include <stddef.h>

#define DISPATCH(dtype, func, arr1, arr2, result, b_strides1, b_strides2,      \
                 strides, ndim, total_size, shape)                             \
    do {                                                                       \
        switch (dtype) {                                                       \
        case DTYPE_INT: {                                                      \
            const int *A = get_array_data(arr1), *B = get_array_data(arr2);    \
            int *C = get_array_data(result);                                   \
            func##_i(A, B, C, b_strides1, b_strides2, strides, ndim,           \
                     total_size, shape);                                       \
            break;                                                             \
        }                                                                      \
        case DTYPE_FLOAT: {                                                    \
            const float *A = get_array_data(arr1), *B = get_array_data(arr2);  \
            float *C = get_array_data(result);                                 \
            func##_f(A, B, C, b_strides1, b_strides2, strides, ndim,           \
                     total_size, shape);                                       \
            break;                                                             \
        }                                                                      \
        case DTYPE_DOUBLE: {                                                   \
            const double *A = get_array_data(arr1), *B = get_array_data(arr2); \
            double *C = get_array_data(result);                                \
            func##_d(A, B, C, b_strides1, b_strides2, strides, ndim,           \
                     total_size, shape);                                       \
            break;                                                             \
        }                                                                      \
        case DTYPE_LONG: {                                                     \
            const long int *A = get_array_data(arr1),                          \
                           *B = get_array_data(arr2);                          \
            long int *C = get_array_data(result);                              \
            func##_l(A, B, C, b_strides1, b_strides2, strides, ndim,           \
                     total_size, shape);                                       \
            break;                                                             \
        }                                                                      \
        }                                                                      \
    } while (0)

#define DISPATCH_FUNC(name, kernel)                                            \
    void dispatch_##name(DType dtype, const ndArray *a, const ndArray *b,      \
                         ndArray *c, const size_t *s1, const size_t *s2,       \
                         const size_t *sc, int n, size_t ts,                   \
                         const size_t *sh) {                                   \
        DISPATCH(dtype, kernel, a, b, c, s1, s2, sc, n, ts, sh);               \
    }

DISPATCH_FUNC(add, _array_add)
DISPATCH_FUNC(sub, _array_sub)
DISPATCH_FUNC(mul, _array_mul)
DISPATCH_FUNC(div, _array_div)
DISPATCH_FUNC(max, _array_max)
DISPATCH_FUNC(min, _array_min)

DISPATCH_FUNC(gt, _array_gt)
DISPATCH_FUNC(ge, _array_ge)
DISPATCH_FUNC(lt, _array_lt)
DISPATCH_FUNC(le, _array_le)
DISPATCH_FUNC(eq, _array_eq)
