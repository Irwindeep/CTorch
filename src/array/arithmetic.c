#include "array.h"
#include "ctorch.h"
#include "kernel/array_binops.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

ndArray *array_add(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_add);
}

ndArray *array_sub(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_sub);
}

ndArray *array_mul(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_mul);
}

ndArray *array_div(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_div);
}

ndArray *array_max(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_max);
}

ndArray *array_min(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_min);
}

ndArray *array_gt(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_gt);
}

ndArray *array_ge(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_ge);
}

ndArray *array_lt(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_lt);
}

ndArray *array_le(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_le);
}

ndArray *array_eq(ndArray *arr1, ndArray *arr2) {
    return array_binop_kernel(arr1, arr2, dispatch_eq);
}

ndArray *negative(ndArray *array) {
    size_t *shape = get_shape(array);
    int ndim = get_ndim(array);
    DType dtype = get_dtype(array);

    ndArray *zeros_arr = zeros(ndim, shape, dtype);
    ndArray *result = array_sub(zeros_arr, array);

    free_array(zeros_arr);
    return result;
}

ndArray *inverse(ndArray *array) {
    size_t *shape = get_shape(array);
    int ndim = get_ndim(array);
    DType dtype = get_dtype(array);

    ndArray *ones_arr = ones(ndim, shape, dtype);
    ndArray *result = array_div(ones_arr, array);

    free_array(ones_arr);
    return result;
}

#define _ARRAY_SUM_KERNEL(T, NAME)                                             \
    static void NAME(const T *A, T *B, size_t total_size) {                    \
        T sum = (T)0;                                                          \
                                                                               \
        _Pragma(                                                               \
            "omp parallel for simd reduction(+:sum)                       \
                schedule(static)") for (size_t i = 0; i < total_size; i++) {   \
            sum += A[i];                                                       \
        }                                                                      \
                                                                               \
        B[0] = sum;                                                            \
    }

_ARRAY_SUM_KERNEL(int, _array_sum_i)
_ARRAY_SUM_KERNEL(float, _array_sum_f)
_ARRAY_SUM_KERNEL(double, _array_sum_d)
_ARRAY_SUM_KERNEL(long int, _array_sum_l)

ndArray *array_sum(ndArray *array) {
    DType dtype = get_dtype(array);
    size_t total_size = get_total_size(array);

    ndArray *result = array_init(0, (size_t[]){0}, dtype);

    switch (dtype) {
    case DTYPE_INT: {
        const int *A = get_array_data(array);
        int *B = get_array_data(result);
        _array_sum_i(A, B, total_size);
        break;
    }
    case DTYPE_FLOAT: {
        const float *A = get_array_data(array);
        float *B = get_array_data(result);
        _array_sum_f(A, B, total_size);
        break;
    }
    case DTYPE_DOUBLE: {
        const double *A = get_array_data(array);
        double *B = get_array_data(result);
        _array_sum_d(A, B, total_size);
        break;
    }
    case DTYPE_LONG: {
        const long int *A = get_array_data(array);
        long int *B = get_array_data(result);
        _array_sum_l(A, B, total_size);
        break;
    }
    }

    return result;
}

#define _ARRAY_SUM_DIM_KERNEL(T, NAME)                                          \
    static void NAME(const T *restrict A, T *restrict B, int ndimA, int dim,    \
                     const size_t *shapeA, const size_t *stridesA,              \
                     size_t total_sizeB) {                                      \
        if (total_sizeB == 0)                                                   \
            return;                                                             \
                                                                                \
        /* scalar result (all reduced to one value) */                          \
        if (total_sizeB == 1) {                                                 \
            size_t stride_dim = stridesA[dim];                                  \
            T sum = (T)0;                                                       \
            const T *p = A;                                                     \
            if (stride_dim == 1) {                                              \
                /* contiguous reduce */                                         \
                const T *end = p + shapeA[dim];                                 \
                /* vectorize */                                                 \
                _Pragma("omp simd reduction(+:sum)") for (const T *q = p;       \
                                                          q < end; ++q) sum +=  \
                    *q;                                                         \
            } else {                                                            \
                _Pragma("omp simd reduction(+:sum)") for (size_t k = 0;         \
                                                          k < shapeA[dim];      \
                                                          ++k) sum +=           \
                    p[k * stride_dim];                                          \
            }                                                                   \
            B[0] = sum;                                                         \
            return;                                                             \
        }                                                                       \
                                                                                \
        /* General case: work over each output element independently */         \
        _Pragma("omp parallel for schedule(static)") for (size_t out_idx = 0;   \
                                                          out_idx <             \
                                                          total_sizeB;          \
                                                          ++out_idx) {          \
            /* decode output index into multi-index using repeated div/mod */   \
            /* We keep decoding — still fine for medium dims — but accelerate \
               the inner reduction heavily (vectorized pointer loop). */        \
            size_t tmp = out_idx;                                               \
            size_t offsetA = 0;                                                 \
                                                                                \
            /* map out_idx -> offsetA (skipping dim) */                         \
            for (int dA = ndimA - 1; dA >= 0; --dA) {                           \
                if (dA == dim)                                                  \
                    continue;                                                   \
                size_t idx = tmp % shapeA[dA];                                  \
                tmp /= shapeA[dA];                                              \
                offsetA += idx * stridesA[dA];                                  \
            }                                                                   \
                                                                                \
            T sum = (T)0;                                                       \
            size_t stride_dim = stridesA[dim];                                  \
                                                                                \
            const T *base = A + offsetA;                                        \
            if (stride_dim == 1) {                                              \
                /* contiguous across reduction dimension: best case */          \
                const T *end = base + shapeA[dim];                              \
                _Pragma("omp simd reduction(+:sum)") for (const T *q = base;    \
                                                          q < end; ++q) sum +=  \
                    *q;                                                         \
            } else {                                                            \
                /* non-contiguous: advance pointer by stride_dim */             \
                const T *q = base;                                              \
                _Pragma("omp simd reduction(+:sum)") for (size_t k = 0;         \
                                                          k < shapeA[dim];      \
                                                          ++k) {                \
                    sum += *q;                                                  \
                    q += stride_dim;                                            \
                }                                                               \
            }                                                                   \
                                                                                \
            B[out_idx] = sum;                                                   \
        }                                                                       \
    }

_ARRAY_SUM_DIM_KERNEL(int, _array_sum_dim_i)
_ARRAY_SUM_DIM_KERNEL(float, _array_sum_dim_f)
_ARRAY_SUM_DIM_KERNEL(double, _array_sum_dim_d)
_ARRAY_SUM_DIM_KERNEL(long, _array_sum_dim_l)

ndArray *array_sum_dim(ndArray *array, int dim, bool keepdims) {
    int ndim = get_ndim(array);
    const size_t *shape = get_shape(array);
    DType dtype = get_dtype(array);

    int new_ndim = keepdims ? ndim : ndim - 1;
    size_t new_shape[MAX_NDIM];

    int j = 0;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            if (keepdims)
                new_shape[j++] = 1;
            continue;
        }
        new_shape[j++] = shape[i];
    }

    ndArray *result = zeros(new_ndim, new_shape, dtype);

    size_t totalB = get_total_size(result), itemsize = get_itemsize(array);
    const size_t *strides = get_strides(array);

    size_t stridesA[MAX_NDIM];
    for (int i = 0; i < ndim; i++)
        stridesA[i] = strides[i] / itemsize;

    switch (dtype) {
    case DTYPE_INT: {
        const int *A = get_array_data(array);
        int *B = get_array_data(result);
        _array_sum_dim_i(A, B, ndim, dim, shape, stridesA, totalB);
        break;
    }
    case DTYPE_FLOAT: {
        const float *A = get_array_data(array);
        float *B = get_array_data(result);
        _array_sum_dim_f(A, B, ndim, dim, shape, stridesA, totalB);
        break;
    }
    case DTYPE_DOUBLE: {
        const double *A = get_array_data(array);
        double *B = get_array_data(result);
        _array_sum_dim_d(A, B, ndim, dim, shape, stridesA, totalB);
        break;
    }
    case DTYPE_LONG: {
        const long int *A = get_array_data(array);
        long int *B = get_array_data(result);
        _array_sum_dim_l(A, B, ndim, dim, shape, stridesA, totalB);
        break;
    }
    }

    return result;
}
