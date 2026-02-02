#include "array.h"
#include "error_codes.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define _ARRAY_OP(T, NAME, OP)                                                 \
    static void NAME(const T *A, const T *B, T *C, const size_t *sA,           \
                     const size_t *sB, const size_t *sC, int ndim,             \
                     size_t total_size, const size_t *shapeC) {                \
        _Pragma("omp parallel for schedule(static)") for (size_t i = 0;        \
                                                          i < total_size;      \
                                                          i++) {               \
            size_t tmp = i;                                                    \
            size_t offsetA = 0, offsetB = 0, offsetC = 0;                      \
            for (int d = ndim - 1; d >= 0; d--) {                              \
                size_t idx = tmp % shapeC[d];                                  \
                tmp /= shapeC[d];                                              \
                                                                               \
                offsetA += idx * sA[d];                                        \
                offsetB += idx * sB[d];                                        \
                offsetC += idx * sC[d];                                        \
            }                                                                  \
            C[offsetC] = A[offsetA] OP B[offsetB];                             \
        }                                                                      \
    }

_ARRAY_OP(int, _array_add_i, +)
_ARRAY_OP(float, _array_add_f, +)
_ARRAY_OP(double, _array_add_d, +)
_ARRAY_OP(long int, _array_add_l, +)

_ARRAY_OP(int, _array_sub_i, -)
_ARRAY_OP(float, _array_sub_f, -)
_ARRAY_OP(double, _array_sub_d, -)
_ARRAY_OP(long int, _array_sub_l, -)

_ARRAY_OP(int, _array_mul_i, *)
_ARRAY_OP(float, _array_mul_f, *)
_ARRAY_OP(double, _array_mul_d, *)
_ARRAY_OP(long int, _array_mul_l, *)

_ARRAY_OP(int, _array_div_i, /)
_ARRAY_OP(float, _array_div_f, /)
_ARRAY_OP(double, _array_div_d, /)
_ARRAY_OP(long int, _array_div_l, /)

#define CAT(a, b) a##b

#define DISPATCH(dtype, func, arr1, arr2, result, b_strides1, b_strides2,      \
                 strides, ndim, total_size, shape)                             \
    do {                                                                       \
        switch (dtype) {                                                       \
        case DTYPE_INT: {                                                      \
            const int *A = get_array_data(arr1), *B = get_array_data(arr2);    \
            int *C = get_array_data(result);                                   \
            CAT(func, _i)(A, B, C, b_strides1, b_strides2, strides, ndim,      \
                          total_size, shape);                                  \
            break;                                                             \
        }                                                                      \
        case DTYPE_FLOAT: {                                                    \
            const float *A = get_array_data(arr1), *B = get_array_data(arr2);  \
            float *C = get_array_data(result);                                 \
            CAT(func, _f)(A, B, C, b_strides1, b_strides2, strides, ndim,      \
                          total_size, shape);                                  \
            break;                                                             \
        }                                                                      \
        case DTYPE_DOUBLE: {                                                   \
            const double *A = get_array_data(arr1), *B = get_array_data(arr2); \
            double *C = get_array_data(result);                                \
            CAT(func, _d)(A, B, C, b_strides1, b_strides2, strides, ndim,      \
                          total_size, shape);                                  \
            break;                                                             \
        }                                                                      \
        case DTYPE_LONG: {                                                     \
            const long int *A = get_array_data(arr1),                          \
                           *B = get_array_data(arr2);                          \
            long int *C = get_array_data(result);                              \
            CAT(func, _l)(A, B, C, b_strides1, b_strides2, strides, ndim,      \
                          total_size, shape);                                  \
            break;                                                             \
        }                                                                      \
        }                                                                      \
    } while (0);

ndArray *array_add(ndArray *arr1, ndArray *arr2) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2);
    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot add arrays with dtypes `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);
    dtype = dtype1;

    size_t *shape1 = get_shape(arr1), *shape2 = get_shape(arr2);
    size_t shape[ndim];
    broadcast_shape(shape1, shape2, shape, ndim1, ndim2, ndim);
    ndArray *result = array_init(ndim, shape, dtype);

    size_t *strides1 = get_strides(arr1), *strides2 = get_strides(arr2);
    const size_t *strides = get_strides(result);
    size_t b_strides1[ndim], b_strides2[ndim], sC[ndim];

    broadcasted_strides(b_strides1, strides1, shape1, ndim1, shape, ndim);
    broadcasted_strides(b_strides2, strides2, shape2, ndim2, shape, ndim);

    size_t total_size = get_total_size(result);
    size_t itemsize = get_itemsize(result);
    for (int i = 0; i < ndim; i++) {
        b_strides1[i] /= itemsize;
        b_strides2[i] /= itemsize;
        sC[i] = strides[i] / itemsize;
    }

    DISPATCH(dtype, _array_add, arr1, arr2, result, b_strides1, b_strides2, sC,
             ndim, total_size, shape)

    return result;
}

ndArray *array_sub(ndArray *arr1, ndArray *arr2) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2);
    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot subtract arrays with dtypes `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);
    dtype = dtype1;

    size_t *shape1 = get_shape(arr1), *shape2 = get_shape(arr2);
    size_t shape[ndim];
    broadcast_shape(shape1, shape2, shape, ndim1, ndim2, ndim);
    ndArray *result = array_init(ndim, shape, dtype);

    size_t *strides1 = get_strides(arr1), *strides2 = get_strides(arr2);
    const size_t *strides = get_strides(result);
    size_t b_strides1[ndim], b_strides2[ndim], sC[ndim];

    broadcasted_strides(b_strides1, strides1, shape1, ndim1, shape, ndim);
    broadcasted_strides(b_strides2, strides2, shape2, ndim2, shape, ndim);

    size_t total_size = get_total_size(result);
    size_t itemsize = get_itemsize(result);
    for (int i = 0; i < ndim; i++) {
        b_strides1[i] /= itemsize;
        b_strides2[i] /= itemsize;
        sC[i] = strides[i] / itemsize;
    }

    DISPATCH(dtype, _array_sub, arr1, arr2, result, b_strides1, b_strides2, sC,
             ndim, total_size, shape)

    return result;
}

ndArray *array_mul(ndArray *arr1, ndArray *arr2) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2);
    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot multiply arrays with dtypes `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);
    dtype = dtype1;

    size_t *shape1 = get_shape(arr1), *shape2 = get_shape(arr2);
    size_t shape[ndim];
    broadcast_shape(shape1, shape2, shape, ndim1, ndim2, ndim);
    ndArray *result = array_init(ndim, shape, dtype);

    size_t *strides1 = get_strides(arr1), *strides2 = get_strides(arr2);
    const size_t *strides = get_strides(result);
    size_t b_strides1[ndim], b_strides2[ndim], sC[ndim];

    broadcasted_strides(b_strides1, strides1, shape1, ndim1, shape, ndim);
    broadcasted_strides(b_strides2, strides2, shape2, ndim2, shape, ndim);

    size_t total_size = get_total_size(result);
    size_t itemsize = get_itemsize(result);
    for (int i = 0; i < ndim; i++) {
        b_strides1[i] /= itemsize;
        b_strides2[i] /= itemsize;
        sC[i] = strides[i] / itemsize;
    }

    DISPATCH(dtype, _array_mul, arr1, arr2, result, b_strides1, b_strides2, sC,
             ndim, total_size, shape)

    return result;
}

ndArray *array_div(ndArray *arr1, ndArray *arr2) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2);
    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot divide arrays with dtypes `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);
    dtype = dtype1;

    size_t *shape1 = get_shape(arr1), *shape2 = get_shape(arr2);
    size_t shape[ndim];
    broadcast_shape(shape1, shape2, shape, ndim1, ndim2, ndim);
    ndArray *result = array_init(ndim, shape, dtype);

    size_t *strides1 = get_strides(arr1), *strides2 = get_strides(arr2);
    const size_t *strides = get_strides(result);
    size_t b_strides1[ndim], b_strides2[ndim], sC[ndim];

    broadcasted_strides(b_strides1, strides1, shape1, ndim1, shape, ndim);
    broadcasted_strides(b_strides2, strides2, shape2, ndim2, shape, ndim);

    size_t total_size = get_total_size(result);
    size_t itemsize = get_itemsize(result);
    for (int i = 0; i < ndim; i++) {
        b_strides1[i] /= itemsize;
        b_strides2[i] /= itemsize;
        sC[i] = strides[i] / itemsize;
    }

    DISPATCH(dtype, _array_div, arr1, arr2, result, b_strides1, b_strides2, sC,
             ndim, total_size, shape)

    return result;
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
        _Pragma("omp parallel for reduction(+:sum) schedule(static)") for (    \
            size_t i = 0; i < total_size; i++) {                               \
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

    ndArray *result = array_init(0, (size_t[]){}, dtype);

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

ndArray *array_sum_dim(ndArray *array, int dim, bool keepdims) {
    int ndim = get_ndim(array);
    const size_t *shape = get_shape(array);
    DType dtype = get_dtype(array);

    int new_ndim = keepdims ? ndim : ndim - 1;

    size_t new_shape[new_ndim], in_idx[ndim], out_idx[new_ndim];

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
    size_t total = get_total_size(array);

    for (size_t i = 0; i < total; i++) {
        offset_to_index(i, in_idx, shape, ndim);

        int oi = 0;
        for (int ii = 0; ii < ndim; ii++) {
            if (ii == dim) {
                if (keepdims)
                    out_idx[oi++] = 0;
                continue;
            }
            out_idx[oi++] = in_idx[ii];
        }

        ArrayVal v = get_value(array, in_idx);
        ArrayVal acc = get_value(result, out_idx);
        set_value(result, out_idx, array_val_add(v, acc, dtype));
    }

    return result;
}
