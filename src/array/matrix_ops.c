#include "array.h"
#include "error_codes.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _MATMUL_KERNEL(T, NAME)                                                \
    static void NAME(const T *A, const T *B, T *C, size_t m, size_t n,         \
                     size_t k, size_t sAr, size_t sAc, size_t sBr, size_t sBc, \
                     size_t sCr, size_t sCc) {                                 \
        _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < m; \
                                                          i++) {               \
            for (size_t j = 0; j < n; j++) {                                   \
                T sum = (T)0;                                                  \
                for (size_t p = 0; p < k; p++) {                               \
                    sum += A[i * sAr + p * sAc] * B[p * sBr + j * sAc];        \
                }                                                              \
                C[i * sCr + j * sCc] = sum;                                    \
            }                                                                  \
        }                                                                      \
    }

_MATMUL_KERNEL(int, _matmul_i)
_MATMUL_KERNEL(float, _matmul_f)
_MATMUL_KERNEL(double, _matmul_d)
_MATMUL_KERNEL(long int, _matmul_l)

static void _matmul(ndArray *arr1, ndArray *arr2, ndArray *result,
                    const size_t *idx1, const size_t *idx2, const size_t *idx) {
    size_t m = get_shape(result)[get_ndim(result) - 2],
           n = get_shape(result)[get_ndim(result) - 1],
           k = get_shape(arr1)[get_ndim(arr1) - 1];

    DType dtype = get_dtype(result);
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2), ndim = get_ndim(result);

    size_t sAr = get_strides(arr1)[ndim1 - 2] / get_itemsize(arr1),
           sAc = get_strides(arr1)[ndim1 - 1] / get_itemsize(arr1);
    size_t sBr = get_strides(arr2)[ndim2 - 2] / get_itemsize(arr2),
           sBc = get_strides(arr2)[ndim2 - 1] / get_itemsize(arr2);
    size_t sCr = get_strides(result)[ndim - 2] / get_itemsize(result),
           sCc = get_strides(result)[ndim - 1] / get_itemsize(result);

    size_t offsetA = index_to_offset(idx1, get_strides(arr1), ndim1 - 2),
           offsetB = index_to_offset(idx2, get_strides(arr2), ndim2 - 2),
           offsetC = index_to_offset(idx, get_strides(result), ndim - 2);

    switch (dtype) {
    case DTYPE_INT: {
        const int *A = (int *)(get_array_data(arr1) + offsetA),
                  *B = (int *)(get_array_data(arr2) + offsetB);
        int *C = (int *)(get_array_data(result) + offsetC);
        _matmul_i(A, B, C, m, n, k, sAr, sAc, sBr, sBc, sCr, sCc);
        break;
    }
    case DTYPE_FLOAT: {
        const float *A = (float *)(get_array_data(arr1) + offsetA),
                    *B = (float *)(get_array_data(arr2) + offsetB);
        float *C = (float *)(get_array_data(result) + offsetC);
        _matmul_f(A, B, C, m, n, k, sAr, sAc, sBr, sBc, sCr, sCc);
        break;
    }
    case DTYPE_DOUBLE: {
        const double *A = (double *)get_array_data(arr1) + offsetA,
                     *B = (double *)get_array_data(arr2) + offsetB;
        double *C = (double *)get_array_data(result) + offsetC;
        _matmul_d(A, B, C, m, n, k, sAr, sAc, sBr, sBc, sCr, sCc);
        break;
    }
    case DTYPE_LONG: {
        const long int *A = (long int *)get_array_data(arr1) + offsetA,
                       *B = (long int *)get_array_data(arr2) + offsetB;
        long int *C = (long int *)get_array_data(result) + offsetC;
        _matmul_l(A, B, C, m, n, k, sAr, sAc, sBr, sBc, sCr, sCc);
        break;
    }
    }
}

static void _batch_matmul(ndArray *arr1, ndArray *arr2, ndArray *result) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2), ndim = get_ndim(result);
    int batch_ndim1 = ndim1 - 2, batch_ndim2 = ndim2 - 2, batch_ndim = ndim - 2;

    size_t idx1[ndim1], idx2[ndim2], idx[ndim];
    size_t batch_size = 1;

    for (int i = 0; i < batch_ndim; i++)
        batch_size *= get_shape(result)[i];

    for (int b = 0; b < (batch_ndim == 0 ? 1 : batch_size); b++) {
        get_broadcasted_indices(get_shape(arr1), get_shape(arr2),
                                get_shape(result), batch_ndim1, batch_ndim2,
                                batch_ndim, idx1, idx2, idx, b);
        _matmul(arr1, arr2, result, idx1, idx2, idx);
    }
}

/*
(..., m, k), (..., k, n) -> (..., m, n)
*/
ndArray *matmul(ndArray *arr1, ndArray *arr2) {
    if (get_ndim(arr1) < 2 || get_ndim(arr2) < 2)
        RUNTIME_ERROR(INVALID_ARRAY, "matmul requires arrays with ndim >= 2");

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot matmul arrays with dtypes `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);

    dtype = dtype1;

    size_t m = get_shape(arr1)[get_ndim(arr1) - 2],
           k1 = get_shape(arr1)[get_ndim(arr1) - 1],
           k2 = get_shape(arr2)[get_ndim(arr2) - 2],
           n = get_shape(arr2)[get_ndim(arr2) - 1];

    if (k1 != k2)
        RUNTIME_ERRORF(
            SHAPE_MISMATCH,
            "matmul shape mismatch: k1 (%zu) != k2 (%zu) in (..., m, k1), "
            "(..., k2, n) -> (..., m, n)",
            k1, k2);

    int batch_ndim1 = get_ndim(arr1) - 2, batch_ndim2 = get_ndim(arr2) - 2;
    int batch_ndim = (batch_ndim1 > batch_ndim2) ? batch_ndim1 : batch_ndim2;

    size_t batch_shape[batch_ndim];
    broadcast_shape(get_shape(arr1), get_shape(arr2), batch_shape, batch_ndim1,
                    batch_ndim2, batch_ndim);
    size_t shape[batch_ndim + 2];

    for (int i = 0; i < batch_ndim; i++)
        shape[i] = batch_shape[i];

    // final shape = (..., m, n)
    shape[batch_ndim] = m;
    shape[batch_ndim + 1] = n;

    ndArray *result = array_init(batch_ndim + 2, shape, dtype);
    _batch_matmul(arr1, arr2, result);

    return result;
}

static bool _repeated_dims(const int *dims, int ndim) {
    for (int i = 0; i < ndim; i++) {
        for (int j = i + 1; j < ndim; j++) {
            if (dims[i] == dims[j])
                return true;
        }
    }
    return false;
}

ndArray *transpose(ndArray *array, const int *dims) {
    ndArray *result = copy_array(array);
    size_t *shape = get_shape(result);
    size_t *strides = get_strides(result);
    int ndim = get_ndim(result);

    size_t new_shape[ndim], new_strides[ndim];

    if (_repeated_dims(dims, ndim))
        RUNTIME_ERROR(REPEATED_ARRAY_DIMS, "Repeated Array dims");

    for (int d = 0; d < ndim; d++) {
        int dim = dims[d];
        if (dim > ndim)
            RUNTIME_ERRORF(INVALID_DIM, "Invalid dim - %d", dim);

        new_shape[d] = shape[dim];
        new_strides[d] = strides[dim];
    }

    for (int d = 0; d < ndim; d++) {
        shape[d] = new_shape[d];
        strides[d] = new_strides[d];
    }

    return result;
};
