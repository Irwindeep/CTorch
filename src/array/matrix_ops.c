#include "array.h"
#include "error_codes.h"
#include "kernel/ops.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        matmul_kernel(arr1, arr2, result, idx1, idx2, idx);
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

    if (dtype == DTYPE_INT || dtype == DTYPE_LONG)
        RUNTIME_ERRORF(INVALID_DTYPE,
                       "Cannot matmul arrays with dtype - `%s`, only "
                       "DTYPE_FLOAT and DTYPE_DOUBLE are supported",
                       DTypeNames[dtype]);

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
