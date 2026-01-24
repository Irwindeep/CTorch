#include "array.h"
#include "error_codes.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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
    size_t *shape = broadcast_shape(shape1, shape2, ndim1, ndim2);

    ndArray *result = array_init(ndim, shape, dtype);

    size_t total_size = get_total_size(result);
    size_t idx1[ndim1], idx2[ndim2], idx[ndim];
    for (size_t b = 0; b < total_size; b++) {
        get_broadcasted_indices(shape1, shape2, shape, ndim1, ndim2, ndim, idx1,
                                idx2, idx, b);

        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_add(val1, val2, dtype));
    }

    free(shape);
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
    size_t *shape = broadcast_shape(shape1, shape2, ndim1, ndim2);

    ndArray *result = array_init(ndim, shape, dtype);

    size_t total_size = get_total_size(result);
    size_t idx1[ndim1], idx2[ndim2], idx[ndim];
    for (size_t b = 0; b < total_size; b++) {
        get_broadcasted_indices(shape1, shape2, shape, ndim1, ndim2, ndim, idx1,
                                idx2, idx, b);

        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_sub(val1, val2, dtype));
    }

    free(shape);
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
    size_t *shape = broadcast_shape(shape1, shape2, ndim1, ndim2);

    ndArray *result = array_init(ndim, shape, dtype);

    size_t total_size = get_total_size(result);
    size_t idx1[ndim1], idx2[ndim2], idx[ndim];
    for (size_t b = 0; b < total_size; b++) {
        get_broadcasted_indices(shape1, shape2, shape, ndim1, ndim2, ndim, idx1,
                                idx2, idx, b);
        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_mul(val1, val2, dtype));
    }

    free(shape);
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
    size_t *shape = broadcast_shape(shape1, shape2, ndim1, ndim2);

    ndArray *result = array_init(ndim, shape, dtype);

    size_t total_size = get_total_size(result);
    size_t idx1[ndim1], idx2[ndim2], idx[ndim];
    for (size_t b = 0; b < total_size; b++) {
        get_broadcasted_indices(shape1, shape2, shape, ndim1, ndim2, ndim, idx1,
                                idx2, idx, b);
        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_div(val1, val2, dtype));
    }

    free(shape);
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

ndArray *array_sum(ndArray *array) {
    int ndim = get_ndim(array);
    DType dtype = get_dtype(array);
    size_t total = get_total_size(array);

    const size_t *shape = get_shape(array);
    size_t indices[ndim];

    ArrayVal sum = array_val_zero(dtype);
    for (size_t i = 0; i < total; i++) {
        offset_to_index(i, indices, shape, ndim);

        ArrayVal val = get_value(array, indices);
        sum = array_val_add(sum, val, dtype);
    }

    ndArray *result = array_init(0, (size_t[]){}, dtype);
    set_value(result, (size_t[]){}, sum);

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
