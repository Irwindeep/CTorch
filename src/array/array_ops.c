#include "array.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// array comparators
bool array_equal(ndArray *arr1, ndArray *arr2) {
    if (get_dtype(arr1) != get_dtype(arr2))
        return false;
    if (get_ndim(arr1) != get_ndim(arr2))
        return false;

    DType dtype = get_dtype(arr1);
    int ndim = get_ndim(arr1);
    size_t idx[ndim];

    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (get_shape(arr1)[i] != get_shape(arr2)[i])
            return false;
        total *= get_shape(arr1)[i];
    }

    for (size_t i = 0; i < total; i++) {
        size_t tmp = i;
        for (int dim = ndim - 1; dim >= 0; dim--) {
            idx[dim] = tmp % get_shape(arr1)[dim];
            tmp /= get_shape(arr1)[dim];
        }
        if (!array_val_equal(get_value(arr1, idx), get_value(arr2, idx), dtype))
            return false;
    }

    return true;
}

// array operations
bool broadcastable(const size_t *shape1, const size_t *shape2, int ndim1,
                   int ndim2) {
    if (!shape1 || !shape2)
        return false;

    int i = ndim1 - 1, j = ndim2 - 1;

    while (i >= 0 && j >= 0) {
        size_t dim1 = shape1[i];
        size_t dim2 = shape2[j];
        if (!(dim1 == dim2 || dim1 == 1 || dim2 == 1))
            return false;

        i--;
        j--;
    }

    return true;
}

size_t *broadcast_shape(const size_t *shape1, const size_t *shape2, int ndim1,
                        int ndim2) {
    if (!broadcastable(shape1, shape2, ndim1, ndim2)) {
        printf("Dimensions not broadcastable\n");
        exit(NON_BROADCASTABLE_ARRAYS);
    }

    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    size_t *shape = malloc(ndim * sizeof(size_t));

    if (shape == NULL) {
        printf("Failed to initialize array shape\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (int i = 0; i < ndim; i++) {
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        size_t s1 = (idx1 >= 0) ? shape1[idx1] : 1;
        size_t s2 = (idx2 >= 0) ? shape2[idx2] : 1;
        shape[ndim - i - 1] = (s1 > s2) ? s1 : s2;
    }

    return shape;
}

static void _get_broadcasted_indices(const size_t *shape1, const size_t *shape2,
                                     const size_t *shape, int ndim1, int ndim2,
                                     int ndim, size_t *idx1, size_t *idx2,
                                     size_t *idx, size_t batch) {
    size_t tmp = batch;
    for (int d = ndim - 1; d >= 0; d--) {
        idx[d] = tmp % shape[d];
        tmp /= shape[d];

        int dim1 = d - (ndim - ndim1), dim2 = d - (ndim - ndim2);
        if (dim1 >= 0) {
            size_t s1 = shape1[dim1];
            size_t coord1 = (ndim == 0) ? 0 : idx[d];
            idx1[dim1] = (s1 == 1) ? 0 : coord1;
        }
        if (dim2 >= 0) {
            size_t s2 = shape2[dim2];
            size_t coord2 = (ndim == 0) ? 0 : idx[d];
            idx2[dim2] = (s2 == 1) ? 0 : coord2;
        }
    }
}

ndArray *add(ndArray *arr1, ndArray *arr2) {
    int ndim =
        (get_ndim(arr1) > get_ndim(arr2)) ? get_ndim(arr1) : get_ndim(arr2);

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2) {
        printf("Cannot add arrays with dtypes `%s` and `%s`\n",
               DTypeNames[dtype1], DTypeNames[dtype2]);
        exit(INVALID_DTYPE);
    }
    dtype = dtype1;

    size_t *shape = broadcast_shape(get_shape(arr1), get_shape(arr2),
                                    get_ndim(arr1), get_ndim(arr2));
    ndArray *result = array_init(ndim, shape, dtype);

    size_t total = 1;
    for (int i = 0; i < get_ndim(result); i++)
        total *= get_shape(result)[i];

    size_t *idx1 = malloc(get_ndim(arr1) * sizeof(size_t)),
           *idx2 = malloc(get_ndim(arr2) * sizeof(size_t)),
           *idx = malloc(get_ndim(result) * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (size_t b = 0; b < total; b++) {
        _get_broadcasted_indices(
            get_shape(arr1), get_shape(arr2), get_shape(result), get_ndim(arr1),
            get_ndim(arr2), get_ndim(result), idx1, idx2, idx, b);

        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_add(val1, val2, dtype));
    }

    free(shape);
    free(idx1);
    free(idx2);
    free(idx);

    return result;
}

ndArray *sub(ndArray *arr1, ndArray *arr2) {
    int ndim =
        (get_ndim(arr1) > get_ndim(arr2)) ? get_ndim(arr1) : get_ndim(arr2);

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2) {
        printf("Cannot subtract arrays with dtypes `%s` and `%s`\n",
               DTypeNames[dtype1], DTypeNames[dtype2]);
        exit(INVALID_DTYPE);
    }
    dtype = dtype1;

    size_t *shape = broadcast_shape(get_shape(arr1), get_shape(arr2),
                                    get_ndim(arr1), get_ndim(arr2));
    ndArray *result = array_init(ndim, shape, dtype);

    size_t total = 1;
    for (int i = 0; i < get_ndim(result); i++)
        total *= get_shape(result)[i];

    size_t *idx1 = malloc(get_ndim(arr1) * sizeof(size_t)),
           *idx2 = malloc(get_ndim(arr2) * sizeof(size_t)),
           *idx = malloc(get_ndim(result) * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (size_t b = 0; b < total; b++) {
        _get_broadcasted_indices(
            get_shape(arr1), get_shape(arr2), get_shape(result), get_ndim(arr1),
            get_ndim(arr2), get_ndim(result), idx1, idx2, idx, b);

        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_sub(val1, val2, dtype));
    }

    free(shape);
    free(idx1);
    free(idx2);
    free(idx);

    return result;
}

ndArray *mul(ndArray *arr1, ndArray *arr2) {
    int ndim =
        (get_ndim(arr1) > get_ndim(arr2)) ? get_ndim(arr1) : get_ndim(arr2);

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2) {
        printf("Cannot multiply arrays with dtypes `%s` and `%s`\n",
               DTypeNames[dtype1], DTypeNames[dtype2]);
        exit(INVALID_DTYPE);
    }
    dtype = dtype1;

    size_t *shape = broadcast_shape(get_shape(arr1), get_shape(arr2),
                                    get_ndim(arr1), get_ndim(arr2));
    ndArray *result = array_init(ndim, shape, dtype);

    size_t total = 1;
    for (int i = 0; i < get_ndim(result); i++)
        total *= get_shape(result)[i];

    size_t *idx1 = malloc(get_ndim(arr1) * sizeof(size_t)),
           *idx2 = malloc(get_ndim(arr2) * sizeof(size_t)),
           *idx = malloc(get_ndim(result) * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (size_t b = 0; b < total; b++) {
        _get_broadcasted_indices(
            get_shape(arr1), get_shape(arr2), get_shape(result), get_ndim(arr1),
            get_ndim(arr2), get_ndim(result), idx1, idx2, idx, b);

        ArrayVal val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, array_val_mul(val1, val2, dtype));
    }

    free(shape);
    free(idx1);
    free(idx2);
    free(idx);

    return result;
}

static void _matmul(ndArray *arr1, ndArray *arr2, ndArray *result, size_t *idx1,
                    size_t *idx2, size_t *idx) {
    size_t m = get_shape(result)[get_ndim(result) - 2],
           n = get_shape(result)[get_ndim(result) - 1],
           k = get_shape(arr1)[get_ndim(arr1) - 1];

    DType dtype = get_dtype(result);
    for (size_t i = 0; i < m; i++) {
        idx1[get_ndim(arr1) - 2] = i;
        idx[get_ndim(result) - 2] = i;
        for (size_t j = 0; j < n; j++) {
            idx2[get_ndim(arr2) - 1] = j;
            idx[get_ndim(result) - 1] = j;

            ArrayVal sum = array_val_zero(dtype);
            for (size_t p = 0; p < k; p++) {
                idx1[get_ndim(arr1) - 1] = p;
                idx2[get_ndim(arr2) - 2] = p;

                ArrayVal val1 = get_value(arr1, idx1),
                         val2 = get_value(arr2, idx2);

                ArrayVal value = array_val_mul(val1, val2, dtype);
                sum = array_val_add(sum, value, dtype);
            }
            set_value(result, idx, sum);
        }
    }
}

static void _batch_matmul(ndArray *arr1, ndArray *arr2, ndArray *result) {
    int batch_ndim1 = get_ndim(arr1) - 2, batch_ndim2 = get_ndim(arr2) - 2,
        batch_ndim = get_ndim(result) - 2;

    size_t *idx1 = malloc(get_ndim(arr1) * sizeof(size_t)),
           *idx2 = malloc(get_ndim(arr2) * sizeof(size_t)),
           *idx = malloc(get_ndim(result) * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    size_t batch_size = 1;
    for (int i = 0; i < batch_ndim; i++)
        batch_size *= get_shape(result)[i];

    for (int b = 0; b < (batch_ndim == 0 ? 1 : batch_size); b++) {
        _get_broadcasted_indices(get_shape(arr1), get_shape(arr2),
                                 get_shape(result), batch_ndim1, batch_ndim2,
                                 batch_ndim, idx1, idx2, idx, b);
        _matmul(arr1, arr2, result, idx1, idx2, idx);
    }

    free(idx1);
    free(idx2);
    free(idx);
}

/*
(..., m, k), (..., k, n) -> (..., m, n)
*/
ndArray *matmul(ndArray *arr1, ndArray *arr2) {
    if (get_ndim(arr1) < 2 || get_ndim(arr2) < 2) {
        printf("matmul requires arrays with ndim >= 2\n");
        exit(INVALID_ARRAY);
    }

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2) {
        printf("Cannot matmul arrays with dtypes `%s` and `%s`\n",
               DTypeNames[dtype1], DTypeNames[dtype2]);
        exit(INVALID_DTYPE);
    }
    dtype = dtype1;

    size_t m = get_shape(arr1)[get_ndim(arr1) - 2],
           k1 = get_shape(arr1)[get_ndim(arr1) - 1],
           k2 = get_shape(arr2)[get_ndim(arr2) - 2],
           n = get_shape(arr2)[get_ndim(arr2) - 1];

    if (k1 != k2) {
        printf("matmul shape mismatch: k1 (%zu) != k2 (%zu) in (..., m, k1), "
               "(..., k2, n) -> (..., m, n)\n",
               k1, k2);
        exit(SHAPE_MISMATCH);
    }

    int batch_ndim1 = get_ndim(arr1) - 2, batch_ndim2 = get_ndim(arr2) - 2;
    int batch_ndim = (batch_ndim1 > batch_ndim2) ? batch_ndim1 : batch_ndim2;

    size_t *batch_shape = broadcast_shape(get_shape(arr1), get_shape(arr2),
                                          get_ndim(arr1), get_ndim(arr2));

    size_t *shape = malloc((batch_ndim + 2) * sizeof(size_t));
    if (!shape) {
        free(batch_shape);
        printf("Failed to allocate output shape\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (int i = 0; i < batch_ndim; i++)
        shape[i] = batch_shape[i];

    // final shape = (..., m, n)
    shape[batch_ndim] = m;
    shape[batch_ndim + 1] = n;

    ndArray *result = array_init(batch_ndim + 2, shape, dtype);
    _batch_matmul(arr1, arr2, result);

    free(shape);
    free(batch_shape);

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

void transpose(ndArray *array, const int *dims) {
    size_t *shape = get_shape(array);
    size_t *strides = get_strides(array);
    int ndim = get_ndim(array);

    size_t new_shape[ndim], new_strides[ndim];

    if (_repeated_dims(dims, ndim)) {
        printf("Repeated Array dims\n");
        exit(REPEATED_ARRAY_DIMS);
    }

    for (int d = 0; d < ndim; d++) {
        int dim = dims[d];
        if (dim > ndim) {
            printf("Invalid dim - %d\n", dim);
            exit(INVALID_DIM);
        }

        new_shape[d] = shape[dim];
        new_strides[d] = strides[dim];
    }

    for (int d = 0; d < ndim; d++) {
        shape[d] = new_shape[d];
        strides[d] = new_strides[d];
    }
};

ndArray *array_sum(ndArray *array) {
    int ndim = get_ndim(array);
    DType dtype = get_dtype(array);
    size_t total = get_total_size(array);

    const size_t *shape = get_shape(array);

    size_t *indices = malloc(ndim * sizeof(size_t));
    if (!indices) {
        printf("Failure to create index buffer\n");
        exit(ARRAY_INIT_FAILURE);
    }

    ArrayVal sum = array_val_zero(dtype);
    for (size_t i = 0; i < total; i++) {
        size_t tmp = i;
        for (int d = 0; d < ndim; d++) {
            indices[d] = tmp % shape[d];
            tmp /= shape[d];
        }
        ArrayVal val = get_value(array, indices);
        sum = array_val_add(sum, val, dtype);
    }

    ndArray *result = array_init(0, (size_t[]){}, dtype);
    set_value(result, (size_t[]){}, sum);

    free(indices);
    return result;
}

ndArray *array_sum_dim(ndArray *array, int dim, bool keepdims) {
    int ndim = get_ndim(array);
    const size_t *shape = get_shape(array);
    DType dtype = get_dtype(array);

    int new_ndim = keepdims ? ndim : ndim - 1;

    size_t *new_shape = malloc(new_ndim * sizeof(size_t)),
           *in_idx = malloc(ndim * sizeof(size_t)),
           *out_idx = malloc(new_ndim * sizeof(size_t));
    if (!new_shape || !in_idx || !out_idx) {
        printf("Failure to create new array\n");
        exit(ARRAY_INIT_FAILURE);
    }
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

    free(new_shape);
    free(in_idx);
    free(out_idx);
    return result;
}
