#include "array/array_ops.h"
#include "array/array.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// array comparators
bool array_equal(ndArray *arr1, ndArray *arr2) {
    if (arr1->ndim != arr2->ndim)
        return false;

    int ndim = arr1->ndim;
    size_t *idx = malloc(ndim * sizeof(size_t));
    if (!idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (arr1->shape[i] != arr2->shape[i])
            return false;
        total *= arr1->shape[i];
    }

    for (size_t i = 0; i < total; i++) {
        size_t tmp = i;
        for (int dim = ndim - 1; dim >= 0; dim--) {
            idx[dim] = tmp % arr1->shape[dim];
            tmp /= arr1->shape[dim];
        }
        if (get_value(arr1, idx) != get_value(arr2, idx))
            return false;
    }

    free(idx);
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
    int ndim = (arr1->ndim > arr2->ndim) ? arr1->ndim : arr2->ndim;
    size_t *shape =
        broadcast_shape(arr1->shape, arr2->shape, arr1->ndim, arr2->ndim);
    ndArray *result = array_init(ndim, shape, sizeof(float));

    size_t total = 1;
    for (int i = 0; i < result->ndim; i++)
        total *= result->shape[i];

    size_t *idx1 = malloc(arr1->ndim * sizeof(size_t)),
           *idx2 = malloc(arr2->ndim * sizeof(size_t)),
           *idx = malloc(result->ndim * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (size_t b = 0; b < total; b++) {
        _get_broadcasted_indices(arr1->shape, arr2->shape, result->shape,
                                 arr1->ndim, arr2->ndim, result->ndim, idx1,
                                 idx2, idx, b);

        float val1 = get_value(arr1, idx1), val2 = get_value(arr2, idx2);
        set_value(result, idx, val1 + val2);
    }

    free(shape);
    free(idx1);
    free(idx2);
    free(idx);

    return result;
}

static void _matmul(ndArray *arr1, ndArray *arr2, ndArray *result, size_t *idx1,
                    size_t *idx2, size_t *idx) {
    size_t m = result->shape[result->ndim - 2],
           n = result->shape[result->ndim - 1], k = arr1->shape[arr1->ndim - 1];

    for (size_t i = 0; i < m; i++) {
        idx1[arr1->ndim - 2] = i;
        idx[result->ndim - 2] = i;
        for (size_t j = 0; j < n; j++) {
            idx2[arr2->ndim - 1] = j;
            idx[result->ndim - 1] = j;

            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                idx1[arr1->ndim - 1] = p;
                idx2[arr2->ndim - 2] = p;

                float val1 = get_value(arr1, idx1),
                      val2 = get_value(arr2, idx2);

                sum += val1 * val2;
            }
            set_value(result, idx, sum);
        }
    }
}

static void _batch_matmul(ndArray *arr1, ndArray *arr2, ndArray *result) {
    int batch_ndim1 = arr1->ndim - 2, batch_ndim2 = arr2->ndim - 2,
        batch_ndim = result->ndim - 2;

    size_t *idx1 = malloc(arr1->ndim * sizeof(size_t)),
           *idx2 = malloc(arr2->ndim * sizeof(size_t)),
           *idx = malloc(result->ndim * sizeof(size_t));

    if (!idx1 || !idx2 || !idx) {
        printf("Failed to allocate index buffers\n");
        exit(ARRAY_INIT_FAILURE);
    }

    size_t batch_size = 1;
    for (int i = 0; i < batch_ndim; i++)
        batch_size *= result->shape[i];

    for (int b = 0; b < (batch_ndim == 0 ? 1 : batch_size); b++) {
        _get_broadcasted_indices(arr1->shape, arr2->shape, result->shape,
                                 batch_ndim1, batch_ndim2, batch_ndim, idx1,
                                 idx2, idx, b);
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
    if (arr1->ndim < 2 || arr2->ndim < 2) {
        printf("matmul requires arrays with ndim >= 2\n");
        exit(INVALID_ARRAY);
    }

    size_t m = arr1->shape[arr1->ndim - 2], k1 = arr1->shape[arr1->ndim - 1],
           k2 = arr2->shape[arr2->ndim - 2], n = arr2->shape[arr2->ndim - 1];

    if (k1 != k2) {
        printf("matmul shape mismatch: k1 (%zu) != k2 (%zu) in (..., m, k1), "
               "(..., k2, n) -> (..., m, n)\n",
               k1, k2);
        exit(SHAPE_MISMATCH);
    }

    int batch_ndim1 = arr1->ndim - 2, batch_ndim2 = arr2->ndim - 2;
    int batch_ndim = (batch_ndim1 > batch_ndim2) ? batch_ndim1 : batch_ndim2;

    size_t *batch_shape =
        broadcast_shape(arr1->shape, arr2->shape, arr1->ndim, arr2->ndim);

    size_t *shape = malloc((batch_ndim + 2) * sizeof(size_t));
    if (!shape) {
        printf("Failed to allocate output shape\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (int i = 0; i < batch_ndim; i++)
        shape[i] = batch_shape[i];

    // final shape = (..., m, n)
    shape[batch_ndim] = m;
    shape[batch_ndim + 1] = n;

    ndArray *result = array_init(batch_ndim + 2, shape, sizeof(float));
    _batch_matmul(arr1, arr2, result);

    free(shape);
    free(batch_shape);

    return result;
}
