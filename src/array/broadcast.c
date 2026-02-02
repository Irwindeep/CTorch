#include "array.h"
#include "error_codes.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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

void broadcast_shape(const size_t *shape1, const size_t *shape2, size_t *shape,
                     int ndim1, int ndim2, int ndim) {
    if (!broadcastable(shape1, shape2, ndim1, ndim2))
        RUNTIME_ERROR(NON_BROADCASTABLE_ARRAYS, "Dimensions not broadcastable");

    for (int i = 0; i < ndim; i++) {
        int idx1 = ndim1 - 1 - i;
        int idx2 = ndim2 - 1 - i;

        size_t s1 = (idx1 >= 0) ? shape1[idx1] : 1;
        size_t s2 = (idx2 >= 0) ? shape2[idx2] : 1;
        shape[ndim - i - 1] = (s1 > s2) ? s1 : s2;
    }
}

void broadcasted_strides(size_t *strides, const size_t *src_strides,
                         const size_t *src_shape, int src_ndim,
                         const size_t *dst_shape, int dst_ndim) {
    int dim = dst_ndim - src_ndim;

    for (int i = 0; i < dst_ndim; i++) {
        if (i < dim)
            strides[i] = 0;
        else if (src_shape[i - dim] == 1)
            strides[i] = 0;
        else
            strides[i] = src_strides[i - dim];
    }
}

void get_broadcasted_indices(const size_t *shape1, const size_t *shape2,
                             const size_t *shape, int ndim1, int ndim2,
                             int ndim, size_t *idx1, size_t *idx2, size_t *idx,
                             size_t batch) {
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
