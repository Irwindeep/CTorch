#include "array.h"
#include "array_binops.h"
#include "ctorch.h"
#include "error_codes.h"

#include <stddef.h>
#include <stdio.h>

typedef void (*dispatcher)(DType dtype, const ndArray *arr1,
                           const ndArray *arr2, ndArray *result,
                           const size_t *b_strides1, const size_t *b_strides2,
                           const size_t *strides, int ndim, size_t total_size,
                           const size_t *shape);

ndArray *array_binop_kernel(ndArray *arr1, ndArray *arr2, dispatcher dispatch) {
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2);
    int ndim = (ndim1 > ndim2) ? ndim1 : ndim2;

    DType dtype1 = get_dtype(arr1), dtype2 = get_dtype(arr2), dtype;
    if (dtype1 != dtype2)
        RUNTIME_ERRORF(INVALID_DTYPE, "Dtype mismatch `%s` and `%s`",
                       DTypeNames[dtype1], DTypeNames[dtype2]);
    dtype = dtype1;

    size_t *shape1 = get_shape(arr1), *shape2 = get_shape(arr2);
    size_t shape[MAX_NDIM];
    broadcast_shape(shape1, shape2, shape, ndim1, ndim2, ndim);
    ndArray *result = array_init(ndim, shape, dtype);

    size_t *strides1 = get_strides(arr1), *strides2 = get_strides(arr2);
    const size_t *strides = get_strides(result);

    size_t b_strides1[MAX_NDIM], b_strides2[MAX_NDIM], sC[MAX_NDIM];

    broadcasted_strides(b_strides1, strides1, shape1, ndim1, ndim);
    broadcasted_strides(b_strides2, strides2, shape2, ndim2, ndim);

    size_t total_size = get_total_size(result);
    size_t itemsize = get_itemsize(result);

    for (int i = 0; i < ndim; i++) {
        b_strides1[i] /= itemsize;
        b_strides2[i] /= itemsize;
        sC[i] = strides[i] / itemsize;
    }

    dispatch(dtype, arr1, arr2, result, b_strides1, b_strides2, sC, ndim,
             total_size, shape);

    return result;
}
