#include "array.h"
#include "error_codes.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *DTypeNames[] = {
    "DTYPE_INT",
    "DTYPE_FLOAT",
    "DTYPE_DOUBLE",
    "DTYPE_LONG",
};

struct ndArray {
    void *data;
    int ndim;
    size_t *shape;
    size_t *strides;
    size_t itemsize;
    size_t total_size;
    DType dtype;
};

ndArray *array_init(int ndim, const size_t *shape, DType dtype) {
    if (ndim > MAX_NDIM)
        RUNTIME_ERRORF(ARRAY_INIT_FAILURE,
                       "Given array with ndim (%d) > MAX_NDIM (%d)", ndim,
                       MAX_NDIM);

    ndArray *array = malloc(sizeof(ndArray));
    if (!array)
        RUNTIME_ERROR(ARRAY_INIT_FAILURE, "Failed to initialize array");

    array->ndim = ndim;
    array->dtype = dtype;

    size_t itemsize;
    switch (dtype) {
    case DTYPE_INT:
        itemsize = sizeof(int);
        break;
    case DTYPE_FLOAT:
        itemsize = sizeof(float);
        break;
    case DTYPE_DOUBLE:
        itemsize = sizeof(double);
        break;
    case DTYPE_LONG:
        itemsize = sizeof(long);
        break;
    }

    array->itemsize = itemsize;
    array->shape = malloc(ndim * sizeof(size_t));
    array->strides = malloc(ndim * sizeof(size_t));

    size_t size = 1;
    for (size_t i = ndim; i-- > 0;) {
        array->shape[i] = shape[i];
        array->strides[i] = size * itemsize;
        size = size * shape[i];
    }
    array->data = malloc(size * itemsize);
    array->total_size = size;

    return array;
}

void free_array(ndArray *array) {
    free(array->data);
    free(array->shape);
    free(array->strides);
    free(array);
}

// getters and setters
static const void *array_idx_const(const ndArray *array,
                                   const size_t *indices) {
    size_t offset = 0;
    for (int i = 0; i < array->ndim; i++) {
        if (indices[i] < array->shape[i])
            offset += array->strides[i] * indices[i];
        else
            RUNTIME_ERRORF(INVALID_IDX, "Invalid index `%zu` at position `%d`",
                           indices[i], i);
    }
    return (const char *)array->data + offset;
}

// cppcheck-suppress constParameterCallback
static void *array_idx_mut(ndArray *array, const size_t *indices) {
    return (void *)array_idx_const(array, indices);
}

#define array_idx(array, indices)                                              \
    _Generic((array),                                                          \
        const ndArray *: array_idx_const,                                      \
        ndArray *: array_idx_mut)(array, indices)

int get_ndim(const ndArray *array) { return array->ndim; }
size_t get_itemsize(const ndArray *array) { return array->itemsize; }
size_t get_total_size(const ndArray *array) { return array->total_size; }
DType get_dtype(const ndArray *array) { return array->dtype; }
size_t *get_shape(const ndArray *array) { return array->shape; }
size_t *get_strides(const ndArray *array) { return array->strides; }
void *get_array_data(const ndArray *array) { return array->data; }

ArrayVal get_value(const ndArray *array, const size_t *indices) {
    const void *ptr = array_idx(array, indices);

    ArrayVal value;
    switch (array->dtype) {
    case DTYPE_INT:
        value.int_val = *(int *)ptr;
        break;
    case DTYPE_FLOAT:
        value.float_val = *(float *)ptr;
        break;
    case DTYPE_DOUBLE:
        value.double_val = *(double *)ptr;
        break;
    case DTYPE_LONG:
        value.long_val = *(long *)ptr;
        break;
    }

    return value;
}

void set_value(ndArray *array, const size_t *indices, ArrayVal value) {
    void *ptr = array_idx(array, indices);

    switch (array->dtype) {
    case DTYPE_INT:
        *(int *)ptr = value.int_val;
        break;
    case DTYPE_FLOAT:
        *(float *)ptr = value.float_val;
        break;
    case DTYPE_DOUBLE:
        *(double *)ptr = value.double_val;
        break;
    case DTYPE_LONG:
        *(long *)ptr = value.long_val;
        break;
    }
}

void set_strides(ndArray *array, const size_t *strides) {
    memcpy(array->strides, strides, array->ndim);
}

void populate_array(ndArray *array, const void *data) {
    memcpy(array->data, data, array->total_size * array->itemsize);
}

void offset_to_index(size_t offset, size_t *idx, const size_t *shape,
                     int ndim) {
    for (int d = 0; d < ndim; d++) {
        idx[d] = offset % shape[d];
        offset /= shape[d];
    }
}

size_t index_to_offset(const size_t *idx, const size_t *strides, int ndim) {
    size_t offset = 0;
    for (int d = 0; d < ndim; d++)
        offset += strides[d] * idx[d];

    return offset;
}

ndArray *copy_array(const ndArray *array) {
    int ndim = array->ndim;
    const size_t *shape = array->shape;
    DType dtype = array->dtype;

    ndArray *new_arr = array_init(ndim, shape, dtype);
    populate_array(new_arr, array->data);

    return new_arr;
}

bool is_array_contiguous(const ndArray *array) {
    size_t expected = array->itemsize;
    const size_t *strides = array->strides;
    int ndim = array->ndim;

    for (int d = ndim - 1; d >= 0; d--) {
        if (strides[d] != expected)
            return false;

        expected *= array->shape[d];
    }
    return true;
}

// some important arrays
ndArray *eye(size_t m, size_t n, DType dtype) {
    const size_t shape[] = {m, n};
    ndArray *array = array_init(2, shape, dtype);
    size_t indices[2] = {0, 0};

    ArrayVal one = array_val_one(array->dtype);
    ArrayVal zero = array_val_zero(array->dtype);

    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            indices[0] = row, indices[1] = col;
            set_value(array, indices, (row == col) ? one : zero);
        }
    }

    return array;
}

ndArray *zeros(int ndim, const size_t *shape, DType dtype) {
    ndArray *array = array_init(ndim, shape, dtype);
    size_t indices[ndim];

    ArrayVal zero = array_val_zero(array->dtype);
    size_t total_size = array->total_size;
    for (size_t i = 0; i < total_size; i++) {
        offset_to_index(i, indices, shape, ndim);
        set_value(array, indices, zero);
    }

    return array;
}

ndArray *ones(int ndim, const size_t *shape, DType dtype) {
    ndArray *array = array_init(ndim, shape, dtype);
    size_t indices[ndim];

    ArrayVal one = array_val_one(array->dtype);
    size_t total_size = array->total_size;
    for (size_t i = 0; i < total_size; i++) {
        offset_to_index(i, indices, shape, ndim);
        set_value(array, indices, one);
    }

    return array;
}

bool array_equal(const ndArray *arr1, const ndArray *arr2) {
    if ((arr1->dtype != arr2->dtype) || arr1->ndim != arr2->ndim)
        return false;

    DType dtype = arr1->dtype;
    int ndim = arr1->ndim;
    size_t idx[ndim];

    for (int i = 0; i < ndim; i++) {
        if (arr1->shape[i] != arr2->shape[i])
            return false;
    }

    size_t total_size = arr1->total_size;
    for (size_t i = 0; i < total_size; i++) {
        offset_to_index(i, idx, arr1->shape, ndim);
        if (!array_val_equal(get_value(arr1, idx), get_value(arr2, idx), dtype))
            return false;
    }

    return true;
}
