#include "array/array.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// array initialization
ndArray *array_init(int ndim, const size_t *shape, size_t itemsize) {
    ndArray *array = malloc(sizeof(ndArray));

    if (array == NULL) {
        printf("Failed to initialize array\n");
        exit(ARRAY_INIT_FAILURE);
    }

    array->ndim = ndim;
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
    return array;
}

void *array_idx(ndArray *array, const size_t *indices) {
    size_t offset = 0;
    for (int i = 0; i < array->ndim; i++) {
        if (indices[i] < array->shape[i])
            offset += array->strides[i] * indices[i];
        else {
            printf("Invalid index `%zu` at position `%d`\n", indices[i], i);
            exit(INVALID_IDX);
        }
    }
    return (char *)array->data + offset;
}

void set_value(ndArray *array, const size_t *indices, float value) {
    float *ptr = (float *)array_idx(array, indices);
    *ptr = value;
}

float get_value(ndArray *array, const size_t *indices) {
    void *ptr = array_idx(array, indices);
    return *(float *)ptr;
}

void populate_array(ndArray *array, const float *data) {
    size_t total_elems = 1;
    for (int i = 0; i < array->ndim; i++) {
        total_elems *= array->shape[i];
    }

    memcpy(array->data, data, total_elems * array->itemsize);
}

void free_array(ndArray *array) {
    free(array->data);
    free(array->shape);
    free(array->strides);
    free(array);
}
