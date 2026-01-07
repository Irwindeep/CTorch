#include "array/array.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ndArray struct
typedef struct ndArray {
    void *data;
    int ndim;
    size_t *shape;
    size_t *strides;
    size_t itemsize;
} ndArray;

// array initialization
ndArray *array_init(int ndim, const size_t *shape, size_t itemsize) {
    ndArray *array = malloc(sizeof(ndArray));
    if (!array) {
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

void free_array(ndArray *array) {
    free(array->data);
    free(array->shape);
    free(array->strides);
    free(array);
}

// getters and setters
static void *array_idx(const ndArray *array, const size_t *indices) {
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

float get_value(const ndArray *array, const size_t *indices) {
    void *ptr = array_idx(array, indices);
    return *(float *)ptr;
}

int get_ndim(const ndArray *array) { return array->ndim; }
size_t get_itemsize(const ndArray *array) { return array->itemsize; }
size_t *get_shape(const ndArray *array) { return array->shape; }
size_t *get_strides(const ndArray *array) { return array->strides; }

void set_value(const ndArray *array, const size_t *indices, float value) {
    float *ptr = (float *)array_idx(array, indices);
    *ptr = value;
}

void populate_array(ndArray *array, const float *data) {
    size_t total_elems = 1;
    for (int i = 0; i < array->ndim; i++) {
        total_elems *= array->shape[i];
    }

    memcpy(array->data, data, total_elems * array->itemsize);
}

// some important arrays
ndArray *eye(size_t m, size_t n) {
    const size_t shape[] = {m, n};
    ndArray *array = array_init(2, shape, sizeof(float));
    size_t indices[2] = {0, 0};

    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            indices[0] = row, indices[1] = col;
            set_value(array, indices, (row == col) ? 1.0f : 0.0f);
        }
    }

    return array;
}

ndArray *zeros(int ndim, const size_t *shape) {
    ndArray *array = array_init(ndim, shape, sizeof(float));
    size_t *indices = malloc(ndim * sizeof(size_t));
    if (!indices) {
        printf("Failure to create index buffer\n");
        exit(ARRAY_INIT_FAILURE);
    }

    size_t total = 1;
    for (int i = 0; i < ndim; i++)
        total *= shape[i];

    for (size_t i = 0; i < total; i++) {
        size_t tmp = i;
        for (int d = 0; d < ndim; d++) {
            indices[d] = tmp % shape[d];
            tmp /= shape[d];
        }
        set_value(array, indices, 0.0f);
    }

    free(indices);
    return array;
}

ndArray *ones(int ndim, const size_t *shape) {
    ndArray *array = array_init(ndim, shape, sizeof(float));
    size_t *indices = malloc(ndim * sizeof(size_t));
    if (!indices) {
        printf("Failure to create index buffer\n");
        exit(ARRAY_INIT_FAILURE);
    }

    size_t total = 1;
    for (int i = 0; i < ndim; i++)
        total *= shape[i];

    for (size_t i = 0; i < total; i++) {
        size_t tmp = i;
        for (int d = 0; d < ndim; d++) {
            indices[d] = tmp % shape[d];
            tmp /= shape[d];
        }
        set_value(array, indices, 1.0f);
    }

    free(indices);
    return array;
}
