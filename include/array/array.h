/*
NumPy style NDArray implementation - only supported for float data types
*/

#ifndef ARRAY_H
#define ARRAY_H

#include <stddef.h>

typedef struct ndArray {
    void *data;
    int ndim;
    size_t *shape;
    size_t *strides;
    size_t itemsize;
} ndArray;

// error codes
#define ARRAY_INIT_FAILURE 1
#define INVALID_IDX 2
#define NON_BROADCASTABLE_ARRAYS 3
#define SHAPE_MISMATCH 4
#define INVALID_ARRAY 5

// array initialization
ndArray *array_init(int ndim, const size_t *shape, size_t itemsize);
void *array_idx(ndArray *array, const size_t *indices);
void set_value(ndArray *array, const size_t *indices, float value);
float get_value(ndArray *array, const size_t *indices);
void populate_array(ndArray *array, const float *data);
void free_array(ndArray *array);

#endif // !ARRAY_H
