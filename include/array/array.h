/*
NumPy style NDArray implementation - only supported for float data types
*/

#ifndef ARRAY_H
#define ARRAY_H

#include <stddef.h>

typedef struct ndArray ndArray;

// error codes
#define ARRAY_INIT_FAILURE 1
#define INVALID_IDX 2
#define NON_BROADCASTABLE_ARRAYS 3
#define SHAPE_MISMATCH 4
#define INVALID_ARRAY 5

// array initialization
ndArray *array_init(int ndim, const size_t *shape, size_t itemsize);
void free_array(ndArray *array);

// getters and setters
float get_value(const ndArray *array, const size_t *indices);
int get_ndim(const ndArray *array);
size_t get_itemsize(const ndArray *array);
size_t *get_shape(const ndArray *array);
size_t *get_strides(const ndArray *array);

void set_value(const ndArray *array, const size_t *indices, float value);
void populate_array(ndArray *array, const float *data);

// some important arrays
ndArray *eye(size_t m, size_t n);
ndArray *zeros(int ndim, const size_t *shape);
ndArray *ones(int ndim, const size_t *shape);

#endif // !ARRAY_H
