#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include "array.h"
#include <stdbool.h>
#include <stddef.h>

#define TOL 1e-6

// array comparators
bool array_equal(ndArray *arr1, ndArray *arr2);

// array operations
bool broadcastable(const size_t *shape1, const size_t *shape2, int ndim1,
                   int ndim2);
size_t *broadcast_shape(const size_t *shape1, const size_t *shape2, int ndim1,
                        int ndim2);
ndArray *add(ndArray *arr1, ndArray *arr2);
ndArray *sub(ndArray *arr1, ndArray *arr2);
ndArray *mul(ndArray *arr1, ndArray *arr2);
ndArray *matmul(ndArray *arr1, ndArray *arr2);

#endif // !ARRAY_OPS_H
