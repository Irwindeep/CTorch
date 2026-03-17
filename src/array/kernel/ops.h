#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include "array.h"

#include <stddef.h>

void matmul_kernel(const ndArray *arr1, const ndArray *arr2, ndArray *result,
                   const size_t *idx1, const size_t *idx2, const size_t *idx);

#endif // !KERNEL_OPS_H
