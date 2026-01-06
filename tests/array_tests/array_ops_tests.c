#include "array_ops_tests.h"
#include "array/array.h"
#include "array/array_ops.h"

#include <stddef.h>
#include <stdio.h>

void test_matmul() {
    const size_t shape1[] = {3, 3}, shape2[] = {3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]);

    ndArray *arr1 = array_init(ndim1, shape1, sizeof(float)),
            *arr2 = array_init(ndim2, shape2, sizeof(float));

    const float data1[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float data2[] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    populate_array(arr1, data1);
    populate_array(arr2, data2);

    ndArray *result = matmul(arr1, arr2);

    free_array(arr1);
    free_array(arr2);
    free_array(result);
}
