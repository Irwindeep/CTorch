#include "tensor_tests.h"
#include "array.h"
#include "tensor.h"

#include <CUnit/CUnit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

void test_tensor_init() {
    const size_t shape[] = {50};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    Tensor *tensor = tensor_init(array, false, NULL);

    size_t size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        CU_ASSERT(get_tensor_shape(tensor)[i] == shape[i]);
        CU_ASSERT(get_strides(get_tensor_data(tensor))[i] ==
                  size * get_itemsize(get_tensor_data(tensor)));
        size *= get_tensor_shape(tensor)[i];
    }
    CU_ASSERT(size == get_total_size(get_tensor_data(tensor)));

    free_tensor(tensor);
}

void test_tensor_getters_and_setters() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    const float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    populate_array(array, data);

    Tensor *tensor = tensor_init(array, false, NULL);

    CU_ASSERT(array_equal(get_tensor_data(tensor), array));
    free_tensor(tensor);
}

void test_eye_tensor() {
    size_t m = 3, n = 4;

    ndArray *array = eye(m, n, DTYPE_INT);
    Tensor *tensor = eye_tensor(m, n, DTYPE_INT, false, NULL);

    CU_ASSERT(array_equal(get_tensor_data(tensor), array));
    free_array(array);
    free_tensor(tensor);
}

void test_zeros_tensor() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = zeros(ndim, shape, DTYPE_INT);
    Tensor *tensor = zeros_tensor(ndim, shape, DTYPE_INT, false, NULL);

    CU_ASSERT(array_equal(get_tensor_data(tensor), array));
    free_array(array);
    free_tensor(tensor);
}

void test_ones_tensor() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = ones(ndim, shape, DTYPE_INT);
    Tensor *tensor = ones_tensor(ndim, shape, DTYPE_INT, false, NULL);

    CU_ASSERT(array_equal(get_tensor_data(tensor), array));
    free_array(array);
    free_tensor(tensor);
}
