#include "array_tests.h"
#include "array/array.h"
#include "array/array_ops.h"

#include <CUnit/CUnit.h>
#include <stddef.h>
#include <stdio.h>

void test_array_init() {
    const size_t shape[] = {50};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, sizeof(float));

    size_t size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        CU_ASSERT(get_shape(array)[i] == shape[i]);
        CU_ASSERT(get_strides(array)[i] == size * get_itemsize(array));
        size *= get_shape(array)[i];
    }

    free_array(array);
}

void test_getters_and_setters() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, sizeof(float));

    // set array as [[0., 1.], [-1., 2.]]
    size_t indices[] = {0, 0};

    set_value(array, indices, 0.0f);
    CU_ASSERT(get_value(array, indices) == 0.0f);

    indices[1] = 1;
    set_value(array, indices, 1.0f);
    CU_ASSERT(get_value(array, indices) == 1.0f);

    indices[0] = 1;
    indices[1] = 0;
    set_value(array, indices, -1.0f);
    CU_ASSERT(get_value(array, indices) == -1.0f);

    indices[1] = 1;
    set_value(array, indices, 2.0f);
    CU_ASSERT(get_value(array, indices) == 2.0f);

    free_array(array);
}

void test_populate_array() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, sizeof(float));
    const float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    populate_array(array, data);

    size_t indices[] = {0, 0};
    CU_ASSERT(get_value(array, indices) == 0.0f);

    indices[1] = 1;
    CU_ASSERT(get_value(array, indices) == 1.0f);

    indices[0] = 1;
    indices[1] = 0;
    CU_ASSERT(get_value(array, indices) == -1.0f);

    indices[1] = 1;
    CU_ASSERT(get_value(array, indices) == 2.0f);

    free_array(array);
}

void test_eye_array() {
    size_t m = 3, n = 4;
    const size_t shape[] = {3, 4};

    ndArray *array = eye(m, n);
    ndArray *new_array = array_init(2, shape, sizeof(float));

    const float data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}

void test_zeroes_array() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);
    ndArray *array = zeros(ndim, shape);
    ndArray *new_array = array_init(3, shape, sizeof(float));

    const float data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}

void test_ones_array() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = ones(ndim, shape);
    ndArray *new_array = array_init(3, shape, sizeof(float));

    const float data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}
