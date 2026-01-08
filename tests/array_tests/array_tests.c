#include "array_tests.h"
#include "array.h"

#include <CUnit/CUnit.h>
#include <stddef.h>
#include <stdio.h>

void test_array_init() {
    const size_t shape[] = {50};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);

    size_t size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        CU_ASSERT(get_shape(array)[i] == shape[i]);
        CU_ASSERT(get_strides(array)[i] == size * get_itemsize(array));
        size *= get_shape(array)[i];
    }
    CU_ASSERT(size == get_total_size(array));

    free_array(array);
}

void test_getters_and_setters() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);

    // set array as [[0., 1.], [-1., 2.]]
    size_t indices[] = {0, 0};
    ArrayVal value;

    value.float_val = 0.0f;
    set_value(array, indices, value);
    CU_ASSERT(array_val_equal(get_value(array, indices), value, DTYPE_FLOAT));

    indices[1] = 1;
    set_value(array, indices, value);
    CU_ASSERT(array_val_equal(get_value(array, indices), value, DTYPE_FLOAT));

    indices[0] = 1;
    indices[1] = 0;
    set_value(array, indices, value);
    CU_ASSERT(array_val_equal(get_value(array, indices), value, DTYPE_FLOAT));

    indices[1] = 1;
    set_value(array, indices, value);
    CU_ASSERT(array_val_equal(get_value(array, indices), value, DTYPE_FLOAT));

    free_array(array);
}

void test_populate_array() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    ArrayVal v1, v2, v3, v4;

    v1.float_val = 0.0f;
    v2.float_val = 1.0f;
    v3.float_val = -1.0f;
    v4.float_val = 2.0f;

    const float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    populate_array(array, data);

    size_t indices[] = {0, 0};
    CU_ASSERT(array_val_equal(get_value(array, indices), v1, DTYPE_FLOAT));

    indices[1] = 1;
    CU_ASSERT(array_val_equal(get_value(array, indices), v2, DTYPE_FLOAT));

    indices[0] = 1;
    indices[1] = 0;
    CU_ASSERT(array_val_equal(get_value(array, indices), v3, DTYPE_FLOAT));

    indices[1] = 1;
    CU_ASSERT(array_val_equal(get_value(array, indices), v4, DTYPE_FLOAT));

    free_array(array);
}

void test_eye_array() {
    size_t m = 3, n = 4;
    const size_t shape[] = {3, 4};

    ndArray *array = eye(m, n, DTYPE_INT);
    ndArray *new_array = array_init(2, shape, DTYPE_INT);

    const int data[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}

void test_zeroes_array() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);
    ndArray *array = zeros(ndim, shape, DTYPE_INT);
    ndArray *new_array = array_init(3, shape, DTYPE_INT);

    const int data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}

void test_ones_array() {
    const size_t shape[] = {3, 2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = ones(ndim, shape, DTYPE_FLOAT);
    ndArray *new_array = array_init(3, shape, DTYPE_FLOAT);

    const float data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    populate_array(new_array, data);
    CU_ASSERT(array_equal(array, new_array));

    free_array(array);
    free_array(new_array);
}
