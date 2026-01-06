#include "array_tests.h"
#include "array/array.h"

#include <CUnit/CUnit.h>
#include <stdio.h>

void test_array_init() {
    const size_t shape[] = {10, 5};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, sizeof(float));

    CU_ASSERT(array->shape[0] == shape[0]);
    CU_ASSERT(array->shape[1] == shape[1]);
    CU_ASSERT(array->strides[0] == 5 * array->itemsize);
    CU_ASSERT(array->strides[1] == array->itemsize);

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

    // array should be contiguously allocated in memory
    void *base = array->data;
    CU_ASSERT(*(float *)base == 0.0f);
    CU_ASSERT(*((float *)base + 1) == 1.0f);
    CU_ASSERT(*((float *)base + 2) == -1.0f);
    CU_ASSERT(*((float *)base + 3) == 2.0f);

    free_array(array);
}

void test_populate_array() {
    const size_t shape[] = {2, 2};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    ndArray *array = array_init(ndim, shape, sizeof(float));
    const float data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    populate_array(array, data);

    void *base = array->data;
    CU_ASSERT(*(float *)base == 0.0f);
    CU_ASSERT(*((float *)base + 1) == 1.0f);
    CU_ASSERT(*((float *)base + 2) == -1.0f);
    CU_ASSERT(*((float *)base + 3) == 2.0f);

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
