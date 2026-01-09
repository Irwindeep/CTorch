#include "array.h"
#include "array_tests.h"
#include "print.h"

#include <CUnit/CUnit.h>
#include <stddef.h>
#include <stdio.h>

void test_array_equal() {
    const size_t shape1[] = {3, 3}, shape2[] = {3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]);

    ndArray *arr1 = array_init(ndim1, shape1, DTYPE_INT),
            *arr2 = array_init(ndim2, shape2, DTYPE_INT);

    const int data1[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const int data2[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    populate_array(arr1, data1);
    populate_array(arr2, data2);

    CU_ASSERT(array_equal(arr1, arr2));

    const size_t idx[] = {0, 0};
    ArrayVal val;
    val.int_val = 3;
    set_value(arr2, idx, val);
    CU_ASSERT(!array_equal(arr1, arr2));

    free_array(arr1);
    free_array(arr2);
}

void test_array_add() {
    const size_t shape1[] = {3, 3}, shape2[] = {3}, shape3[] = {3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]),
        ndim3 = sizeof(shape3) / sizeof(shape3[0]);

    ndArray *arr1 = array_init(ndim1, shape1, DTYPE_INT),
            *arr2 = array_init(ndim2, shape2, DTYPE_INT),
            *arr3 = array_init(ndim3, shape3, DTYPE_INT);

    const int data1[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const int data2[] = {2, 3, 4};
    const int data3[] = {3, 3, 4, 2, 4, 4, 2, 3, 5};

    populate_array(arr1, data1);
    populate_array(arr2, data2);
    populate_array(arr3, data3);

    ndArray *result = add(arr1, arr2);
    CU_ASSERT(array_equal(result, arr3));

    free_array(arr1);
    free_array(arr2);
    free_array(arr3);
    free_array(result);
}

void test_array_sub() {
    const size_t shape1[] = {3, 3}, shape2[] = {3}, shape3[] = {3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]),
        ndim3 = sizeof(shape3) / sizeof(shape3[0]);

    ndArray *arr1 = array_init(ndim1, shape1, DTYPE_LONG),
            *arr2 = array_init(ndim2, shape2, DTYPE_LONG),
            *arr3 = array_init(ndim3, shape3, DTYPE_LONG);

    const long int data1[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const long int data2[] = {2, 3, 4};
    const long int data3[] = {-1, -3, -4, -2, -2, -4, -2, -3, -3};
    populate_array(arr1, data1);
    populate_array(arr2, data2);
    populate_array(arr3, data3);

    ndArray *result = sub(arr1, arr2);
    CU_ASSERT(array_equal(result, arr3));

    free_array(arr1);
    free_array(arr2);
    free_array(arr3);
    free_array(result);
}

void test_array_mul() {
    const size_t shape1[] = {3, 3}, shape2[] = {3}, shape3[] = {3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]),
        ndim3 = sizeof(shape3) / sizeof(shape3[0]);

    ndArray *arr1 = array_init(ndim1, shape1, DTYPE_DOUBLE),
            *arr2 = array_init(ndim2, shape2, DTYPE_DOUBLE),
            *arr3 = array_init(ndim3, shape3, DTYPE_DOUBLE);

    const double data1[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double data2[] = {2, 3, 4};
    const double data3[] = {2, 0, 0, 0, 3, 0, 0, 0, 4};
    populate_array(arr1, data1);
    populate_array(arr2, data2);
    populate_array(arr3, data3);

    ndArray *result = mul(arr1, arr2);
    CU_ASSERT(array_equal(result, arr3));

    free_array(arr1);
    free_array(arr2);
    free_array(arr3);
    free_array(result);
}

void test_array_matmul() {
    const size_t shape1[] = {2, 3, 3}, shape2[] = {1, 3, 3},
                 shape3[] = {2, 3, 3};
    int ndim1 = sizeof(shape1) / sizeof(shape1[0]),
        ndim2 = sizeof(shape2) / sizeof(shape2[0]),
        ndim3 = sizeof(shape3) / sizeof(shape3[0]);

    ndArray *arr1 = array_init(ndim1, shape1, DTYPE_FLOAT),
            *arr2 = array_init(ndim2, shape2, DTYPE_FLOAT),
            *arr3 = array_init(ndim3, shape3, DTYPE_FLOAT);

    const float data1[] = {0.47f, -0.68f, 0.24f,  -1.70f, 0.75f,  -1.53f,
                           0.01f, -0.12f, -0.81f, 2.87f,  -0.60f, 0.47f,
                           1.10f, -1.22f, 1.34f,  -0.12f, 1.01f,  -0.91f};
    const float data2[] = {-1.00f, -0.71f, 0.04f, -0.68f, -0.57f,
                           -0.11f, 1.34f,  0.32f, -0.34f};
    const float data3[] = {0.3140f,  0.1307f,  0.0120f,  -0.8602f, 0.2899f,
                           0.3697f,  -1.0138f, -0.1979f, 0.2890f,  -1.8322f,
                           -1.5453f, 0.0210f,  1.5252f,  0.3432f,  -0.2774f,
                           -1.7862f, -0.7817f, 0.1935f};
    populate_array(arr1, data1);
    populate_array(arr2, data2);
    populate_array(arr3, data3);

    ndArray *result = matmul(arr1, arr2);
    CU_ASSERT(array_equal(result, arr3));

    free_array(arr1);
    free_array(arr2);
    free_array(arr3);
    free_array(result);
}

void test_array_transpose() {
    const size_t shape[] = {2, 3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]);
    const float data[] = {0.3140f,  0.1307f,  0.0120f,  -0.8602f, 0.2899f,
                          0.3697f,  -1.0138f, -0.1979f, 0.2890f,  -1.8322f,
                          -1.5453f, 0.0210f,  1.5252f,  0.3432f,  -0.2774f,
                          -1.7862f, -0.7817f, 0.1935f};

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(array, data);

    transpose(array, (int[]){1, 2, 0});
    ndArray *array_T = array_init(ndim, (size_t[]){3, 3, 2}, DTYPE_FLOAT);
    const float data1[] = {0.3140f,  -1.8322f, 0.1307f,  -1.5453f, 0.0120f,
                           0.0210f,  -0.8602f, 1.5252f,  0.2899f,  0.3432f,
                           0.3697f,  -0.2774f, -1.0138f, -1.7862f, -0.1979f,
                           -0.7817f, 0.2890f,  0.1935f};
    populate_array(array_T, data1);

    CU_ASSERT(array_equal(array, array_T));
    free_array(array);
    free_array(array_T);
}

void test_array_sum() {
    const size_t shape[] = {2, 3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]);
    const float data[] = {0.3140f,  0.1307f,  0.0120f,  -0.8602f, 0.2899f,
                          0.3697f,  -1.0138f, -0.1979f, 0.2890f,  -1.8322f,
                          -1.5453f, 0.0210f,  1.5252f,  0.3432f,  -0.2774f,
                          -1.7862f, -0.7817f, 0.1935f};

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(array, data);

    ndArray *arr_sum = array_sum(array);
    ndArray *truth_arr = array_init(0, (size_t[]){}, DTYPE_FLOAT);
    populate_array(truth_arr, (float[]){-4.8065f});
    CU_ASSERT(array_equal(arr_sum, truth_arr));

    free_array(array);
    free_array(arr_sum);
    free_array(truth_arr);
}

void test_array_sum_dim() {
    const size_t shape[] = {2, 3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]);
    const float data[] = {0.3140f,  0.1307f,  0.0120f,  -0.8602f, 0.2899f,
                          0.3697f,  -1.0138f, -0.1979f, 0.2890f,  -1.8322f,
                          -1.5453f, 0.0210f,  1.5252f,  0.3432f,  -0.2774f,
                          -1.7862f, -0.7817f, 0.1935f};

    ndArray *array = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(array, data);

    ndArray *arr_sum = array_sum_dim(array, 1, true);
    ndArray *truth_arr = array_init(ndim, (size_t[]){2, 1, 3}, DTYPE_FLOAT);
    const float data1[] = {-1.5600f, 0.2227f,  0.6707f,
                           -2.0932f, -1.9838f, -0.0629f};
    populate_array(truth_arr, data1);
    CU_ASSERT(array_equal(arr_sum, truth_arr));

    free_array(array);
    free_array(arr_sum);
    free_array(truth_arr);
}
