#include "array.h"
#include "autograd.h"
#include "tensor.h"
#include "tensor_tests.h"

#include <CUnit/CUnit.h>
#include <stdbool.h>
#include <stddef.h>

void test_tensor_add() {
    const size_t shape[] = {3}, shape_result[] = {3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]),
        ndim_result = sizeof(shape_result) / sizeof(shape_result[0]);

    Environment *env = env_init();

    Tensor *t1 = eye_tensor(3, 3, DTYPE_FLOAT, true, env);

    ndArray *arr = array_init(ndim, shape, DTYPE_FLOAT);
    const float data[] = {2.0f, 3.0f, 4.0f};
    populate_array(arr, data);

    Tensor *t2 = tensor_init(arr, true, env);
    Tensor *t3 = tensor_add(t1, t2);

    ndArray *result = array_init(ndim_result, shape_result, DTYPE_FLOAT);
    const float result_data[] = {3.0f, 3.0f, 4.0f, 2.0f, 4.0f,
                                 4.0f, 2.0f, 3.0f, 5.0f};
    populate_array(result, result_data);
    CU_ASSERT(array_equal(result, get_tensor_data(t3)));

    backward(t3, ones_like(t3, false, env));

    Tensor *t1_grad = get_tensor_grad(t1), *t2_grad = get_tensor_grad(t2);

    ndArray *t1_grad_arr = ones(2, (const size_t[]){3, 3}, DTYPE_FLOAT),
            *t2_grad_arr = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(t2_grad_arr, (const float[]){3.0f, 3.0f, 3.0f});

    CU_ASSERT(array_equal(get_tensor_data(t1_grad), t1_grad_arr));
    CU_ASSERT(array_equal(get_tensor_data(t2_grad), t2_grad_arr));

    free_array(result);
    free_array(t1_grad_arr);
    free_array(t2_grad_arr);

    free_env(env);
}

void test_tensor_sub() {
    const size_t shape[] = {3}, shape_result[] = {3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]),
        ndim_result = sizeof(shape_result) / sizeof(shape_result[0]);

    Environment *env = env_init();

    Tensor *t1 = eye_tensor(3, 3, DTYPE_FLOAT, true, env);

    ndArray *arr = array_init(ndim, shape, DTYPE_FLOAT);
    const float data[] = {2.0f, 3.0f, 4.0f};
    populate_array(arr, data);

    Tensor *t2 = tensor_init(arr, true, env);
    Tensor *t3 = tensor_sub(t1, t2);

    ndArray *result = array_init(ndim_result, shape_result, DTYPE_FLOAT);
    const float result_data[] = {-1.0f, -3.0f, -4.0f, -2.0f, -2.0f,
                                 -4.0f, -2.0f, -3.0f, -3.0f};
    populate_array(result, result_data);
    CU_ASSERT(array_equal(result, get_tensor_data(t3)));

    backward(t3, ones_like(t3, false, env));

    Tensor *t1_grad = get_tensor_grad(t1), *t2_grad = get_tensor_grad(t2);

    ndArray *t1_grad_arr = ones(2, (const size_t[]){3, 3}, DTYPE_FLOAT),
            *t2_grad_arr = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(t2_grad_arr, (const float[]){-3.0f, -3.0f, -3.0f});

    CU_ASSERT(array_equal(get_tensor_data(t1_grad), t1_grad_arr));
    CU_ASSERT(array_equal(get_tensor_data(t2_grad), t2_grad_arr));

    free_array(result);
    free_array(t1_grad_arr);
    free_array(t2_grad_arr);

    free_env(env);
}

void test_tensor_mul() {
    const size_t shape[] = {3}, shape_result[] = {3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]),
        ndim_result = sizeof(shape_result) / sizeof(shape_result[0]);

    Environment *env = env_init();

    Tensor *t1 = eye_tensor(3, 3, DTYPE_FLOAT, true, env);

    ndArray *arr = array_init(ndim, shape, DTYPE_FLOAT);
    const float data[] = {2.0f, 3.0f, 4.0f};
    populate_array(arr, data);

    Tensor *t2 = tensor_init(arr, true, env);
    Tensor *t3 = tensor_mul(t1, t2);

    ndArray *result = array_init(ndim_result, shape_result, DTYPE_FLOAT);
    const float result_data[] = {2.0f, 0.0f, 0.0f, 0.0f, 3.0f,
                                 0.0f, 0.0f, 0.0f, 4.0f};
    populate_array(result, result_data);
    CU_ASSERT(array_equal(result, get_tensor_data(t3)));

    backward(t3, ones_like(t3, false, env));

    Tensor *t1_grad = get_tensor_grad(t1), *t2_grad = get_tensor_grad(t2);

    ndArray *t1_grad_arr = array_init(2, (const size_t[]){3, 3}, DTYPE_FLOAT),
            *t2_grad_arr = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(t1_grad_arr, (const float[]){2.0f, 3.0f, 4.0f, 2.0f, 3.0f,
                                                4.0f, 2.0f, 3.0f, 4.0f});
    populate_array(t2_grad_arr, (const float[]){1.0f, 1.0f, 1.0f});

    CU_ASSERT(array_equal(get_tensor_data(t1_grad), t1_grad_arr));
    CU_ASSERT(array_equal(get_tensor_data(t2_grad), t2_grad_arr));

    free_array(result);
    free_array(t1_grad_arr);
    free_array(t2_grad_arr);

    free_env(env);
}

void test_tensor_div() {
    const size_t shape[] = {3}, shape_result[] = {3, 3};
    int ndim = sizeof(shape) / sizeof(shape[0]),
        ndim_result = sizeof(shape_result) / sizeof(shape_result[0]);

    Environment *env = env_init();

    Tensor *t1 = eye_tensor(3, 3, DTYPE_FLOAT, true, env);

    ndArray *arr = array_init(ndim, shape, DTYPE_FLOAT);
    const float data[] = {2.0f, 3.0f, 4.0f};
    populate_array(arr, data);

    Tensor *t2 = tensor_init(arr, true, env);
    Tensor *t3 = tensor_div(t1, t2);

    ndArray *result = array_init(ndim_result, shape_result, DTYPE_FLOAT);
    const float result_data[] = {0.500000f, 0.000000f, 0.000000f,
                                 0.000000f, 0.333333f, 0.000000f,
                                 0.000000f, 0.000000f, 0.250000f};
    populate_array(result, result_data);
    CU_ASSERT(array_equal(result, get_tensor_data(t3)));

    backward(t3, ones_like(t3, false, env));

    Tensor *t1_grad = get_tensor_grad(t1), *t2_grad = get_tensor_grad(t2);

    ndArray *t1_grad_arr = array_init(2, (const size_t[]){3, 3}, DTYPE_FLOAT),
            *t2_grad_arr = array_init(ndim, shape, DTYPE_FLOAT);
    populate_array(t1_grad_arr,
                   (const float[]){0.500000f, 0.333333f, 0.250000f, 0.500000f,
                                   0.333333f, 0.250000f, 0.500000f, 0.333333f,
                                   0.250000f});
    populate_array(t2_grad_arr,
                   (const float[]){-0.250000f, -0.111111f, -0.062500f});

    CU_ASSERT(array_equal(get_tensor_data(t1_grad), t1_grad_arr));
    CU_ASSERT(array_equal(get_tensor_data(t2_grad), t2_grad_arr));

    free_array(result);
    free_array(t1_grad_arr);
    free_array(t2_grad_arr);

    free_env(env);
}
