#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = array_add(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad);

    if (requires_grad) {
        size_t dependency_cnt = (t1_requires_grad) ? 1 : 0;
        dependency_cnt += (t2_requires_grad) ? 1 : 0;

        Dependency **dependency_arr = dependency_arr_init(dependency_cnt);
        size_t idx = 0;

        TwoTensorCtx *ctx = init_two_tensor_ctx(t1, t2);
        set_tensor_ctx(tensor, ctx, TWO_TENSOR_CTX);

        if (t1_requires_grad) {
            Dependency *dep = create_dependency(t1, _add_grad_fn1);
            dependency_arr[idx++] = dep;
        }
        if (t2_requires_grad) {
            Dependency *dep = create_dependency(t2, _add_grad_fn2);
            dependency_arr[idx] = dep;
        }

        set_dependency_arr(tensor, dependency_arr, dependency_cnt);
    }

    return tensor;
}

Tensor *tensor_sub(Tensor *t1, Tensor *t2) {
    Tensor *t2_neg = tensor_neg(t2);
    Tensor *result = tensor_add(t1, t2_neg);

    return result;
}

Tensor *tensor_mul(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = array_mul(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad);

    if (requires_grad) {
        size_t dependency_cnt = (t1_requires_grad) ? 1 : 0;
        dependency_cnt += (t2_requires_grad) ? 1 : 0;

        Dependency **dependency_arr = dependency_arr_init(dependency_cnt);
        size_t idx = 0;

        TwoTensorCtx *ctx = init_two_tensor_ctx(t1, t2);
        set_tensor_ctx(tensor, ctx, TWO_TENSOR_CTX);

        if (t1_requires_grad) {
            Dependency *dep = create_dependency(t1, _mul_grad_fn1);
            dependency_arr[idx++] = dep;
        }
        if (t2_requires_grad) {
            Dependency *dep = create_dependency(t2, _mul_grad_fn2);
            dependency_arr[idx] = dep;
        }

        set_dependency_arr(tensor, dependency_arr, dependency_cnt);
    }

    return tensor;
}

Tensor *tensor_div(Tensor *t1, Tensor *t2) {
    Tensor *t2_inv = tensor_inv(t2);
    Tensor *result = tensor_mul(t1, t2_inv);

    return result;
}

Tensor *tensor_neg(Tensor *tensor) {
    ndArray *data = get_tensor_data(tensor);
    ndArray *new_data = negative(data);
    bool required_grad = get_requires_grad(tensor);

    Tensor *new_tensor = tensor_init(new_data, required_grad);
    if (required_grad) {
        Dependency **dependency_arr = dependency_arr_init(1);

        SingleTensorCtx *ctx = init_single_tensor_ctx(tensor);
        set_tensor_ctx(new_tensor, ctx, SINGLE_TENSOR_CTX);

        Dependency *dep = create_dependency(tensor, _neg_grad_fn);
        dependency_arr[0] = dep;
        set_dependency_arr(tensor, dependency_arr, 1);
    }

    return new_tensor;
}

Tensor *tensor_inv(Tensor *tensor) {
    ndArray *data = get_tensor_data(tensor);
    ndArray *new_data = inverse(data);
    bool required_grad = get_requires_grad(tensor);

    Tensor *new_tensor = tensor_init(new_data, required_grad);
    if (required_grad) {
        Dependency **dependency_arr = dependency_arr_init(1);

        SingleTensorCtx *ctx = init_single_tensor_ctx(tensor);
        set_tensor_ctx(new_tensor, ctx, SINGLE_TENSOR_CTX);

        Dependency *dep = create_dependency(tensor, _inv_grad_fn);
        dependency_arr[0] = dep;
        set_dependency_arr(new_tensor, dependency_arr, 1);
    }

    return new_tensor;
}
