#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = array_add(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad);
    Tensor *dependency_arr[] = {t1, t2};
    size_t dependency_cnt = sizeof(dependency_arr) / sizeof(dependency_arr[0]);

    if (requires_grad) {
        BackwardFn *backward_fn = AddBackward(
            (Tensor *[]){tensor}, dependency_arr, 1, dependency_cnt);
        set_backward_fn(tensor, backward_fn);
    }

    return tensor;
}

Tensor *tensor_sub(Tensor *t1, Tensor *t2) {
    Tensor *t2_neg = tensor_neg(t2);
    Tensor *result = tensor_add(t1, t2_neg);

    Tensor *dependency_arr[] = {t1, t2_neg};
    size_t dependency_cnt = sizeof(dependency_arr) / sizeof(dependency_arr[0]);

    return result;
}

Tensor *tensor_mul(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = array_mul(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad);
    Tensor *dependency_arr[] = {t1, t2};
    size_t dependency_cnt = sizeof(dependency_arr) / sizeof(dependency_arr[0]);

    return tensor;
}

Tensor *tensor_div(Tensor *t1, Tensor *t2) {
    Tensor *t2_inv = tensor_inv(t2);
    Tensor *result = tensor_mul(t1, t2_inv);

    Tensor *dependency_arr[] = {t1, t2_inv};
    size_t dependency_cnt = sizeof(dependency_arr) / sizeof(dependency_arr[0]);

    return result;
}

Tensor *tensor_neg(Tensor *tensor) {
    ndArray *data = get_tensor_data(tensor);
    ndArray *new_data = negative(data);
    bool required_grad = get_requires_grad(tensor);

    Tensor *new_tensor = tensor_init(new_data, required_grad);

    return new_tensor;
}

Tensor *tensor_inv(Tensor *tensor) {
    ndArray *data = get_tensor_data(tensor);
    ndArray *new_data = inverse(data);
    bool required_grad = get_requires_grad(tensor);

    Tensor *new_tensor = tensor_init(new_data, required_grad);

    return new_tensor;
}
