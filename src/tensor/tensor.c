#include "tensor.h"
#include "array.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Tensor {
    ndArray *data;
    ndArray *grad;
    Dependency **dependency_arr;
    size_t dependency_cnt;
    bool requires_grad;
};

Tensor *tensor_init(ndArray *data, bool requires_grad) {
    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor) {
        printf("Failure to allocate tensor\n");
        exit(TENSOR_INIT_FAILURE);
    }

    tensor->data = data;
    tensor->grad = NULL;
    if (requires_grad)
        zero_grad(tensor);

    tensor->dependency_arr = NULL;
    tensor->dependency_cnt = 0;

    return tensor;
}

void free_tensor(Tensor *tensor) {
    free_array(tensor->data);
    if (tensor->grad)
        free_array(tensor->grad);
    free(tensor->dependency_arr);
    free(tensor);
}

ndArray *get_tensor_data(const Tensor *tensor) { return tensor->data; }

ndArray *get_tensor_grad(const Tensor *tensor) {
    if (!tensor->requires_grad) {
        printf("Cannot have gradient for non-requires_grad Tensor.\n");
        exit(INVALID_GRAD);
    }
    return tensor->grad;
}

void zero_grad(Tensor *tensor) {
    int ndim = get_ndim(tensor->data);
    size_t *shape = get_shape(tensor->data);
    DType dtype = get_dtype(tensor->data);
    tensor->grad = zeros(ndim, shape, dtype);
}

void backward(Tensor *tensor, ndArray *grad) {
    if (!tensor->requires_grad) {
        printf("Invalid backward pass on non-requires_grad Tensor\n");
        exit(INVALID_BACKWARD);
    }
    if (!grad) {
        if (get_ndim(tensor->data) > 0) {
            printf("Invalid backward pass gradient\n");
            exit(INVALID_GRAD);
        }
        grad = ones(0, (size_t[]){}, get_dtype(tensor->data));
    }

    tensor->grad = add(tensor->grad, grad);
    for (size_t i = 0; i < tensor->dependency_cnt; i++) {
        Dependency *dep = tensor->dependency_arr[i];
        Tensor *dep_tensor = get_dependency_tensor(dep);
        gradFn grad_fn = get_dependency_grad_fn(dep);

        ndArray *backward_grad = grad_fn(tensor->grad);
        backward(dep_tensor, backward_grad);
    }
}
