#include "tensor.h"
#include "array.h"
#include "autograd.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

const char *CtxNames[] = {
    "DEFAULT_CTX_TYPE",
    "SINGLE_TENSOR_CTX",
    "TWO_TENSOR_CTX",
};

struct Tensor {
    ndArray *data;
    ndArray *grad;
    Dependency **dependency_arr;
    size_t dependency_cnt;
    bool requires_grad;
    void *ctx;
    CtxType ctx_type;
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
    tensor->requires_grad = requires_grad;
    tensor->ctx = NULL;
    tensor->ctx_type = DEFAULT_CTX_TYPE;

    return tensor;
}

void free_tensor(Tensor *tensor) {
    free_array(tensor->data);
    free_array(tensor->grad);
    if (tensor->dependency_arr) {
        for (size_t i = 0; i < tensor->dependency_cnt; i++)
            free_dependency(tensor->dependency_arr[i]);
        free(tensor->dependency_arr);
    }
    free_context(tensor->ctx, tensor->ctx_type);
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
bool get_requires_grad(const Tensor *tensor) { return tensor->requires_grad; }
size_t get_dependency_cnt(const Tensor *tensor) {
    return tensor->dependency_cnt;
}

int get_tensor_ndim(const Tensor *tensor) { return get_ndim(tensor->data); }
size_t *get_tensor_shape(const Tensor *tensor) {
    return get_shape(tensor->data);
}
void *get_tensor_ctx(const Tensor *tensor) { return tensor->ctx; }
CtxType get_tensor_ctx_type(const Tensor *tensor) { return tensor->ctx_type; }

void set_requires_grad(Tensor *tensor, bool requires_grad) {
    tensor->requires_grad = requires_grad;
}

void set_dependency_arr(Tensor *tensor, Dependency **dependency_arr,
                        size_t dependency_cnt) {
    tensor->dependency_cnt = dependency_cnt;
    tensor->dependency_arr = dependency_arr;
}
void set_tensor_ctx(Tensor *tensor, void *ctx, CtxType ctx_type) {
    tensor->ctx = ctx;
    tensor->ctx_type = ctx_type;
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

    array_addi(&tensor->grad, grad);
    for (size_t i = 0; i < tensor->dependency_cnt; i++) {
        Dependency *dep = tensor->dependency_arr[i];
        Tensor *dep_tensor = get_dependency_tensor(dep);
        gradFn grad_fn = get_dependency_grad_fn(dep);

        ndArray *backward_grad = grad_fn(tensor->grad, tensor->ctx);
        backward(dep_tensor, backward_grad);
    }

    free_array(grad);
}
