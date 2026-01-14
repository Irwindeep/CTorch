#include "tensor.h"
#include "array.h"
#include "autograd.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Tensor {
    ndArray *data;
    Tensor *grad;
    BackwardFn *backward_fn;
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

    tensor->backward_fn = NULL;
    tensor->requires_grad = requires_grad;

    return tensor;
}

void free_tensor(Tensor *tensor) {
    if (!tensor)
        return;

    free_array(tensor->data);
    free_tensor(tensor->grad);
    free_backward_fn(tensor->backward_fn);

    free(tensor);
}

ndArray *get_tensor_data(const Tensor *tensor) { return tensor->data; }
Tensor *get_tensor_grad(const Tensor *tensor) {
    if (!tensor->requires_grad) {
        printf("Cannot have gradient for non-requires_grad Tensor.\n");
        exit(INVALID_GRAD);
    }
    return tensor->grad;
}
bool get_requires_grad(const Tensor *tensor) { return tensor->requires_grad; }
int get_tensor_ndim(const Tensor *tensor) { return get_ndim(tensor->data); }

size_t *get_tensor_shape(const Tensor *tensor) {
    return get_shape(tensor->data);
}

BackwardFn *get_backward_fn(const Tensor *tensor) {
    return tensor->backward_fn;
}

void set_requires_grad(Tensor *tensor, bool requires_grad) {
    tensor->requires_grad = requires_grad;
}

void set_tensor_grad(Tensor *tensor, Tensor *grad) { tensor->grad = grad; }
void set_backward_fn(Tensor *tensor, BackwardFn *backward_fn) {
    tensor->backward_fn = backward_fn;
}

void zero_grad(Tensor *tensor) {
    int ndim = get_ndim(tensor->data);
    const size_t *shape = get_shape(tensor->data);
    DType dtype = get_dtype(tensor->data);
    tensor->grad = zeros_tensor(ndim, shape, dtype, false);
}

Tensor *eye_tensor(size_t m, size_t n, DType dtype, bool requires_grad) {
    ndArray *data = eye(m, n, dtype);
    Tensor *tensor = tensor_init(data, requires_grad);

    return tensor;
}

Tensor *zeros_tensor(int ndim, const size_t *shape, DType dtype,
                     bool requires_grad) {
    ndArray *data = zeros(ndim, shape, dtype);
    Tensor *tensor = tensor_init(data, requires_grad);

    return tensor;
}

Tensor *ones_tensor(int ndim, const size_t *shape, DType dtype,
                    bool requires_grad) {
    ndArray *data = ones(ndim, shape, dtype);
    Tensor *tensor = tensor_init(data, requires_grad);

    return tensor;
}
