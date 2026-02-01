#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <stdbool.h>

Tensor *tensor_transpose(Tensor *tensor, const int *dims) {
    ndArray *_data = get_tensor_data(tensor);
    bool requires_grad = get_requires_grad(tensor);
    Environment *env = get_tensor_environ(tensor);

    ndArray *data = transpose(_data, dims);
    Tensor *new_tensor = tensor_init(data, requires_grad, env);

    if (requires_grad) {
        BackwardFn *backward_fn = TransposeBackward((Tensor *[]){new_tensor},
                                                    (Tensor *[]){tensor}, 1, 1);

        TransposeCtx ctx = {.ndim = get_tensor_ndim(tensor),
                            .dims = (int *)dims};
        set_ctx(backward_fn, &ctx, TRANSPOSE_CTX);
        set_backward_fn(new_tensor, backward_fn);
    }

    return new_tensor;
}

Tensor *tensor_matmul(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = matmul(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad, get_tensor_environ(t1));
    if (requires_grad) {
        BackwardFn *backward_fn =
            MatMulBackward((Tensor *[]){tensor}, (Tensor *[]){t1, t2}, 1, 2);
        set_backward_fn(tensor, backward_fn);
    }

    return tensor;
}
