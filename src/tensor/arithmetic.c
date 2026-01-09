#include "array.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ArithmeticCtx {
    Tensor *t1;
    Tensor *t2;
} ArithmeticCtx;

static ndArray *_add_grad_fn1(ndArray *grad, void *ctx) {
    ArithmeticCtx *_ctx = (ArithmeticCtx *)ctx;
    Tensor *t1 = _ctx->t1;

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t1);
    for (int i = 0; i < ndims_added; i++)
        array_sum_dim(grad, 0, false);

    int t1_ndim = get_tensor_ndim(t1);
    const size_t *t1_shape = get_tensor_shape(t1);
    for (int i = 0; i < t1_ndim; i++) {
        if (t1_shape[i] == 1)
            array_sum_dim(grad, i, true);
    }

    return grad;
}
static ndArray *_add_grad_fn2(ndArray *grad, void *ctx) {
    ArithmeticCtx *_ctx = (ArithmeticCtx *)ctx;
    Tensor *t2 = _ctx->t2;

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t2);
    for (int i = 0; i < ndims_added; i++)
        array_sum_dim(grad, 0, false);

    int t2_ndim = get_tensor_ndim(t2);
    const size_t *t2_shape = get_tensor_shape(t2);
    for (int i = 0; i < t2_ndim; i++) {
        if (t2_shape[i] == 1)
            array_sum_dim(grad, i, true);
    }

    return grad;
}

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);
    ndArray *data = add(data1, data2);

    bool t1_requires_grad = get_requires_grad(t1),
         t2_requires_grad = get_requires_grad(t2);
    bool requires_grad = t1_requires_grad || t2_requires_grad;

    Tensor *tensor = tensor_init(data, requires_grad);

    if (requires_grad) {
        size_t dependency_cnt = (t1_requires_grad) ? 1 : 0;
        dependency_cnt += (t2_requires_grad) ? 1 : 0;

        Dependency **dependency_arr = dependency_arr_init(dependency_cnt);
        size_t idx = 0;

        ArithmeticCtx *ctx = malloc(sizeof(ArithmeticCtx));
        if (!ctx) {
            printf("Failure to create gradient context\n");
            exit(GRAD_CTX_FAILURE);
        }
        ctx->t1 = t1, ctx->t2 = t2;

        if (t1_requires_grad) {
            Dependency *dep = create_dependency(t1, _add_grad_fn1, ctx);
            dependency_arr[idx++] = dep;
        }
        if (t2_requires_grad) {
            Dependency *dep = create_dependency(t2, _add_grad_fn2, ctx);
            dependency_arr[idx] = dep;
        }

        set_dependency_arr(tensor, dependency_arr, dependency_cnt);
    }

    return tensor;
}
