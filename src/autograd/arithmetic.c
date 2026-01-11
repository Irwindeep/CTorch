#include "array.h"
#include "autograd.h"

ndArray *_add_grad_fn1(ndArray *grad, void *ctx) {
    TwoTensorCtx *_ctx = (TwoTensorCtx *)ctx;
    Tensor *t1 = get_tensor1_ttc(_ctx);

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t1);
    for (int i = 0; i < ndims_added; i++)
        array_sum_dimi(&grad, 0, false);

    int t1_ndim = get_tensor_ndim(t1);
    const size_t *t1_shape = get_tensor_shape(t1);
    for (int i = 0; i < t1_ndim; i++) {
        if (t1_shape[i] == 1)
            array_sum_dimi(&grad, i, true);
    }

    return grad;
}

ndArray *_add_grad_fn2(ndArray *grad, void *ctx) {
    TwoTensorCtx *_ctx = (TwoTensorCtx *)ctx;
    Tensor *t2 = get_tensor2_ttc(_ctx);

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t2);
    for (int i = 0; i < ndims_added; i++)
        array_sum_dimi(&grad, 0, false);

    int t2_ndim = get_tensor_ndim(t2);
    const size_t *t2_shape = get_tensor_shape(t2);
    for (int i = 0; i < t2_ndim; i++) {
        if (t2_shape[i] == 1)
            array_sum_dimi(&grad, i, true);
    }

    return grad;
}

ndArray *_mul_grad_fn1(ndArray *grad, void *ctx) {
    TwoTensorCtx *_ctx = (TwoTensorCtx *)ctx;
    Tensor *t1 = get_tensor1_ttc(_ctx), *t2 = get_tensor2_ttc(_ctx);
    grad = array_mul(grad, get_tensor_data(t2));

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t1);
    for (int i = 0; i < ndims_added; i++) {
        array_sum_dimi(&grad, 0, false);
    }

    int t1_ndim = get_tensor_ndim(t1);
    const size_t *t1_shape = get_tensor_shape(t1);
    for (int i = 0; i < t1_ndim; i++) {
        if (t1_shape[i] == 1)
            array_sum_dimi(&grad, i, true);
    }

    return grad;
}
ndArray *_mul_grad_fn2(ndArray *grad, void *ctx) {
    TwoTensorCtx *_ctx = (TwoTensorCtx *)ctx;
    Tensor *t1 = get_tensor1_ttc(_ctx), *t2 = get_tensor2_ttc(_ctx);
    grad = array_mul(grad, get_tensor_data(t1));

    int ndims_added = get_ndim(grad) - get_tensor_ndim(t2);
    for (int i = 0; i < ndims_added; i++)
        array_sum_dimi(&grad, 0, false);

    int t2_ndim = get_tensor_ndim(t2);
    const size_t *t2_shape = get_tensor_shape(t2);
    for (int i = 0; i < t2_ndim; i++) {
        if (t2_shape[i] == 1)
            array_sum_dimi(&grad, i, true);
    }

    return grad;
}

ndArray *_neg_grad_fn(ndArray *grad, void *ctx) { return negative(grad); }

ndArray *_inv_grad_fn(ndArray *grad, void *ctx) {
    SingleTensorCtx *_ctx = (SingleTensorCtx *)ctx;
    Tensor *tensor = get_tensor_stc(_ctx);

    ndArray *data = get_tensor_data(tensor);
    ndArray *data_inv = inverse(data);
    ndArray *data_2 = array_mul(data_inv, data_inv);

    grad = array_mul(grad, data_2);

    free_array(data_inv);
    free_array(data_2);

    negativei(&grad);
    return grad;
}
