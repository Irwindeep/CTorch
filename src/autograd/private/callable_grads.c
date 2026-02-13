#include "callable_grads.h"
#include "array.h"
#include "autograd.h"
#include "error_codes.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define _DEFINE_GRAD_FN(name, n_inputs, n_outputs, ...)                        \
    void name(Tensor **output_grads, Tensor **inputs, Tensor **outputs,        \
              Tensor **input_grads, size_t num_inputs, size_t num_outputs,     \
              bool create_graph) {                                             \
        if (num_inputs != n_inputs || num_outputs != n_outputs) {              \
            RUNTIME_ERRORF(INVALID_NUM_INPUTS_OUTPUTS,                         \
                           "Invalid number of inputs (%zu, expected %zu) or "  \
                           "outputs (%zu, expected %zu) in function `%s`",     \
                           num_inputs, (size_t)n_inputs, num_outputs,          \
                           (size_t)n_outputs, #name);                          \
        }                                                                      \
        __VA_ARGS__                                                            \
    }

_DEFINE_GRAD_FN(_accumulate_grad_fn, 1, 0, {
    if (create_graph) {
        return;
    }

    Tensor *tensor = inputs[0], *grad = input_grads[0];
    Environment *env = get_tensor_environ(tensor);

    Tensor *tensor_grad = get_tensor_grad(tensor);
    if (!tensor_grad) {
        zero_grad(tensor);
        tensor_grad = get_tensor_grad(tensor);
    }

    ndArray *sum =
        array_add(get_tensor_data(tensor_grad), get_tensor_data(grad));

    replace_tensor_data(tensor_grad, sum);
})

#define BLOCK(...) {__VA_ARGS__}

#define _ONE_IP_TWO_OP_GRAD_FN(name, CG_T1_BLOCK, CG_T2_BLOCK, NG_T1_BLOCK,    \
                               NG_T2_BLOCK)                                    \
    _DEFINE_GRAD_FN(name, 1, 2, {                                              \
        Tensor *tensor = inputs[0], *grad = input_grads[0];                    \
        Tensor *t1 = outputs[0], *t2 = outputs[1];                             \
        ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);    \
        ndArray *grad_data = get_tensor_data(grad);                            \
                                                                               \
        size_t t1_ndim = get_tensor_ndim(t1), t2_ndim = get_tensor_ndim(t2);   \
        const size_t *t1_shape = get_tensor_shape(t1),                         \
                     *t2_shape = get_tensor_shape(t2);                         \
        bool t1_requires_grad = get_requires_grad(t1),                         \
             t2_requires_grad = get_requires_grad(t2);                         \
                                                                               \
        Environment *env = get_tensor_environ(tensor);                         \
                                                                               \
        Tensor *t1_grad = NULL, *t2_grad = NULL;                               \
        if (create_graph) {                                                    \
            if (t1_requires_grad)                                              \
                CG_T1_BLOCK                                                    \
            if (t2_requires_grad)                                              \
                CG_T2_BLOCK                                                    \
        } else {                                                               \
            if (t1_requires_grad)                                              \
                NG_T1_BLOCK                                                    \
            if (t2_requires_grad)                                              \
                NG_T2_BLOCK                                                    \
        }                                                                      \
        output_grads[0] = t1_grad;                                             \
        output_grads[1] = t2_grad;                                             \
    })

_ONE_IP_TWO_OP_GRAD_FN(
    _add_grad_fn,
    BLOCK({ t1_grad = broadcast_tensor_grad(grad, t1_ndim, t1_shape); }),
    BLOCK({ t2_grad = broadcast_tensor_grad(grad, t2_ndim, t2_shape); }),

    BLOCK({
        ndArray *data1_grad = copy_array(grad_data);
        data1_grad = broadcast_grad_data(data1_grad, t1_ndim, t1_shape);
        t1_grad = tensor_init(data1_grad, NO_GRAD, env);
    }),
    BLOCK({
        ndArray *data2_grad = copy_array(grad_data);
        data2_grad = broadcast_grad_data(data2_grad, t2_ndim, t2_shape);
        t2_grad = tensor_init(data2_grad, NO_GRAD, env);
    }))

_ONE_IP_TWO_OP_GRAD_FN(
    _mul_grad_fn, BLOCK({
        t1_grad = tensor_mul(t2, grad);
        t1_grad = broadcast_tensor_grad(t1_grad, t1_ndim, t1_shape);
    }),
    BLOCK({
        t2_grad = tensor_mul(t1, grad);
        t2_grad = broadcast_tensor_grad(t2_grad, t2_ndim, t2_shape);
    }),

    BLOCK({
        ndArray *data1_grad = array_mul(data2, grad_data);
        data1_grad = broadcast_grad_data(data1_grad, t1_ndim, t1_shape);
        t1_grad = tensor_init(data1_grad, NO_GRAD, env);
    }),
    BLOCK({
        ndArray *data2_grad = array_mul(data1, grad_data);
        data2_grad = broadcast_grad_data(data2_grad, t2_ndim, t2_shape);
        t2_grad = tensor_init(data2_grad, NO_GRAD, env);
    }))

_DEFINE_GRAD_FN(_neg_grad_fn, 1, 1, {
    Tensor *new_tensor = inputs[0], *grad = input_grads[0];

    Environment *env = get_tensor_environ(new_tensor);
    if (create_graph) {
        output_grads[0] = tensor_neg(grad);
    } else {
        ndArray *data_grad = negative(get_tensor_data(grad));
        output_grads[0] = tensor_init(data_grad, NO_GRAD, env);
    }
})

_DEFINE_GRAD_FN(_inv_grad_fn, 1, 1, {
    Tensor *new_tensor = inputs[0], *grad = input_grads[0];

    Environment *env = get_tensor_environ(new_tensor);
    if (create_graph) {
        Tensor *tensor_grad = tensor_mul(new_tensor, new_tensor);
        tensor_grad = tensor_mul(tensor_grad, grad);
        output_grads[0] = tensor_neg(tensor_grad);
    } else {
        ndArray *new_data = get_tensor_data(new_tensor);

        ndArray *data_grad = array_mul(new_data, new_data);
        array_muli(&data_grad, get_tensor_data(grad));
        negativei(&data_grad);

        output_grads[0] = tensor_init(data_grad, NO_GRAD, env);
    }
})

_DEFINE_GRAD_FN(_transpose_grad_fn, 1, 1, {
    Tensor *new_tensor = inputs[0], *grad = input_grads[0];

    BackwardFn *backward_fn = get_backward_fn(new_tensor);
    Ctx ctx_kind = get_ctx_kind(backward_fn);
    if (ctx_kind != TRANSPOSE_CTX) {
        RUNTIME_ERRORF(INVALID_BACKWARD_PASS, "Invalid context kind for `%s`",
                       __func__);
    }
    TransposeCtx *ctx = (TransposeCtx *)get_ctx(backward_fn);
    int *dims = ctx->dims;

    Environment *env = get_tensor_environ(new_tensor);
    if (create_graph) {
        output_grads[0] = tensor_transpose_env(grad, dims, env);
    } else {
        ndArray *data_grad = transpose(get_tensor_data(grad), dims);
        output_grads[0] = tensor_init(data_grad, NO_GRAD, env);
    }
})

static void inline _get_dims_for_matmul_grad(Tensor *t, int *dims) {
    int ndim = get_tensor_ndim(t);
    for (int d = 0; d < ndim; d++)
        dims[d] = d;

    dims[ndim - 1] = ndim - 2;
    dims[ndim - 2] = ndim - 1;
}

_ONE_IP_TWO_OP_GRAD_FN(
    _matmul_grad_fn, BLOCK({
        int t2_dims[t2_ndim];
        _get_dims_for_matmul_grad(t2, t2_dims);
        Tensor *t2_T = tensor_transpose_env(t2, t2_dims, env);
        t1_grad = tensor_matmul(grad, t2_T);
        t1_grad = broadcast_tensor_grad(t1_grad, t1_ndim, t1_shape);
    }),
    BLOCK({
        int t1_dims[t1_ndim];
        _get_dims_for_matmul_grad(t1, t1_dims);
        Tensor *t1_T = tensor_transpose_env(t1, t1_dims, env);
        t2_grad = tensor_matmul(t1_T, grad);
        t2_grad = broadcast_tensor_grad(t2_grad, t2_ndim, t2_shape);
    }),

    BLOCK({
        int t2_dims[t2_ndim];
        _get_dims_for_matmul_grad(t2, t2_dims);
        ndArray *data2_T = transpose(data2, t2_dims);

        ndArray *data1_grad = matmul(grad_data, data2_T);
        data1_grad = broadcast_grad_data(data1_grad, t1_ndim, t1_shape);
        t1_grad = tensor_init(data1_grad, NO_GRAD, env);
        free_array(data2_T);
    }),
    BLOCK({
        int t1_dims[t1_ndim];
        _get_dims_for_matmul_grad(t1, t1_dims);
        ndArray *data1_T = transpose(data1, t1_dims);

        ndArray *data2_grad = matmul(data1_T, grad_data);
        data2_grad = broadcast_grad_data(data2_grad, t2_ndim, t2_shape);
        t2_grad = tensor_init(data2_grad, NO_GRAD, env);
        free_array(data1_T);
    }))

_DEFINE_GRAD_FN(_sum_grad_fn, 1, 1, {
    Tensor *new_tensor = inputs[0], *grad = input_grads[0];
    Tensor *tensor = outputs[0];

    Environment *env = get_tensor_environ(new_tensor);
    if (create_graph) {
        output_grads[0] = tensor_mul(grad, ones_like(tensor, NO_GRAD, env));
    } else {
        int ndim = get_tensor_ndim(tensor);
        const size_t *shape = get_tensor_shape(tensor);
        DType dtype = get_tensor_dtype(tensor);

        ndArray *ones_arr = ones(ndim, shape, dtype);

        ndArray *data_grad = array_mul(get_tensor_data(grad), ones_arr);
        output_grads[0] = tensor_init(data_grad, NO_GRAD, env);

        free_array(ones_arr);
    }
})

_ONE_IP_TWO_OP_GRAD_FN(
    _max_grad_fn, BLOCK({
        Tensor *t1_ge_t2 = tensor_ge(t1, t2);
        t1_grad = tensor_mul(grad, t1_ge_t2);
        t1_grad = broadcast_tensor_grad(t1_grad, t1_ndim, t1_shape);
    }),
    BLOCK({
        Tensor *t2_gt_t1 = tensor_gt(t2, t1);
        t2_grad = tensor_mul(grad, t2_gt_t1);
        t2_grad = broadcast_tensor_grad(t2_grad, t2_ndim, t2_shape);
    }),
    BLOCK({
        ndArray *data1_ge_data2 = array_ge(data1, data2);
        ndArray *data1_grad = array_mul(grad_data, data1_ge_data2);
        data1_grad = broadcast_grad_data(data1_grad, t1_ndim, t1_shape);

        t1_grad = tensor_init(data1_grad, NO_GRAD, env);
        free_array(data1_ge_data2);
    }),
    BLOCK({
        ndArray *data2_gt_data1 = array_gt(data2, data1);
        ndArray *data2_grad = array_mul(grad_data, data2_gt_data1);
        data2_grad = broadcast_grad_data(data2_grad, t2_ndim, t2_shape);

        t2_grad = tensor_init(data2_grad, NO_GRAD, env);
        free_array(data2_gt_data1);
    }))

_ONE_IP_TWO_OP_GRAD_FN(
    _min_grad_fn, BLOCK({
        Tensor *t1_le_t2 = tensor_le(t1, t2);
        t1_grad = tensor_mul(grad, t1_le_t2);
        t1_grad = broadcast_tensor_grad(t1_grad, t1_ndim, t1_shape);
    }),
    BLOCK({
        Tensor *t2_lt_t1 = tensor_lt(t2, t1);
        t2_grad = tensor_mul(grad, t2_lt_t1);
        t2_grad = broadcast_tensor_grad(t2_grad, t2_ndim, t2_shape);
    }),
    BLOCK({
        ndArray *data1_le_data2 = array_le(data1, data2);
        ndArray *data1_grad = array_mul(grad_data, data1_le_data2);
        data1_grad = broadcast_grad_data(data1_grad, t1_ndim, t1_shape);

        t1_grad = tensor_init(data1_grad, NO_GRAD, env);
        free_array(data1_le_data2);
    }),
    BLOCK({
        ndArray *data2_lt_data1 = array_lt(data2, data1);
        ndArray *data2_grad = array_mul(grad_data, data2_lt_data1);
        data2_grad = broadcast_grad_data(data2_grad, t2_ndim, t2_shape);

        t2_grad = tensor_init(data2_grad, NO_GRAD, env);
        free_array(data2_lt_data1);
    }))
