#include "callable_grads.h"
#include "array.h"
#include "autograd.h"
#include "error_codes.h"
#include "tensor.h"

#include <stddef.h>
#include <stdlib.h>

Tensor **_accumulate_grad_fn(Tensor **inputs, Tensor **outputs,
                             Tensor **input_grads, size_t num_inputs,
                             size_t num_outputs, bool create_graph) {
    if (num_inputs != 1 || num_outputs != 0)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    if (create_graph)
        return NULL;

    Tensor *tensor = inputs[0];
    Tensor *grad = input_grads[0];

    Tensor *tensor_grad = get_tensor_grad(tensor);
    if (!tensor_grad) {
        zero_grad(tensor);
        tensor_grad = get_tensor_grad(tensor);
    }

    ndArray *sum =
        array_add(get_tensor_data(tensor_grad), get_tensor_data(grad));

    set_tensor_grad(tensor,
                    tensor_init(sum, false, get_tensor_environ(tensor)));
    return NULL;
}

Tensor **_add_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs,
                      bool create_graph) {
    if (num_inputs != 1 || num_outputs != 2)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    ndArray *grad = get_tensor_data(input_grads[0]);

    for (size_t i = 0; i < num_outputs; i++) {
        int ndim = get_tensor_ndim(outputs[i]);
        const size_t *shape = get_tensor_shape(outputs[i]);

        Tensor *t;
        if (create_graph) {
            Tensor *grad_tensor = input_grads[0];
            t = broadcast_tensor_grad(grad_tensor, ndim, shape);
        } else {
            ndArray *data = copy_array(grad);
            data = broadcast_grad_data(data, ndim, shape);

            t = tensor_init(data, false, get_tensor_environ(outputs[i]));
        }
        output_grads[i] = t;
    }

    return output_grads;
}

Tensor **_mul_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs,
                      bool create_graph) {
    if (num_inputs != 1 || num_outputs != 2)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    ndArray *grad = get_tensor_data(input_grads[0]);
    Tensor *grad_tensor = input_grads[0];

    for (size_t i = 0; i < num_outputs; i++) {
        int ndim = get_tensor_ndim(outputs[i]);
        const size_t *shape = get_tensor_shape(outputs[i]);

        Tensor *t;
        if (create_graph) {
            Tensor *other_tensor = outputs[num_outputs - i - 1];
            t = tensor_mul(grad_tensor, other_tensor);
            t = broadcast_tensor_grad(t, ndim, shape);
        } else {
            ndArray *other_arr = get_tensor_data(outputs[num_outputs - i - 1]);
            ndArray *data = copy_array(grad);
            array_muli(&data, other_arr);

            data = broadcast_grad_data(data, ndim, shape);
            t = tensor_init(data, false, get_tensor_environ(outputs[i]));
        }
        output_grads[i] = t;
    }

    return output_grads;
}

Tensor **_neg_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs,
                      bool create_graph) {
    if (num_inputs != 1 || num_outputs != 1)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    ndArray *grad = get_tensor_data(input_grads[0]);

    Tensor *t;
    if (create_graph) {
        Tensor *grad_tensor = input_grads[0];
        t = tensor_neg(grad_tensor);
    } else {
        ndArray *data = copy_array(grad);
        negativei(&data);

        t = tensor_init(data, false, get_tensor_environ(outputs[0]));
    }
    output_grads[0] = t;

    return output_grads;
}

Tensor **_inv_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs,
                      bool create_graph) {
    if (num_inputs != 1 || num_outputs != 1)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    ndArray *grad = get_tensor_data(input_grads[0]);

    Tensor *t;
    if (create_graph) {
        Tensor *grad_tensor = input_grads[0];

        t = inputs[0];
        t = tensor_mul(t, grad_tensor);
        t = tensor_mul(t, t);
        t = tensor_neg(t);
    } else {
        ndArray *data = copy_array(get_tensor_data(inputs[0]));
        array_muli(&data, grad);
        array_muli(&data, data);
        negativei(&data);

        t = tensor_init(data, false, get_tensor_environ(outputs[0]));
    }
    output_grads[0] = t;

    return output_grads;
}

Tensor **_transpose_grad_fn(Tensor **inputs, Tensor **outputs,
                            Tensor **input_grads, size_t num_inputs,
                            size_t num_outputs, bool create_graph) {
    if (num_inputs != 1 || num_outputs != 1)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    BackwardFn *bk_fn = get_backward_fn(inputs[0]);
    Ctx ctx_kind = get_ctx_kind(bk_fn);
    if (ctx_kind != TRANSPOSE_CTX)
        RUNTIME_ERROR(INVALID_BACKWARD_PASS,
                      "Invalid context kind for _transpose_grad_fn");

    TransposeCtx *ctx = (TransposeCtx *)get_ctx(bk_fn);
    int *dims = ctx->dims;

    Tensor *t;
    if (create_graph) {
        Tensor *grad_tensor = input_grads[0];
        t = tensor_transpose(grad_tensor, dims);
    } else {
        ndArray *grad = get_tensor_data(input_grads[0]);
        ndArray *data = transpose(grad, dims);
        t = tensor_init(data, false, get_tensor_environ(outputs[0]));
    }
    output_grads[0] = t;

    return output_grads;
}

Tensor **_matmul_grad_fn(Tensor **inputs, Tensor **outputs,
                         Tensor **input_grads, size_t num_inputs,
                         size_t num_outputs, bool create_graph) {
    if (num_inputs != 1 || num_outputs != 2)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor **output_grads = malloc(num_outputs * sizeof(Tensor *));
    if (!output_grads)
        RUNTIME_ERROR(GRAD_INIT_FAILURE, "Failure to allocate gradient tensor");

    ndArray *grad = get_tensor_data(input_grads[0]);
    Tensor *grad_tensor = input_grads[0];

    for (size_t i = 0; i < num_outputs; i++) {
        int ndim = get_tensor_ndim(outputs[i]);
        const size_t *shape = get_tensor_shape(outputs[i]);

        int ndim_oth = get_tensor_ndim(outputs[num_outputs - i - 1]);
        int dims[ndim_oth] = {};
        for (int d = 0; d < ndim_oth; d++)
            dims[d] = d;

        dims[ndim_oth - 1] = ndim_oth - 2;
        dims[ndim_oth - 2] = ndim_oth - 1;

        Tensor *t;
        if (create_graph) {
            Tensor *other_tensor = outputs[num_outputs - i - 1];
            Tensor *other_tensor_T = tensor_transpose(other_tensor, dims);
            Tensor *args[2] = {grad_tensor, other_tensor_T};

            t = tensor_matmul(args[i], args[1 - i]);
            t = broadcast_tensor_grad(t, ndim, shape);
        } else {
            ndArray *other_arr = get_tensor_data(outputs[num_outputs - i - 1]);
            ndArray *data = copy_array(grad);

            ndArray *other_arr_T = transpose(other_arr, dims);

            ndArray *tmp = data;
            ndArray *args[2] = {tmp, other_arr_T};

            data = matmul(args[i], args[1 - i]);
            data = broadcast_grad_data(data, ndim, shape);

            free_array(tmp);
            free_array(other_arr_T);
            t = tensor_init(data, false, get_tensor_environ(outputs[i]));
        }
        output_grads[i] = t;
    }

    return output_grads;
}
