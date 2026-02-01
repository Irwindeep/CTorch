#include "callable_grads.h"
#include "array.h"
#include "error_codes.h"
#include "tensor.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

Tensor **_accumulate_grad_fn(Tensor **inputs, Tensor **outputs,
                             Tensor **input_grads, size_t num_inputs,
                             size_t num_outputs, bool create_graph) {
    if (num_inputs != 1 || num_outputs != 0)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor *tensor = inputs[0];
    Tensor *grad = input_grads[0];

    Tensor *tensor_grad = get_tensor_grad(tensor);
    if (!tensor_grad) {
        zero_grad(tensor);
        tensor_grad = get_tensor_grad(tensor);
    }

    if (create_graph)
        RUNTIME_ERROR(
            INVALID_GRAD,
            "AccumlateGrad function called with parameter `create_graph=True`");

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
    if (!output_grads) {
        printf("Failure to allocate gradients tensor\n");
        exit(GRAD_INIT_FAILURE);
    }
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
    if (!output_grads) {
        printf("Failure to allocate gradients tensor\n");
        exit(GRAD_INIT_FAILURE);
    }
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
    if (!output_grads) {
        printf("Failure to allocate gradients tensor\n");
        exit(GRAD_INIT_FAILURE);
    }
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
