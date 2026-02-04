#include "array.h"
#include "autograd.h"
#include "callable_grads.h"
#include "error_codes.h"
#include "tensor.h"

#include <stddef.h>
#include <stdlib.h>

ndArray *broadcast_grad_data(ndArray *data, int ndim, const size_t *shape) {
    int ndims_added = get_ndim(data) - ndim;
    if (ndims_added < 0)
        return data;

    for (int i = 0; i < ndims_added; i++) {
        ndArray *tmp = data;
        data = array_sum_dim(data, 0, false);
        free_array(tmp);
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] == 1) {
            ndArray *tmp = data;
            data = array_sum_dim(data, i, true);
            free_array(tmp);
        }
    }

    return data;
}

static inline void _broadcast_grad_fn(Tensor **output_grads, Tensor **inputs,
                                      Tensor **outputs, Tensor **input_grads,
                                      size_t num_inputs, size_t num_outputs,
                                      bool create_graph) {
    if (num_inputs != 1 || num_outputs != 1)
        RUNTIME_ERROR(INVALID_NUM_INPUTS_OUTPUTS,
                      "Invalid number of inputs/outputs");

    Tensor *grad_tensor = input_grads[0];

    int ndim = get_tensor_ndim(outputs[0]);
    const size_t *shape = get_tensor_shape(outputs[0]);

    Tensor *t;
    if (create_graph) {
        t = broadcast_tensor_grad(grad_tensor, ndim, shape);
    } else {
        ndArray *grad = get_tensor_data(grad_tensor);
        ndArray *data = copy_array(grad);

        data = broadcast_grad_data(data, ndim, shape);
        t = tensor_init(data, false, get_tensor_environ(outputs[0]));
    }

    output_grads[0] = t;
}

static inline BackwardFn *BroadcastBackward(Tensor **inputs, Tensor **outputs,
                                            size_t num_inputs,
                                            size_t num_outputs) {
    CallableGradFn grad_fn = _broadcast_grad_fn;
    BackwardFn *backward_fn = backward_fn_init(
        grad_fn, inputs, outputs, num_inputs, num_outputs, "BroadcastBackward");

    BackwardFn **next_functions = create_next_fns(outputs, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}

Tensor *broadcast_tensor_grad(Tensor *tensor, int ndim, const size_t *shape) {
    ndArray *data =
        broadcast_grad_data(copy_array(get_tensor_data(tensor)), ndim, shape);
    bool requires_grad = get_requires_grad(tensor);
    Environment *env = get_tensor_environ(tensor);

    Tensor *t = tensor_init(data, requires_grad, env);
    if (requires_grad) {
        BackwardFn *backward_fn =
            BroadcastBackward((Tensor *[]){t}, (Tensor *[]){tensor}, 1, 1);
        set_backward_fn(t, backward_fn);
    }

    return t;
}
