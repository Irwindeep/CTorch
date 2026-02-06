#include "array.h"
#include "autograd.h"
#include "error_codes.h"
#include "tensor.h"

#include <CUnit/TestRun.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static inline ssize_t find_input(const Tensor *t, Tensor **inputs,
                                 size_t num_inputs) {
    for (size_t i = 0; i < num_inputs; i++)
        if (inputs[i] == t)
            return (ssize_t)i;
    return -1;
}

static void _gradient_backward(Tensor **inputs, Tensor **grads,
                               size_t num_inputs, Tensor **cur_inputs,
                               Tensor **cur_grads, BackwardFn *backward_fn,
                               bool create_graph) {
    size_t num_fn_inputs = get_backward_inputs(backward_fn);
    size_t num_fn_outputs = get_backward_outputs(backward_fn);

    Tensor **outputs = get_backward_fn_op_tensors(backward_fn);
    BackwardFn **next_fns = get_next_functions(backward_fn);

    CallableGradFn grad_fn = get_grad_fn(backward_fn);

    Tensor *op_grads[num_fn_outputs];
    printf("<%s %p>\n", get_backward_name(backward_fn), backward_fn);
    grad_fn(op_grads, cur_inputs, outputs, cur_grads, num_fn_inputs,
            num_fn_outputs, create_graph);

    for (size_t i = 0; i < num_fn_outputs; i++) {
        const Tensor *out = outputs[i];
        Tensor *g = op_grads[i];

        ssize_t idx = find_input(out, inputs, num_inputs);
        if (idx >= 0) {
            grads[idx] = grads[idx] ? tensor_add(grads[idx], g) : g;
            continue;
        }

        BackwardFn *next_fn = next_fns[i];
        if (!next_fn)
            continue;

        Tensor **next_inputs = get_backward_fn_ip_tensors(next_fn);
        size_t next_n = get_backward_inputs(next_fn);

        Tensor *next_grads[next_n];
        for (size_t j = 0; j < next_n; j++)
            next_grads[j] = g;

        _gradient_backward(inputs, grads, num_inputs, next_inputs, next_grads,
                           next_fn, create_graph);
    }
}

void gradient(Tensor **grads, size_t num_inputs, Tensor **inputs,
              size_t num_outputs, Tensor **outputs, Tensor **grad_outputs,
              bool create_graph) {
    for (size_t i = 0; i < num_inputs; i++) {
        bool requires_grad = get_requires_grad(inputs[i]);
        if (!requires_grad)
            RUNTIME_ERRORF(INVALID_BACKWARD_PASS,
                           "Input tensor at index `%zu` does not requires_grad",
                           i);
    }
    for (size_t i = 0; i < num_outputs; i++) {
        bool requires_grad = get_requires_grad(outputs[i]);
        if (!requires_grad)
            RUNTIME_ERRORF(
                INVALID_BACKWARD_PASS,
                "Output tensor at index `%zu` does not requires_grad", i);
    }

    for (size_t i = 0; i < num_outputs; i++) {
        BackwardFn *fn = get_backward_fn(outputs[i]);
        if (!fn)
            continue;

        _gradient_backward(inputs, grads, num_inputs, (Tensor *[]){outputs[i]},
                           (Tensor *[]){grad_outputs[i]}, fn, create_graph);
    }
}

static inline void _backward(Tensor **inputs, Tensor **grads,
                             BackwardFn *backward_fn) {
    size_t num_inputs = get_backward_inputs(backward_fn),
           num_outputs = get_backward_outputs(backward_fn);

    Tensor **outputs = get_backward_fn_op_tensors(backward_fn);
    BackwardFn **next_fns = get_next_functions(backward_fn);

    CallableGradFn grad_fn = get_grad_fn(backward_fn);
    Tensor *op_grads[num_outputs];
    grad_fn(op_grads, inputs, outputs, grads, num_inputs, num_outputs, false);

    size_t i = 0;
    while (i < num_outputs) {
        BackwardFn *next_fn = next_fns[i];
        if (!next_fn) {
            i++;
            continue;
        }

        Tensor **next_fn_inputs = get_backward_fn_ip_tensors(next_fn);
        size_t next_fn_num_ips = get_backward_inputs(next_fn);

        Tensor *next_fn_ip_grads[next_fn_num_ips];
        for (size_t j = 0; j < next_fn_num_ips; j++)
            next_fn_ip_grads[j] = op_grads[i++];

        _backward(next_fn_inputs, next_fn_ip_grads, next_fn);
    }
}

void backward(Tensor *tensor, Tensor *grad) {
    bool requires_grad = get_requires_grad(tensor);
    if (!requires_grad)
        RUNTIME_ERROR(INVALID_BACKWARD_PASS,
                      "Invalid backward pass on non-requires_grad tensor");

    if (!grad) {
        size_t ndim = get_tensor_ndim(tensor);
        size_t *shape = get_tensor_shape(tensor);
        DType dtype = get_tensor_dtype(tensor);

        if (ndim != 0)
            RUNTIME_ERROR(GRAD_INIT_FAILURE,
                          "Invalid backward gradient for non-zero dim tensor");

        grad = ones_tensor(ndim, shape, dtype, false, NULL);
    }

    BackwardFn *backward_fn = get_backward_fn(tensor);
    _backward((Tensor *[]){tensor}, (Tensor *[]){grad}, backward_fn);

    const Environment *environ = get_tensor_environ(grad);
    if (!environ)
        free_tensor(grad);
}
