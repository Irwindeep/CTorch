#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <CUnit/TestRun.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

Tensor **gradient(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                  size_t num_inputs, size_t num_outputs);

static void _backward(Tensor **inputs, Tensor **grads,
                      BackwardFn *backward_fn) {
    size_t num_inputs = get_backward_inputs(backward_fn),
           num_outputs = get_backward_outputs(backward_fn);

    Tensor **outputs = get_backward_fn_op_tensors(backward_fn);
    BackwardFn **next_fns = get_next_functions(backward_fn);

    CallableGradFn grad_fn = get_grad_fn(backward_fn);
    Tensor **op_grads =
        grad_fn(inputs, outputs, grads, num_inputs, num_outputs);

    size_t i = 0;
    while (i < num_outputs) {
        BackwardFn *next_fn = next_fns[i];
        Tensor **next_fn_inputs = get_backward_fn_ip_tensors(next_fn);
        size_t next_fn_num_ips = get_backward_inputs(next_fn);

        Tensor *next_fn_ip_grads[next_fn_num_ips];
        for (size_t j = 0; j < next_fn_num_ips; j++)
            next_fn_ip_grads[j] = op_grads[i++];

        _backward(next_fn_inputs, next_fn_ip_grads, next_fn);
    }

    free(op_grads);
}

void backward(Tensor *tensor, Tensor *grad) {
    bool requires_grad = get_requires_grad(tensor);
    if (!requires_grad) {
        printf("Invalid backward pass on non-requires_grad tensor\n");
        exit(INVALID_BACKWARD_PASS);
    }

    if (!grad) {
        size_t ndim = get_tensor_ndim(tensor);
        size_t *shape = get_tensor_shape(tensor);
        DType dtype = get_tensor_dtype(tensor);

        if (ndim != 0) {
            printf("Invalid backward gradient for non-zero dim tensor\n");
            exit(GRAD_INIT_FAILURE);
        }

        grad = ones_tensor(ndim, shape, dtype, false, NULL);
    }

    BackwardFn *backward_fn = get_backward_fn(tensor);
    _backward((Tensor *[]){tensor}, (Tensor *[]){grad}, backward_fn);

    const Environment *environ = get_tensor_environ(grad);
    if (!environ)
        free_tensor(grad);
}
