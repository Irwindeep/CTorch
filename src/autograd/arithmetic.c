#include "autograd.h"
#include "private/callable_grads.h"
#include "tensor.h"

#include <stddef.h>

BackwardFn *AccumulateGrad(Tensor *tensor) {
    CallableGradFn grad_fn = _accumulate_grad_fn;
    Tensor *tensors[] = {tensor};
    BackwardFn *backward_fn =
        backward_fn_init(grad_fn, tensors, 1, 0, "AccumulateGrad");

    return backward_fn;
}

BackwardFn *AddBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _add_grad_fn;
    BackwardFn *backward_fn = backward_fn_init(
        grad_fn, input_tensors, num_inputs, num_outputs, "AddBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}

BackwardFn *MulBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _mul_grad_fn;
    BackwardFn *backward_fn = backward_fn_init(
        grad_fn, input_tensors, num_inputs, num_outputs, "MulBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}

BackwardFn *NegBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _neg_grad_fn;
    BackwardFn *backward_fn = backward_fn_init(
        grad_fn, input_tensors, num_inputs, num_outputs, "NegBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}

BackwardFn *InvBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _inv_grad_fn;
    BackwardFn *backward_fn = backward_fn_init(
        grad_fn, input_tensors, num_inputs, num_outputs, "InvBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}
