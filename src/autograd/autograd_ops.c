#include "autograd.h"
#include "private/callable_grads.h"
#include "tensor.h"

BackwardFn *TransposeBackward(Tensor **input_tensors, Tensor **output_tensors,
                              size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _transpose_grad_fn;
    BackwardFn *backward_fn =
        backward_fn_init(grad_fn, input_tensors, output_tensors, num_inputs,
                         num_outputs, "TransposeBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}

BackwardFn *MatMulBackward(Tensor **input_tensors, Tensor **output_tensors,
                           size_t num_inputs, size_t num_outputs) {
    CallableGradFn grad_fn = _matmul_grad_fn;
    BackwardFn *backward_fn =
        backward_fn_init(grad_fn, input_tensors, output_tensors, num_inputs,
                         num_outputs, "MatMulBackward");

    BackwardFn **next_functions = create_next_fns(output_tensors, num_outputs);
    set_next_functions(backward_fn, next_functions);

    return backward_fn;
}
