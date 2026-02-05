#include "autograd.h"
#include "private/callable_grads.h"
#include "tensor.h"

#include <stddef.h>

BackwardFn *AccumulateGrad(Tensor *tensor) {
    CallableGradFn grad_fn = _accumulate_grad_fn;
    Tensor *tensors[] = {tensor};
    BackwardFn *backward_fn =
        backward_fn_init(grad_fn, tensors, NULL, 1, 0, "AccumulateGrad");

    return backward_fn;
}

#define DEFINE_BACKWARD_FN(NAME, _grad_fn)                                     \
    BackwardFn *NAME(Tensor **input_tensors, Tensor **output_tensors,          \
                     size_t num_inputs, size_t num_outputs) {                  \
        CallableGradFn grad_fn = _grad_fn;                                     \
        BackwardFn *backward_fn =                                              \
            backward_fn_init(grad_fn, input_tensors, output_tensors,           \
                             num_inputs, num_outputs, #NAME);                  \
                                                                               \
        BackwardFn **next_functions =                                          \
            create_next_fns(output_tensors, num_outputs);                      \
        set_next_functions(backward_fn, next_functions);                       \
        return backward_fn;                                                    \
    }

DEFINE_BACKWARD_FN(AddBackward, _add_grad_fn)
DEFINE_BACKWARD_FN(MulBackward, _mul_grad_fn)
DEFINE_BACKWARD_FN(NegBackward, _neg_grad_fn)
DEFINE_BACKWARD_FN(InvBackward, _inv_grad_fn)

DEFINE_BACKWARD_FN(MaxBackward, _max_grad_fn)
DEFINE_BACKWARD_FN(MinBackward, _min_grad_fn)
