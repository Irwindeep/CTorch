#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"
#include <stddef.h>

#define BACKWARD_FN_INIT_FAILURE 1
#define NEXT_FNS_INIT_FAILURE 2
#define GRAD_INIT_FAILURE 3

typedef Tensor **(*CallableGradFn)(Tensor **inputs, Tensor **outputs,
                                   Tensor **input_grads, size_t num_inputs,
                                   size_t num_outputs);

BackwardFn *backward_fn_init(CallableGradFn grad_fn, Tensor **tensors,
                             size_t num_inputs, size_t num_outputs,
                             const char *name);
void free_backward_fn(BackwardFn *backward_fn);
BackwardFn **create_next_fns(Tensor **output_tensors, size_t num_outputs);

BackwardFn **get_next_functions(const BackwardFn *backward_fn);
char *get_backward_name(const BackwardFn *backward_fn);
size_t get_backward_inputs(const BackwardFn *backward_fn);
size_t get_backward_outputs(const BackwardFn *backward_fn);
CallableGradFn get_grad_fn(const BackwardFn *backward_fn);

void set_next_functions(BackwardFn *backward_fn, BackwardFn **next_functions);

BackwardFn *AccumulateGrad(Tensor *input);
BackwardFn *AddBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);
BackwardFn *MulBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);

#endif // !AUTOGRAD_H
