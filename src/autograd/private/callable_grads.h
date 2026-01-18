#ifndef CALLABLE_GRADS_H
#define CALLABLE_GRADS_H

#include "array.h"
#include "tensor.h"

#include <stddef.h>

#define INVALID_NUM_INPUTS_OUTPUTS 1

ndArray *broadcast_grad_data(ndArray *data, int ndim, const size_t *shape);

Tensor **_accumulate_grad_fn(Tensor **inputs, Tensor **outputs,
                             Tensor **input_grads, size_t num_inputs,
                             size_t num_outputs);
Tensor **_add_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs);
Tensor **_mul_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs);
Tensor **_neg_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs);
Tensor **_inv_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs);

#endif // !CALLABLE_GRADS_H
