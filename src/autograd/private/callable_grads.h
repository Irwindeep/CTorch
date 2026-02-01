#ifndef CALLABLE_GRADS_H
#define CALLABLE_GRADS_H

#include "array.h"
#include "tensor.h"

#include <stddef.h>

ndArray *broadcast_grad_data(ndArray *data, int ndim, const size_t *shape);
Tensor *broadcast_tensor_grad(Tensor *data, int ndim, const size_t *shape);

Tensor **_accumulate_grad_fn(Tensor **inputs, Tensor **outputs,
                             Tensor **input_grads, size_t num_inputs,
                             size_t num_outputs, bool create_graph);
Tensor **_add_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs, bool create_graph);
Tensor **_mul_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs, bool create_graph);
Tensor **_neg_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs, bool create_graph);
Tensor **_inv_grad_fn(Tensor **inputs, Tensor **outputs, Tensor **input_grads,
                      size_t num_inputs, size_t num_outputs, bool create_graph);

Tensor **_transpose_grad_fn(Tensor **inputs, Tensor **outputs,
                            Tensor **input_grads, size_t num_inputs,
                            size_t num_outputs, bool create_graph);
Tensor **_matmul_grad_fn(Tensor **inputs, Tensor **outputs,
                         Tensor **input_grads, size_t num_inputs,
                         size_t num_outputs, bool create_graph);

#endif // !CALLABLE_GRADS_H
