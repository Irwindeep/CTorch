#ifndef CALLABLE_GRADS_H
#define CALLABLE_GRADS_H

#include "array.h"
#include "tensor.h"

#include <stddef.h>

ndArray *broadcast_grad_data(ndArray *data, int ndim, const size_t *shape);
Tensor *broadcast_tensor_grad(Tensor *data, int ndim, const size_t *shape);

#define _DECLARE_GRAD_FN(NAME)                                                 \
    Tensor **NAME(Tensor **inputs, Tensor **outputs, Tensor **input_grads,     \
                  size_t num_inputs, size_t num_outputs, bool create_graph);

_DECLARE_GRAD_FN(_accumulate_grad_fn)
_DECLARE_GRAD_FN(_add_grad_fn)
_DECLARE_GRAD_FN(_mul_grad_fn)
_DECLARE_GRAD_FN(_neg_grad_fn)
_DECLARE_GRAD_FN(_inv_grad_fn)
_DECLARE_GRAD_FN(_transpose_grad_fn)
_DECLARE_GRAD_FN(_matmul_grad_fn)
_DECLARE_GRAD_FN(_sum_grad_fn)

#endif // !CALLABLE_GRADS_H
