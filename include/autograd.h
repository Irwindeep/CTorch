#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"
#include <stddef.h>

typedef enum Ctx {
    NULL_CTX,
    TRANSPOSE_CTX,
} Ctx;

typedef struct TransposeCtx {
    int ndim;
    int *dims;
} TransposeCtx;

void *deep_copy_ctx(void *ctx, Ctx ctx_kind);
void free_ctx(void *ctx, Ctx ctx_kind);

typedef Tensor **(*CallableGradFn)(Tensor **inputs, Tensor **outputs,
                                   Tensor **input_grads, size_t num_inputs,
                                   size_t num_outputs, bool create_graph);

BackwardFn *backward_fn_init(CallableGradFn grad_fn, Tensor **input_tensors,
                             Tensor **output_tensors, size_t num_inputs,
                             size_t num_outputs, const char *name);
void free_backward_fn(BackwardFn *backward_fn);
BackwardFn **create_next_fns(Tensor **output_tensors, size_t num_outputs);

BackwardFn **get_next_functions(const BackwardFn *backward_fn);
char *get_backward_name(const BackwardFn *backward_fn);
size_t get_backward_inputs(const BackwardFn *backward_fn);
size_t get_backward_outputs(const BackwardFn *backward_fn);
Tensor **get_backward_fn_ip_tensors(const BackwardFn *backward_fn);
Tensor **get_backward_fn_op_tensors(const BackwardFn *backward_fn);
CallableGradFn get_grad_fn(const BackwardFn *backward_fn);

void *get_ctx(const BackwardFn *backward_fn);
Ctx get_ctx_kind(const BackwardFn *backward_fn);

void set_ctx(BackwardFn *backward_fn, void *ctx, Ctx ctx_kind);
void set_next_functions(BackwardFn *backward_fn, BackwardFn **next_functions);

Tensor **gradient(size_t num_inputs, Tensor **inputs, size_t num_outputs,
                  Tensor **outputs, Tensor **grad_outputs, bool create_graph);
void backward(Tensor *tensor, Tensor *grad);

BackwardFn *AccumulateGrad(Tensor *input);
BackwardFn *AddBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);
BackwardFn *MulBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);
BackwardFn *NegBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);
BackwardFn *InvBackward(Tensor **input_tensors, Tensor **output_tensors,
                        size_t num_inputs, size_t num_outputs);
BackwardFn *MatMulBackward(Tensor **input_tensors, Tensor **output_tensors,
                           size_t num_inputs, size_t num_outputs);
BackwardFn *TransposeBackward(Tensor **input_tensors, Tensor **output_tensors,
                              size_t num_inputs, size_t num_outputs);

#endif // !AUTOGRAD_H
