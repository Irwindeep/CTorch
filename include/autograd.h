#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "array.h"
#include "tensor.h"

#define CTX_INIT_FAILURE 1

typedef struct SingleTensorCtx SingleTensorCtx;
typedef struct TwoTensorCtx TwoTensorCtx;

SingleTensorCtx *init_single_tensor_ctx(Tensor *tensor);

Tensor *get_tensor_stc(const SingleTensorCtx *ctx);
void set_tensor_stc(SingleTensorCtx *ctx);

TwoTensorCtx *init_two_tensor_ctx(Tensor *t1, Tensor *t2);

Tensor *get_tensor1_ttc(const TwoTensorCtx *ctx);
Tensor *get_tensor2_ttc(const TwoTensorCtx *ctx);

void free_context(void *ctx, CtxType ctx_type);

void set_tensor1_ttc(TwoTensorCtx *ctx, Tensor *tensor);
void set_tensor2_ttc(TwoTensorCtx *ctx, Tensor *tensor);

ndArray *_add_grad_fn1(ndArray *grad, void *ctx);
ndArray *_add_grad_fn2(ndArray *grad, void *ctx);

ndArray *_mul_grad_fn1(ndArray *grad, void *ctx);
ndArray *_mul_grad_fn2(ndArray *grad, void *ctx);

ndArray *_neg_grad_fn(ndArray *grad, void *ctx);
ndArray *_inv_grad_fn(ndArray *grad, void *ctx);

#endif // !AUTOGRAD_H
