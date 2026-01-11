#include "autograd.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

struct SingleTensorCtx {
    Tensor *tensor;
};
struct TwoTensorCtx {
    Tensor *t1;
    Tensor *t2;
};

SingleTensorCtx *init_single_tensor_ctx(Tensor *tensor) {
    SingleTensorCtx *ctx = malloc(sizeof(SingleTensorCtx));
    if (!ctx) {
        printf("Failure to create context\n");
        exit(CTX_INIT_FAILURE);
    }

    ctx->tensor = tensor;
    return ctx;
}
Tensor *get_tensor_stc(const SingleTensorCtx *ctx) { return ctx->tensor; }

void free_stc(SingleTensorCtx *ctx) {
    free_tensor(ctx->tensor);
    free(ctx);
}

TwoTensorCtx *init_two_tensor_ctx(Tensor *t1, Tensor *t2) {
    TwoTensorCtx *ctx = malloc(sizeof(TwoTensorCtx));
    if (!ctx) {
        printf("Failure to create context\n");
        exit(CTX_INIT_FAILURE);
    }

    ctx->t1 = t1;
    ctx->t2 = t2;

    return ctx;
}

Tensor *get_tensor1_ttc(const TwoTensorCtx *ctx) { return ctx->t1; }
Tensor *get_tensor2_ttc(const TwoTensorCtx *ctx) { return ctx->t2; }

void set_tensor1_ttc(TwoTensorCtx *ctx, Tensor *tensor) { ctx->t1 = tensor; }
void set_tensor2_ttc(TwoTensorCtx *ctx, Tensor *tensor) { ctx->t2 = tensor; }

void free_ttc(TwoTensorCtx *ctx) {
    free_tensor(ctx->t1);
    free_tensor(ctx->t2);

    free(ctx);
}

void free_context(void *ctx, CtxType ctx_type) {
    switch (ctx_type) {
    case DEFAULT_CTX_TYPE:
        free(ctx);
        break;
    case SINGLE_TENSOR_CTX:
        free_stc(ctx);
        break;
    case TWO_TENSOR_CTX:
        free_ttc(ctx);
        break;
    }
}
