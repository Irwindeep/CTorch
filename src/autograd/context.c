#include "autograd.h"
#include "error_codes.h"
#include <stdlib.h>
#include <string.h>

void *deep_copy_ctx(void *ctx, Ctx ctx_kind) {
    switch (ctx_kind) {
    case NULL_CTX:
        RUNTIME_ERROR(INVALID_BACKWARD_PASS,
                      "Invalid Context Kind in Deep Copy");
        return NULL;
    case TRANSPOSE_CTX: {
        TransposeCtx *ctx_copy = malloc(sizeof(TransposeCtx));
        if (!ctx_copy)
            RUNTIME_ERROR(ARRAY_INIT_FAILURE, "Failure to allocate Context");

        ctx_copy->ndim = ((TransposeCtx *)ctx)->ndim;
        ctx_copy->dims = malloc(ctx_copy->ndim * sizeof(int));
        if (!ctx_copy->dims)
            RUNTIME_ERROR(ARRAY_INIT_FAILURE,
                          "Failure to allocate Context dims");
        memcpy(ctx_copy->dims, ((TransposeCtx *)ctx)->dims,
               ctx_copy->ndim * sizeof(int));

        return ctx_copy;
    }
    }

    return NULL;
}

void free_ctx(void *ctx, Ctx ctx_kind) {
    switch (ctx_kind) {
    case NULL_CTX:
        RUNTIME_ERROR(INVALID_BACKWARD_PASS,
                      "Invalid Context Kind in Deep Copy");
        break;
    case TRANSPOSE_CTX: {
        free(((TransposeCtx *)ctx)->dims);
        free(ctx);
    }
    }
}
