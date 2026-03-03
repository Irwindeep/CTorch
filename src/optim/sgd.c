#include "array.h"
#include "error_codes.h"
#include "optim.h"
#include "tensor.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct _SGD {
    Optimizer optim;
} _SGD;

void sgd_step(void *_optim) {
    _SGD *optim = (_SGD *)_optim;

    for (size_t i = 0; i < optim->optim.num_params; i++) {
        if (get_requires_grad(optim->optim.params[i])) {
            Tensor *param_grad = get_tensor_grad(optim->optim.params[i]);
            ndArray *grad = array_mul(get_tensor_data(param_grad),
                                      get_tensor_data(optim->optim.lr));

            ndArray *param = get_tensor_data(optim->optim.params[i]);
            ndArray *new_param = array_sub(param, grad);
            replace_tensor_data(optim->optim.params[i], new_param);
            free_array(grad);
        }
    }
}

Optimizer *SGD(size_t num_params, Tensor **params, float lr) {
    _SGD *optim = malloc(sizeof(_SGD));
    if (!optim)
        RUNTIME_ERROR(ARRAY_INIT_FAILURE, "Failed to initialize optimizer");

    optim->optim.num_params = num_params;
    optim->optim.params = malloc(num_params * sizeof(Tensor *));
    memcpy(optim->optim.params, params, num_params * sizeof(Tensor *));

    optim->optim.lr = SCALAR_NG(lr, NULL);
    optim->optim.step = sgd_step;

    return &optim->optim;
}
