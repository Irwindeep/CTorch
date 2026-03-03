#include "optim.h"
#include "tensor.h"

#include <stddef.h>
#include <stdlib.h>

void optim_zero_grad(Optimizer *optim) {
    for (size_t i = 0; i < optim->num_params; i++) {
        if (get_requires_grad(optim->params[i]))
            zero_grad(optim->params[i]);
    }
}

void optim_step(Optimizer *optim) { optim->step(optim); }

void free_optim(Optimizer **optim) {
    if (!optim || !*optim)
        return;

    free((*optim)->params);
    free_tensor((*optim)->lr);
    free(*optim);

    *optim = NULL;
}
