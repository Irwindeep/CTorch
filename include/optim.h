#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"
#include <stddef.h>

typedef void (*StepFn)(void *optim);

typedef struct Optimizer {
    Tensor **params;
    Tensor *lr;
    size_t num_params;

    StepFn step;
} Optimizer;

void optim_zero_grad(Optimizer *optim);
void optim_step(Optimizer *optim);

Optimizer *SGD(size_t num_params, Tensor **params, float lr);

void free_optim(Optimizer **optim);

#endif // !OPTIM_H
