#ifndef RANDOM_H
#define RANDOM_H

#include "array.h"
#include "tensor.h"

#include <stddef.h>
#include <stdint.h>

typedef struct PRNG PRNG;

PRNG *rng_init(uint64_t seed);
uint64_t rng_rand(PRNG *rng);
void free_rng(PRNG *rng);

Tensor *uniform(PRNG *rng, int ndim, const size_t *shape, DType dtype,
                bool requires_grad, Environment *environ);
Tensor *randn(PRNG *rng, int ndim, const size_t *shape, DType dtype,
              bool requires_grad, Environment *environ);
Tensor *randint(PRNG *rng, int ndim, const size_t *shape, long int low,
                long int high, DType dtype, Environment *environ);

#endif // !RANDOM_H
