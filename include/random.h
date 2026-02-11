#ifndef RANDOM_H
#define RANDOM_H

#include "array.h"
#include "tensor.h"

#include <stddef.h>
#include <stdint.h>

typedef struct PRNG PRNG;

extern PRNG *global_rng;

PRNG *rng_init(uint64_t seed);
uint64_t rng_rand(PRNG *rng);
void free_rng(PRNG *rng);

Tensor *uniform(int ndim, const size_t *shape, float bound, DType dtype,
                bool requires_grad, Environment *env);
Tensor *randn(int ndim, const size_t *shape, DType dtype, bool requires_grad,
              Environment *env);
Tensor *randint(int ndim, const size_t *shape, long int low, long int high,
                DType dtype, Environment *env);

#endif // !RANDOM_H
