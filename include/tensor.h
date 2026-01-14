#ifndef TENSOR_H
#define TENSOR_H

#include "array.h"
#include <stdbool.h>
#include <stddef.h>

#define TENSOR_INIT_FAILURE 1
#define INVALID_GRAD 2
#define DEPENDENCY_ARR_INIT_FAILURE 3

typedef struct Tensor Tensor;
typedef struct BackwardFn BackwardFn;

Tensor *tensor_init(ndArray *data, bool requires_grad);
void free_tensor(Tensor *tensor);

ndArray *get_tensor_data(const Tensor *tensor);
Tensor *get_tensor_grad(const Tensor *tensor);

bool get_requires_grad(const Tensor *tensor);
int get_tensor_ndim(const Tensor *tensor);
size_t *get_tensor_shape(const Tensor *tensor);

BackwardFn *get_backward_fn(const Tensor *tensor);

void set_requires_grad(Tensor *tensor, bool requires_grad);

void set_tensor_grad(Tensor *tensor, Tensor *grad);
void set_backward_fn(Tensor *tensor, BackwardFn *backward_fn);

void zero_grad(Tensor *tensor);
void backward(Tensor *tensor, ndArray *grad);

Tensor *eye_tensor(size_t m, size_t n, DType dtype, bool requires_grad);
Tensor *zeros_tensor(int ndim, const size_t *shape, DType dtype,
                     bool requires_grad);
Tensor *ones_tensor(int ndim, const size_t *shape, DType dtype,
                    bool requires_grad);

Tensor *tensor_add(Tensor *t1, Tensor *t2);
Tensor *tensor_sub(Tensor *t1, Tensor *t2);
Tensor *tensor_mul(Tensor *t1, Tensor *t2);
Tensor *tensor_div(Tensor *t1, Tensor *t2);
Tensor *tensor_neg(Tensor *tensor);
Tensor *tensor_inv(Tensor *tensor);

#endif // !TENSOR_H
