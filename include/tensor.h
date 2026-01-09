#ifndef TENSOR_H
#define TENSOR_H

#include "array.h"
#include <stdbool.h>

#define TENSOR_INIT_FAILURE 1
#define INVALID_GRAD 2
#define INVALID_BACKWARD 3

typedef ndArray *(*gradFn)(ndArray *);
typedef struct Dependency Dependency;
typedef struct Tensor Tensor;

Tensor *get_dependency_tensor(const Dependency *dep);
gradFn get_dependency_grad_fn(const Dependency *dep);
void *get_dependency_ctx(const Dependency *dep);

void set_dependency_tensor(Dependency *dep, Tensor *tensor);
void set_dependency_grad_fn(Dependency *dep, gradFn grad_fn);
void set_dependency_ctx(Dependency *dep, void *ctx);

Tensor *tensor_init(ndArray *data, bool requires_grad);
void free_tensor(Tensor *tensor);

ndArray *get_tensor_data(const Tensor *tensor);
ndArray *get_tensor_grad(const Tensor *tensor);

void zero_grad(Tensor *tensor);
void backward(Tensor *tensor, ndArray *grad);

#endif // !TENSOR_H
