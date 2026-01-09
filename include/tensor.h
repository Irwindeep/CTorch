#ifndef TENSOR_H
#define TENSOR_H

#include "array.h"
#include <stdbool.h>
#include <stddef.h>

#define TENSOR_INIT_FAILURE 1
#define INVALID_GRAD 2
#define INVALID_BACKWARD 3
#define GRAD_CTX_FAILURE 4

#define DEPENDENCY_INIT_FAILURE 1

typedef ndArray *(*gradFn)(ndArray *, void *ctx);
typedef struct Dependency Dependency;
typedef struct Tensor Tensor;

Tensor *get_dependency_tensor(const Dependency *dep);
gradFn get_dependency_grad_fn(const Dependency *dep);
void *get_dependency_ctx(const Dependency *dep);

void set_dependency_tensor(Dependency *dep, Tensor *tensor);
void set_dependency_grad_fn(Dependency *dep, gradFn grad_fn);
void set_dependency_ctx(Dependency *dep, void *ctx);

Dependency **dependency_arr_init(size_t dependency_cnt);
Dependency *create_dependency(Tensor *tensor, gradFn grad_fn, void *ctx);
void free_dependency(Dependency *dependency);

Tensor *tensor_init(ndArray *data, bool requires_grad);
void free_tensor(Tensor *tensor);

ndArray *get_tensor_data(const Tensor *tensor);
ndArray *get_tensor_grad(const Tensor *tensor);
bool get_requires_grad(const Tensor *tensor);
size_t get_dependency_cnt(const Tensor *tensor);
int get_tensor_ndim(const Tensor *tensor);
size_t *get_tensor_shape(const Tensor *tensor);

void set_requires_grad(Tensor *tensor, bool requires_grad);
void set_dependency_arr(Tensor *tensor, Dependency **dependency_arr,
                        size_t dependency_cnt);

void zero_grad(Tensor *tensor);
void backward(Tensor *tensor, ndArray *grad);

Tensor *tensor_add(Tensor *t1, Tensor *t2);

#endif // !TENSOR_H
