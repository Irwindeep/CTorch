#ifndef TENSOR_H
#define TENSOR_H

#include "array.h"
#include <stdbool.h>
#include <stddef.h>

#define REQUIRES_GRAD true
#define NO_GRAD false

#define CREATE_GRAPH true
#define NO_GRAPH false

typedef struct Environment Environment;
typedef struct Tensor Tensor;
typedef struct BackwardFn BackwardFn;

Environment *env_init();
void free_env(Environment *environ);

void env_push(Environment *environ, Tensor *tensor);
Tensor *env_pop(Environment *environ);

Tensor *tensor_init(ndArray *data, bool requires_grad, Environment *environ);
void free_tensor(Tensor *tensor);

void save_tensor(Tensor *tensor, const char *path);
Tensor *load_tensor(const char *path, bool requires_grad, Environment *environ);

ndArray *get_tensor_data(const Tensor *tensor);
Tensor *get_tensor_grad(const Tensor *tensor);

bool get_requires_grad(const Tensor *tensor);
int get_tensor_ndim(const Tensor *tensor);
size_t *get_tensor_shape(const Tensor *tensor);
DType get_tensor_dtype(const Tensor *tensor);
Environment *get_tensor_environ(const Tensor *tensor);

BackwardFn *get_backward_fn(const Tensor *tensor);

void set_requires_grad(Tensor *tensor, bool requires_grad);

void set_tensor_grad(Tensor *tensor, Tensor *grad);
void set_backward_fn(Tensor *tensor, BackwardFn *backward_fn);

void zero_grad(Tensor *tensor);

Tensor *eye_tensor(size_t m, size_t n, DType dtype, bool requires_grad,
                   Environment *environ);
Tensor *zeros_tensor(int ndim, const size_t *shape, DType dtype,
                     bool requires_grad, Environment *environ);
Tensor *ones_tensor(int ndim, const size_t *shape, DType dtype,
                    bool requires_grad, Environment *environ);
Tensor *zeros_like(const Tensor *tensor, bool requires_grad, Environment *env);
Tensor *ones_like(const Tensor *tensor, bool requires_grad, Environment *env);
Tensor *scalar(ArrayVal value, DType dtype, bool requires_grad,
               Environment *environ);

Tensor *tensor_add(Tensor *t1, Tensor *t2);
Tensor *tensor_sub(Tensor *t1, Tensor *t2);
Tensor *tensor_mul(Tensor *t1, Tensor *t2);
Tensor *tensor_div(Tensor *t1, Tensor *t2);
Tensor *tensor_neg(Tensor *tensor);
Tensor *tensor_inv(Tensor *tensor);

Tensor *tensor_transpose(Tensor *tensor, int *dims);
Tensor *tensor_matmul(Tensor *t1, Tensor *t2);

#define SHAPE(...)                                                             \
    (sizeof((size_t[]){__VA_ARGS__}) / sizeof(size_t)),                        \
        ((const size_t[]){__VA_ARGS__})

#define SHAPE_(...) ((const size_t[]){__VA_ARGS__})

#define TENSORS(...)                                                           \
    (sizeof((Tensor *[]){__VA_ARGS__}) / sizeof(Tensor *)),                    \
        ((Tensor *[]){__VA_ARGS__})

#define TENSORS_(...) ((Tensor *[]){__VA_ARGS__})

#define SCALAR_VAL(x)                                                          \
    _Generic((x),                                                              \
        int: (ArrayVal){.int_val = (x)},                                       \
        float: (ArrayVal){.float_val = (x)},                                   \
        double: (ArrayVal){.double_val = (x)},                                 \
        long: (ArrayVal){.long_val = (x)})

#define SCALAR_DTYPE(x)                                                        \
    _Generic((x),                                                              \
        int: DTYPE_INT,                                                        \
        float: DTYPE_FLOAT,                                                    \
        double: DTYPE_DOUBLE,                                                  \
        long: DTYPE_LONG)

#define SCALAR_G(x, env)                                                       \
    scalar(SCALAR_VAL(x), SCALAR_DTYPE(x), REQUIRES_GRAD, (env))

#define SCALAR_NG(x, env) scalar(SCALAR_VAL(x), SCALAR_DTYPE(x), NO_GRAD, (env))

#endif // !TENSOR_H
