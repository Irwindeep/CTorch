#ifndef CTORCH_H
#define CTORCH_H

#include <stdint.h>

typedef enum DType {
    DTYPE_INT,
    DTYPE_FLOAT,
    DTYPE_DOUBLE,
    DTYPE_LONG,
} DType;

extern const char *DTypeNames[];

void CTorchInit(void);
void ManualSeed(uint64_t seed);
void CTorchClose(void);

void ct_exit(int status);

#if defined(__GNUC__) || defined(__clang__)
#define ScopedEnvironment __attribute__((cleanup(free_env))) Environment *
#define ScopedOptimizer   __attribute__((cleanup(free_optim))) Optimizer *
#define ScopedModule      __attribute__((cleanup(free_module))) Module *
#else
#define ScopedEnvironment Environment *
#define ScopedOptimizer   Optimizer *
#define ScoScopedModule   Module *
#endif

#define REQUIRES_GRAD true
#define NO_GRAD       false

#define CREATE_GRAPH true
#define NO_GRAPH     false

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

#endif // !CTORCH_H
