#ifndef NN_H
#define NN_H

#include "error_codes.h"
#include "tensor.h"
#include <stddef.h>

typedef Tensor *(*CallableModule)(void *module, Tensor *input);
typedef struct Module Module;

struct Module {
    Module **modules;
    size_t num_modules;

    Environment *environ;
    CallableModule forward;

    const char *repr;
    bool repr_dynamic;
};

Tensor *Parameter(int ndim, const size_t *shape, float bound, Environment *env);

void freeze(Module *module);
void unfreeze(Module *module);

void module_init(Module *module);
void add_module(Module *base, Module *child);
void add_tensor(Module *base, Tensor *tensor);

#define AddModule(m, field, expr)                                              \
    do {                                                                       \
        (m)->field = (expr);                                                   \
        add_module(&(m)->base, (m)->field);                                    \
    } while (0)

size_t num_parameters(Module *module);
void parameters(Module *module, Tensor **out);
size_t num_trainable_variables(Module *module);
size_t num_non_trainable_variables(Module *module);

Tensor *module_call(Module *module, Tensor *tensor);

Environment *get_environ(const Module *module);
CallableModule get_callable(const Module *module);

Tensor *_linear(Tensor *input, Tensor *weight, Tensor *bias);
Tensor *_relu(Tensor *input);

typedef struct linear linear;
typedef struct relu relu;
typedef struct sequential sequential;

linear *_Linear(size_t in_features, size_t out_features, bool bias);
relu *_ReLU();
sequential *_Sequential(size_t num_modules, Module **modules);

#define Linear(in_features, out_features)                                      \
    (Module *)_Linear(in_features, out_features, true)
#define LinearBias(in_features, out_features, bias)                            \
    (Module *)_Linear(in_features, out_features, bias)

#define ReLU() (Module *)_ReLU()

#define Sequential(...)                                                        \
    (Module *)_Sequential(                                                     \
        (sizeof((Module *[]){__VA_ARGS__}) / sizeof(Module *)),                \
        ((Module *[]){__VA_ARGS__}));

void free_module(Module *module);

#define ModuleInit(ptr, type, name)                                            \
    do {                                                                       \
        (ptr) = calloc(1, sizeof(type));                                       \
        if (!(ptr)) {                                                          \
            RUNTIME_ERRORF(MODULE_ALLOC_FAILURE,                               \
                           "Failure to allocate module `%s`", #type);          \
        }                                                                      \
        module_init(&(ptr)->base);                                             \
                                                                               \
        (ptr)->base.repr = name;                                               \
    } while (0)

#define ModuleClose(ptr)                                                       \
    do {                                                                       \
        set_lock(get_environ(&(ptr)->base));                                   \
    } while (0)

#endif // !NN_H
