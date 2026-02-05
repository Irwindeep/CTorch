#include "array.h"
#include "error_codes.h"
#include "nn.h"
#include "tensor.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void module_init(Module *module) {
    module->modules = NULL;
    module->num_modules = 0;
    module->environ = env_init();
    module->forward = NULL;

    module->repr = NULL;
}

void add_module(Module *base, Module *child) {
    Module **new_modules =
        realloc(base->modules, (base->num_modules + 1) * sizeof(Module *));
    if (!new_modules)
        RUNTIME_ERROR(MODULE_ALLOC_FAILURE, "Failed to allocate module");

    base->modules = new_modules;
    base->modules[base->num_modules++] = child;
}

size_t num_parameters(Module *module) {
    size_t count = 0;

    Environment *env = get_environ(module);
    if (env)
        count += get_num_tensors(env);

    for (size_t i = 0; i < module->num_modules; i++)
        count += num_parameters(module->modules[i]);

    return count;
}

static void _parameters(Module *module, Tensor **out, size_t *count) {
    Environment *env = module->environ;
    if (env) {
        size_t num_tensors = get_num_tensors(env);
        Tensor **env_tensors = get_tensors(module->environ);

        for (size_t i = 0; i < num_tensors; i++)
            out[(*count)++] = env_tensors[i];
    }

    for (size_t i = 0; i < module->num_modules; i++) {
        _parameters(module->modules[i], out, count);
    }
}

void parameters(Module *module, Tensor **out) {
    size_t idx = 0;
    _parameters(module, out, &idx);
}

size_t num_trainable_variables(Module *module) {
    size_t num_params = num_parameters(module);
    Tensor *params[num_params];
    parameters(module, params);

    size_t num_trainable_vars = 0;
    for (size_t i = 0; i < num_params; i++) {
        Tensor *param = params[i];
        if (!get_requires_grad(param))
            continue;

        num_trainable_vars += get_total_size(get_tensor_data(param));
    }

    return num_trainable_vars;
}

size_t num_non_trainable_variables(Module *module) {
    size_t num_params = num_parameters(module);
    Tensor *params[num_params];
    parameters(module, params);

    size_t num_non_trainable_vars = 0;
    for (size_t i = 0; i < num_params; i++) {
        Tensor *param = params[i];
        if (get_requires_grad(param))
            continue;

        num_non_trainable_vars += get_total_size(get_tensor_data(param));
    }

    return num_non_trainable_vars;
}

Environment *get_environ(const Module *module) { return module->environ; }
CallableModule get_callable(const Module *module) { return module->forward; }

void free_module(Module *module) {
    for (size_t i = 0; i < module->num_modules; i++)
        free_module(module->modules[i]);

    if (module->num_modules > 0)
        free(module->modules);

    free_env(module->environ);
    if (module->repr && module->repr_dynamic)
        free((void *)module->repr);
    free(module);
}

Tensor *module_call(Module *module, Tensor *input) {
    return module->forward(module, input);
}
