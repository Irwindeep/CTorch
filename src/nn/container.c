#include "error_codes.h"
#include "nn.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline char *update_cap(char *tmp, size_t needed, size_t capacity) {
    if (needed > capacity) {
        capacity *= 2;
        char *new_tmp = realloc(tmp, capacity);
        if (!new_tmp)
            RUNTIME_ERROR(
                MODULE_ALLOC_FAILURE,
                "Failed to allocate string representation for Sequential");
        return new_tmp;
    }
    return tmp;
}

struct sequential {
    Module base;
};

Tensor *_sequential_forward(void *module, Tensor *input) {
    Module *m = (Module *)module;
    Tensor *output = input;
    for (size_t i = 0; i < m->num_modules; i++)
        output = module_call(m->modules[i], output);

    return output;
}

sequential *_Sequential(size_t num_modules, Module **modules) {
    sequential *layer = calloc(1, sizeof(sequential));
    if (!layer)
        RUNTIME_ERROR(MODULE_ALLOC_FAILURE, "Failed to allocate Linear layer");

    module_init(&layer->base);
    layer->base.forward = _sequential_forward;

    size_t capacity = 1024;
    char *tmp = malloc(capacity);
    if (!tmp)
        RUNTIME_ERROR(
            MODULE_ALLOC_FAILURE,
            "Failed to allocate string representation for Sequential");

    strcpy(tmp, "Sequential(\n");
    for (size_t i = 0; i < num_modules; i++) {
        size_t needed = strlen(tmp) + strlen(modules[i]->repr) + 3;
        tmp = update_cap(tmp, needed, capacity);

        strcat(tmp, "    ");
        strcat(tmp, modules[i]->repr);
        if (i < num_modules - 1)
            strcat(tmp, ",");
        strcat(tmp, "\n");

        add_module(&layer->base, modules[i]);
    }
    strcat(tmp, ")");
    layer->base.repr = tmp;
    layer->base.repr_dynamic = true;

    return layer;
}
