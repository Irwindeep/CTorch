#include "error_codes.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Environment is a dynamic array that stores tensors, same as
 * `vector<Tensor *>` in C++.
 */
struct Environment {
    Tensor **tensors;
    size_t capacity;
    size_t num_tensors;

    bool lock;
};

Environment *env_init() {
    Environment *environ = malloc(sizeof(Environment));
    if (!environ)
        RUNTIME_ERROR(ENV_INIT_FAILURE, "Failure to create environment");

    environ->capacity = 1;
    environ->tensors = malloc(environ->capacity * sizeof(Tensor *));
    environ->num_tensors = 0;
    environ->lock = false;

    return environ;
}

void free_env(Environment *environ) {
    if (!environ)
        return;

    for (size_t i = 0; i < environ->num_tensors; i++)
        free_tensor(environ->tensors[i]);
    free(environ->tensors);
    free(environ);
}

void env_push(Environment *environ, Tensor *tensor) {
    if (environ->num_tensors == environ->capacity) {
        size_t new_capacity = 2 * environ->capacity;
        Tensor **new_tensors =
            realloc(environ->tensors, new_capacity * sizeof(Tensor *));
        if (!new_tensors)
            RUNTIME_ERROR(ENV_PUSH_FAILURE, "Memory Re-allocation failure");

        environ->tensors = new_tensors;
        environ->capacity = new_capacity;
    }

    environ->tensors[environ->num_tensors++] = tensor;
}

Tensor *env_pop(Environment *environ) {
    if (environ->num_tensors == 0) {
        printf("No Tensors in the environment, invalid pop\n");
        return NULL;
    }

    Tensor *tensor = environ->tensors[--environ->num_tensors];
    return tensor;
}

Tensor **get_tensors(const Environment *environ) { return environ->tensors; }
size_t get_num_tensors(const Environment *environ) {
    return environ->num_tensors;
}

bool get_lock(const Environment *environ) { return environ->lock; }

void set_lock(Environment *environ) { environ->lock = true; }
void open_lock(Environment *environ) { environ->lock = false; }

Environment *resolve_environ(Tensor *t1, Tensor *t2) {
    Environment *env1 = get_tensor_environ(t1), *env2 = get_tensor_environ(t2);

    if (env1->lock) {
        if (env2->lock)
            RUNTIME_ERROR(
                ENV_RESOLVE_FAILURE,
                "Both Environments Locked - Cannot Resolve Environment");

        return env2;
    }
    return env1;
}
