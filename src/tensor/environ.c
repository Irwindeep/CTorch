#include "tensor.h"
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
};

Environment *env_init() {
    Environment *environ = malloc(sizeof(Environment));
    if (!environ) {
        printf("Failure to create environment\n");
        exit(ENV_INIT_FAILURE);
    }

    environ->capacity = 1;
    environ->tensors = malloc(environ->capacity * sizeof(Tensor *));
    environ->num_tensors = 0;

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
        if (!new_tensors) {
            printf("Memory Re-allocation failure in `env_push`\n");
            exit(ENV_PUSH_FAILURE);
        }

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
