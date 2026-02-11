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
    Environment *env = malloc(sizeof(Environment));
    if (!env)
        RUNTIME_ERROR(ENV_INIT_FAILURE, "Failure to create environment");

    env->capacity = 1;
    env->tensors = malloc(env->capacity * sizeof(Tensor *));
    env->num_tensors = 0;
    env->lock = false;

    return env;
}

void free_env(Environment *env) {
    if (!env)
        return;

    for (size_t i = 0; i < env->num_tensors; i++)
        free_tensor(env->tensors[i]);
    free(env->tensors);
    free(env);
}

void env_push(Environment *env, Tensor *tensor) {
    if (env->lock)
        RUNTIME_ERROR(INVALID_ARRAY, "Invalid access to locked environment");

    if (env->num_tensors == env->capacity) {
        size_t new_capacity = 2 * env->capacity;
        Tensor **new_tensors =
            realloc(env->tensors, new_capacity * sizeof(Tensor *));
        if (!new_tensors)
            RUNTIME_ERROR(ENV_PUSH_FAILURE, "Memory Re-allocation failure");

        env->tensors = new_tensors;
        env->capacity = new_capacity;
    }

    env->tensors[env->num_tensors++] = tensor;
}

Tensor *env_pop(Environment *env) {
    if (env->num_tensors == 0) {
        printf("No Tensors in the environment, invalid pop\n");
        return NULL;
    }

    Tensor *tensor = env->tensors[--env->num_tensors];
    return tensor;
}

bool env_remove_and_free(Environment *env, const Tensor *target) {
    if (!env || !target)
        return false;

    for (size_t i = 0; i < env->num_tensors; i++) {
        if (env->tensors[i] == target) {
            free_tensor(env->tensors[i]);

            for (size_t j = i + 1; j < env->num_tensors; j++) {
                env->tensors[j - 1] = env->tensors[j];
            }

            env->num_tensors--;
            return true;
        }
    }

    return false;
}

Tensor **get_tensors(const Environment *env) { return env->tensors; }
size_t get_num_tensors(const Environment *env) { return env->num_tensors; }

bool get_lock(const Environment *env) { return env->lock; }

void set_lock(Environment *env) { env->lock = true; }
void open_lock(Environment *env) { env->lock = false; }

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
