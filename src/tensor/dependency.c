#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

struct Dependency {
    Tensor *tensor;
    gradFn grad_fn;
    void *ctx;
};

Tensor *get_dependency_tensor(const Dependency *dep) { return dep->tensor; }
gradFn get_dependency_grad_fn(const Dependency *dep) { return dep->grad_fn; }
void *get_dependency_ctx(const Dependency *dep) { return dep->ctx; }

void set_dependency_tensor(Dependency *dep, Tensor *tensor) {
    dep->tensor = tensor;
}
void set_dependency_grad_fn(Dependency *dep, gradFn grad_fn) {
    dep->grad_fn = grad_fn;
}
void set_dependency_ctx(Dependency *dep, void *ctx) { dep->ctx = ctx; }

Dependency **dependency_arr_init(size_t dependency_cnt) {
    Dependency **dependency_arr = malloc(dependency_cnt * sizeof(Dependency *));
    if (!dependency_arr) {
        printf("Failure to create dependency array\n");
        exit(DEPENDENCY_INIT_FAILURE);
    }

    return dependency_arr;
}

Dependency *create_dependency(Tensor *tensor, gradFn grad_fn, void *ctx) {
    Dependency *dependency = malloc(sizeof(Dependency));
    if (!dependency) {
        printf("Failure to create dependency\n");
        exit(DEPENDENCY_INIT_FAILURE);
    }

    dependency->tensor = tensor;
    dependency->grad_fn = grad_fn;
    dependency->ctx = ctx;

    return dependency;
}

void free_dependency(Dependency *dependency) {
    free(dependency->ctx);
    free(dependency);
}
