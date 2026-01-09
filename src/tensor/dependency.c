#include "tensor.h"

struct Dependency {
    Tensor *tensor;
    gradFn grad_fn;
    void *ctx;
};

Tensor *get_tensor(const Dependency *dep) { return dep->tensor; }
gradFn get_grad_fn(const Dependency *dep) { return dep->grad_fn; }
void *get_ctx(const Dependency *dep) { return dep->ctx; }

void set_tensor(Dependency *dep, Tensor *tensor) { dep->tensor = tensor; }
void set_grad_fn(Dependency *dep, gradFn grad_fn) { dep->grad_fn = grad_fn; }
void set_ctx(Dependency *dep, void *ctx) { dep->ctx = ctx; }
