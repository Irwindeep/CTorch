#include "autograd.h"
#include "tensor.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct BackwardFn {
    CallableGradFn grad_fn;
    size_t num_inputs;  // num inputs for the grad_fn
    size_t num_outputs; // num outputs for the grad_fn
    BackwardFn **next_functions;
    Tensor **tensors; // inputs to backward function
    char *name;
};

BackwardFn *backward_fn_init(CallableGradFn grad_fn, Tensor **tensors,
                             size_t num_inputs, size_t num_outputs,
                             const char *name) {
    BackwardFn *backward_fn = malloc(sizeof(BackwardFn));
    if (!backward_fn) {
        printf("Failure to allocate GradFn\n");
        exit(BACKWARD_FN_INIT_FAILURE);
    }

    backward_fn->tensors = malloc(num_inputs * sizeof(Tensor *));
    if (!backward_fn->tensors) {
        printf("Failure to allocate BackwardFn\n");
        exit(BACKWARD_FN_INIT_FAILURE);
    }

    memcpy(backward_fn->tensors, tensors, num_inputs * sizeof(Tensor *));

    backward_fn->grad_fn = grad_fn;
    backward_fn->num_inputs = num_inputs;
    backward_fn->num_outputs = num_outputs;
    backward_fn->next_functions = NULL;
    backward_fn->name = strdup(name);

    return backward_fn;
}

BackwardFn **create_next_fns(Tensor **output_tensors, size_t num_outputs) {
    BackwardFn **next_functions = malloc(num_outputs * sizeof(BackwardFn *));
    if (!next_functions) {
        printf("Failure to create next functions\n");
        exit(NEXT_FNS_INIT_FAILURE);
    }

    for (size_t i = 0; i < num_outputs; i++) {
        if (!get_requires_grad(output_tensors[i])) {
            next_functions[i] = NULL;
            continue;
        }

        BackwardFn *backward_fn = get_backward_fn(output_tensors[i]);
        if (!backward_fn) {
            backward_fn = AccumulateGrad(output_tensors[i]);
            set_backward_fn(output_tensors[i], backward_fn);
        }
        next_functions[i] = backward_fn;
    }

    return next_functions;
}

void free_backward_fn(BackwardFn *backward_fn) {
    if (!backward_fn)
        return;

    free(backward_fn->tensors);
    free(backward_fn->next_functions);
    free(backward_fn->name);
    free(backward_fn);
}

BackwardFn **get_next_functions(const BackwardFn *backward_fn) {
    return backward_fn->next_functions;
}

char *get_backward_name(const BackwardFn *backward_fn) {
    return backward_fn->name;
}

size_t get_backward_inputs(const BackwardFn *backward_fn) {
    return backward_fn->num_inputs;
}

size_t get_backward_outputs(const BackwardFn *backward_fn) {
    return backward_fn->num_outputs;
}

Tensor **get_backward_fn_tensors(const BackwardFn *backward_fn) {
    return backward_fn->tensors;
}

CallableGradFn get_grad_fn(const BackwardFn *backward_fn) {
    return backward_fn->grad_fn;
}

void set_next_functions(BackwardFn *backward_fn, BackwardFn **next_functions) {
    backward_fn->next_functions = next_functions;
}
