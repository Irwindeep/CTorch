#include "array.h"
#include "error_codes.h"
#include "nn.h"
#include "random.h"
#include "tensor.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct linear {
    Module base;
    Tensor *weight;
    Tensor *bias;
};

Tensor *linear_forward(void *module, Tensor *input) {
    linear *m = (linear *)module;
    return _linear(input, m->weight, m->bias);
}

linear *_Linear(size_t in_features, size_t out_features, bool bias) {
    linear *layer = calloc(1, sizeof(linear));
    if (!layer)
        RUNTIME_ERROR(MODULE_ALLOC_FAILURE, "Failed to allocate Linear layer");

    module_init(&layer->base);
    Environment *env = get_environ(&layer->base);

    layer->base.forward = linear_forward;

    char tmp[128];
    snprintf(tmp, 128, "Linear(in_features=%zu, out_features=%zu, bias=%s)",
             in_features, out_features, bias ? "True" : "False");
    layer->base.repr = strdup(tmp);
    layer->base.repr_dynamic = true;

    float bound = 1 / sqrtf((float)in_features);
    layer->weight = uniform(SHAPE(in_features, out_features), bound,
                            DTYPE_FLOAT, true, env);
    if (bias)
        layer->bias =
            uniform(SHAPE(1, out_features), bound, DTYPE_FLOAT, true, env);

    set_lock(env);
    return layer;
}

struct relu {
    Module base;
};

Tensor *relu_forward(void *module, Tensor *input) { return _relu(input); }

relu *_ReLU() {
    relu *layer = calloc(1, sizeof(relu));
    if (!layer)
        RUNTIME_ERROR(MODULE_ALLOC_FAILURE, "Failed to allocate ReLU layer");

    module_init(&layer->base);
    layer->base.forward = relu_forward;
    layer->base.repr = "ReLU()";

    Environment *env = get_environ(&layer->base);
    set_lock(env);
    return layer;
}
