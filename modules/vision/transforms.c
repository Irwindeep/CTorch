#include "vision/transforms.h"

#include <stddef.h>
#include <stdlib.h>
#include <vips/vips.h>

void *transform_call(Transform *transform, void *input) {
    TransformType func_type = transform->func_type;

    switch (func_type) {
    default:
        return NULL;
    case ImageFunc:
        return transform->func.image_fn(transform, input);
    case TensorFunc:
        return transform->func.tensor_fn(transform, input);
    case AnyFunc:
        return transform->func.any_fn(transform, input);
    }
}

void free_transform(Transform **transform) {
    if (!transform || !*transform)
        return;

    if ((*transform)->transforms) {
        for (size_t i = 0; i < (*transform)->num_transforms; i++)
            free_transform(&(*transform)->transforms[i]);

        free((*transform)->transforms);
    }

    free(*transform);
    *transform = NULL;
}
