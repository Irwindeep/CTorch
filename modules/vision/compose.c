#include "vision/transforms.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void *compose(Transform *transform, void *input) {
    void *output = input;
    for (size_t i = 0; i < transform->num_transforms; i++) {
        Transform *t = transform->transforms[i];
        output = transform_call(t, output);
    }

    return output;
}

Transform *_Compose(size_t num_transforms, Transform **transforms) {
    Transform *transform = malloc(sizeof(Transform));

    transform->func_type = AnyFunc;
    transform->func.any_fn = compose;
    transform->num_transforms = num_transforms;

    transform->transforms = malloc(num_transforms * sizeof(Transform *));
    memcpy(transform->transforms, transforms,
           num_transforms * sizeof(Transform *));

    return transform;
}
