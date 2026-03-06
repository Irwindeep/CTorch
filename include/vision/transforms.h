#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "tensor.h"
#include "vision/vision.h"

#include <stddef.h>

typedef struct Transform Transform;

typedef void *(*ImageTransformFunc)(Transform *transform, Image *input);
typedef void *(*TensorTransformFunc)(Transform *transform, Tensor *input);
typedef void *(*AnyTransformFunc)(Transform *transform, void *input);

typedef enum TransformType {
    ImageFunc,
    TensorFunc,
    AnyFunc,
} TransformType;

struct Transform {
    union {
        ImageTransformFunc image_fn;
        TensorTransformFunc tensor_fn;
        AnyTransformFunc any_fn;
    } func;
    TransformType func_type;

    Transform **transforms;
    size_t num_transforms;
};

Transform *Grayscale(int num_channels);
Transform *Resize(size_t height, size_t width);
Transform *ToTensor(Environment *env);
Transform *_Compose(size_t num_transforms, Transform **transforms);

#define Compose(...)                                                           \
    _Compose((sizeof((Transform *[]){__VA_ARGS__}) / sizeof(Transform *)),     \
             ((Transform *[]){__VA_ARGS__}))

void *transform_call(Transform *transform, void *input);

void free_transform(Transform **transform);

#if defined(__GNUC__) || defined(__clang__)
#define ScopedTransform __attribute__((cleanup(free_transform))) Transform *
#else
#define ScopedEnvironment Transform *
#endif

#endif // !TRANSFORMS_H
