#include "vision/transforms.h"
#include "vision/vision.h"

#include "array.h"
#include "tensor.h"

#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <vips/vips.h>

typedef struct _ToTensor {
    Transform transform;
    Environment *env;
} _ToTensor;

void *to_tensor(Transform *transform, Image *image) {
    _ToTensor *_transform = (_ToTensor *)transform;
    VipsImage *img = *(VipsImage **)image;

    int channels = vips_image_get_bands(img);
    int height = vips_image_get_height(img);
    int width = vips_image_get_width(img);

    const size_t shape[] = {(size_t)channels, (size_t)height, (size_t)width};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    size_t size = (size_t)channels * (size_t)height * (size_t)width;

    ndArray *data = array_init(ndim, shape, DTYPE_FLOAT);
    float *arr = (float *)get_array_data(data);

    size_t mem_size;
    unsigned char *buffer = vips_image_write_to_memory(img, &mem_size);

    if (!buffer) {
        free_image(image);
        return NULL;
    }

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        arr[i] = (float)buffer[i] * (1.0f / 255.0f);
    }

    g_free(buffer);
    free_image(image);

    Tensor *tensor = tensor_init(data, NO_GRAD, _transform->env);
    return tensor;
}

Transform *ToTensor(Environment *env) {
    _ToTensor *transform = calloc(1, sizeof(_ToTensor));

    transform->transform.func_type = ImageFunc;
    transform->transform.func.image_fn = to_tensor;

    transform->env = env;
    return &transform->transform;
}

typedef struct _Resize {
    Transform transform;
    size_t height;
    size_t width;
} _Resize;

void *resize(Transform *transform, Image *input) {
    _Resize *_transform = (_Resize *)transform;
    VipsImage *image = *(VipsImage **)input;
    VipsImage *tmp = NULL, *output = NULL;

    int height = vips_image_get_height(image),
        width = vips_image_get_width(image);

    int target_h = (int)_transform->height, target_w = (int)_transform->width;

    double scale_h = (double)target_h / height;
    double scale_w = (double)target_w / width;
    double scale = VIPS_MAX(scale_h, scale_w);

    vips_resize(image, &tmp, scale, NULL);

    int left = (vips_image_get_width(tmp) - target_w) / 2,
        top = (vips_image_get_height(tmp) - target_h) / 2;
    vips_extract_area(tmp, &output, left, top, target_w, target_h, NULL);

    free_image(input);
    g_object_unref(tmp);

    Image *out = image_init(output);
    return out;
}

Transform *Resize(size_t height, size_t width) {
    _Resize *transform = calloc(1, sizeof(_Resize));

    transform->transform.func_type = ImageFunc;
    transform->transform.func.image_fn = resize;

    transform->height = height;
    transform->width = width;
    return &transform->transform;
}
