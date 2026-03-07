#include "array.h"
#include "error_codes.h"
#include "random.h"
#include "tensor.h"

#include "vision/transforms.h"
#include "vision/vision.h"

#include <stddef.h>
#include <stdlib.h>
#include <vips/vips.h>

typedef struct _Grayscale {
    Transform transform;
    int num_channels;
} _Grayscale;

void *grayscale(Transform *transform, Image *input) {
    _Grayscale *_transform = (_Grayscale *)transform;
    VipsImage *image = *(VipsImage **)input;

    int channels = vips_image_get_bands(image);
    if (channels == _transform->num_channels)
        return input;

    VipsImage *output = NULL;
    if (_transform->num_channels == 1)
        vips_bandmean(image, &output, NULL);

    else if (_transform->num_channels == 3) {
        if (channels == 1) {
            VipsImage *arr[3] = {image, image, image};
            vips_bandjoin(arr, &output, 3, NULL);
        } else
            vips_colourspace(image, &output, VIPS_INTERPRETATION_RGB, NULL);
    }

    free_image(input);

    Image *out = image_init(output);
    return out;
}

Transform *Grayscale(int num_channels) {
    if (num_channels != 1 && num_channels != 3)
        RUNTIME_ERRORF(
            INVALID_DIM,
            "Grayscale requires `num_channels` to be 1 or 3, provided %d",
            num_channels);

    _Grayscale *transform = calloc(1, sizeof(_Grayscale));

    transform->transform.func_type = ImageFunc;
    transform->transform.func.image_fn = grayscale;

    transform->num_channels = num_channels;
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

typedef struct _RandomRotation {
    Transform transform;
    float degrees;
    bool expand;
} _RandomRotation;

void *random_rotation(Transform *transform, Image *input) {
    _RandomRotation *_transform = (_RandomRotation *)transform;
    VipsImage *image = *(VipsImage **)input;

    Tensor *t = uniform(0, (const size_t[]){0}, _transform->degrees,
                        DTYPE_DOUBLE, NO_GRAD, NULL);
    double angle = item(t).double_val;

    VipsImage *output;
    vips_similarity(image, &output, "angle", angle, NULL);

    if (!_transform->expand) {
        VipsImage *tmp = output;
        int width = vips_image_get_width(image),
            height = vips_image_get_height(image);

        int left = (vips_image_get_width(tmp) - width) / 2,
            top = (vips_image_get_height(tmp) - height) / 2;

        vips_crop(tmp, &output, left, top, width, height, NULL);

        g_object_unref(tmp);
    }

    free_image(input);
    free_tensor(t);

    Image *out = image_init(output);
    return out;
}

Transform *RandomRotation(float degrees, bool expand) {
    _RandomRotation *transform = calloc(1, sizeof(_RandomRotation));

    transform->transform.func_type = ImageFunc;
    transform->transform.func.image_fn = random_rotation;

    transform->degrees = degrees;
    transform->expand = expand;
    return &transform->transform;
}

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
