#include "vision/transforms.h"
#include "vision/vision.h"

#include "array.h"
#include "tensor.h"

#include <stddef.h>
#include <vips/vips.h>

Tensor *ToTensor(VipsImage *image, Environment *env) {
    int channels = vips_image_get_bands(image);
    int height = vips_image_get_height(image);
    int width = vips_image_get_width(image);

    const size_t shape[] = {(size_t)channels, (size_t)height, (size_t)width};
    int ndim = sizeof(shape) / sizeof(shape[0]);

    size_t size = (size_t)channels * (size_t)height * (size_t)width;

    ndArray *data = array_init(ndim, shape, DTYPE_FLOAT);
    float *arr = (float *)get_array_data(data);

    size_t mem_size;
    unsigned char *buffer = vips_image_write_to_memory(image, &mem_size);

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

    Tensor *tensor = tensor_init(data, NO_GRAD, env);
    return tensor;
}
