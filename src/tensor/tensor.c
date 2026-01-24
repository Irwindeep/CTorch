#include "tensor.h"
#include "array.h"
#include "autograd.h"
#include "error_codes.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Tensor {
    ndArray *data;
    Tensor *grad;
    BackwardFn *backward_fn;
    Environment *environ;
    bool requires_grad;
};

struct TensorHeader {
    char magic[8]; // "C-TENSOR"
    uint32_t dtype;
    uint32_t ndim;
    uint64_t buffer_elems;
};

Tensor *tensor_init(ndArray *data, bool requires_grad, Environment *environ) {
    if (requires_grad) {
        DType dtype = get_dtype(data);
        if (!(dtype == DTYPE_FLOAT || dtype == DTYPE_DOUBLE))
            RUNTIME_ERROR(TENSOR_INIT_FAILURE,
                          "Invalid argument `requires_grad=True` for non-float "
                          "tensor");
    }

    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor)
        RUNTIME_ERROR(TENSOR_INIT_FAILURE, "Failure to allocate tensor\n");

    tensor->data = data;
    tensor->grad = NULL;

    tensor->backward_fn = NULL;
    tensor->environ = environ;
    tensor->requires_grad = requires_grad;

    if (environ)
        env_push(environ, tensor);

    return tensor;
}

void free_tensor(Tensor *tensor) {
    if (!tensor)
        return;

    free_array(tensor->data);
    free_backward_fn(tensor->backward_fn);

    free(tensor);
}

void save_tensor(Tensor *tensor, const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file)
        RUNTIME_ERRORF(FILE_WRITE_FAILURE,
                       "Failure to open write binary file: %s", path);

    ndArray *array = tensor->data;

    DType dtype = get_dtype(array);
    int ndim = get_ndim(array);
    const size_t *shape = get_shape(array);
    const size_t *strides = get_strides(array);
    size_t total_size = get_total_size(array);
    size_t itemsize = get_itemsize(array);
    const void *data = get_array_data(array);

    struct TensorHeader header = {
        .magic = "C-TENSOR",
        .dtype = (uint32_t)dtype,
        .ndim = (uint32_t)ndim,
        .buffer_elems = (uint64_t)total_size,
    };

    fwrite(&header, sizeof(header), 1, file);

    for (uint32_t d = 0; d < header.ndim; d++) {
        uint64_t dim = (uint64_t)shape[d];
        fwrite(&dim, sizeof(uint64_t), 1, file);
    }

    for (uint32_t d = 0; d < header.ndim; d++) {
        uint64_t stride = (uint64_t)strides[d];
        fwrite(&stride, sizeof(uint64_t), 1, file);
    }

    fwrite(data, itemsize, total_size, file);

    fclose(file);
}

Tensor *load_tensor(const char *path, bool requires_grad,
                    Environment *environ) {
    FILE *file = fopen(path, "rb");
    if (!file)
        RUNTIME_ERRORF(FILE_READ_FAILURE,
                       "Failure to open read binary file: %s", path);

    struct TensorHeader header;
    fread(&header, sizeof(header), 1, file);

    if (memcmp(header.magic, "C-TENSOR", 8) != 0)
        RUNTIME_ERROR(FILE_FORMAT_ERROR, "Invalid tensor file identifier");

    int ndim = (int)header.ndim;
    uint64_t total_elems = header.buffer_elems;
    DType dtype = (DType)header.dtype;

    size_t shape[ndim], strides[ndim];
    for (uint32_t d = 0; d < ndim; d++) {
        uint64_t dim;
        fread(&dim, sizeof(uint64_t), 1, file);
        shape[d] = (size_t)dim;
    }

    for (uint32_t d = 0; d < ndim; d++) {
        uint64_t stride;
        fread(&stride, sizeof(uint64_t), 1, file);
        strides[d] = (size_t)stride;
    }

    ndArray *array = array_init(ndim, shape, dtype);
    set_strides(array, strides);

    size_t itemsize = get_itemsize(array);
    void *data = get_array_data(array);

    fread(data, itemsize, total_elems, file);

    fclose(file);

    Tensor *tensor = tensor_init(array, requires_grad, environ);
    return tensor;
}

ndArray *get_tensor_data(const Tensor *tensor) { return tensor->data; }
Tensor *get_tensor_grad(const Tensor *tensor) { return tensor->grad; }
bool get_requires_grad(const Tensor *tensor) { return tensor->requires_grad; }
int get_tensor_ndim(const Tensor *tensor) { return get_ndim(tensor->data); }

size_t *get_tensor_shape(const Tensor *tensor) {
    return get_shape(tensor->data);
}

DType get_tensor_dtype(const Tensor *tensor) { return get_dtype(tensor->data); }
Environment *get_tensor_environ(const Tensor *tensor) {
    return tensor->environ;
}

BackwardFn *get_backward_fn(const Tensor *tensor) {
    return tensor->backward_fn;
}

void set_requires_grad(Tensor *tensor, bool requires_grad) {
    tensor->requires_grad = requires_grad;
}

void set_tensor_grad(Tensor *tensor, Tensor *grad) { tensor->grad = grad; }
void set_backward_fn(Tensor *tensor, BackwardFn *backward_fn) {
    tensor->backward_fn = backward_fn;
}

void zero_grad(Tensor *tensor) {
    int ndim = get_ndim(tensor->data);
    const size_t *shape = get_shape(tensor->data);
    DType dtype = get_dtype(tensor->data);
    tensor->grad = zeros_tensor(ndim, shape, dtype, false, tensor->environ);
}

Tensor *eye_tensor(size_t m, size_t n, DType dtype, bool requires_grad,
                   Environment *environ) {
    ndArray *data = eye(m, n, dtype);
    Tensor *tensor = tensor_init(data, requires_grad, environ);

    return tensor;
}

Tensor *zeros_tensor(int ndim, const size_t *shape, DType dtype,
                     bool requires_grad, Environment *environ) {
    ndArray *data = zeros(ndim, shape, dtype);
    Tensor *tensor = tensor_init(data, requires_grad, environ);

    return tensor;
}

Tensor *ones_tensor(int ndim, const size_t *shape, DType dtype,
                    bool requires_grad, Environment *environ) {
    ndArray *data = ones(ndim, shape, dtype);
    Tensor *tensor = tensor_init(data, requires_grad, environ);

    return tensor;
}

Tensor *zeros_like(const Tensor *tensor, bool requires_grad, Environment *env) {
    int ndim = get_tensor_ndim(tensor);
    const size_t *shape = get_tensor_shape(tensor);
    DType dtype = get_tensor_dtype(tensor);

    Tensor *output = zeros_tensor(ndim, shape, dtype, requires_grad, env);
    return output;
}

Tensor *ones_like(const Tensor *tensor, bool requires_grad, Environment *env) {
    int ndim = get_tensor_ndim(tensor);
    const size_t *shape = get_tensor_shape(tensor);
    DType dtype = get_tensor_dtype(tensor);

    Tensor *output = ones_tensor(ndim, shape, dtype, requires_grad, env);
    return output;
}

Tensor *scalar(ArrayVal value, DType dtype, bool requires_grad,
               Environment *environ) {
    ndArray *data = array_init(0, (const size_t[]){}, dtype);
    set_value(data, NULL, value);

    Tensor *tensor = tensor_init(data, requires_grad, environ);
    return tensor;
}
