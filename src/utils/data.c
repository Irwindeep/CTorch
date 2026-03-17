#include "utils/data.h"
#include "error_codes.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BUFF_LEN 64

void dataset_init(Dataset *dataset, __get_item__ get_item) {
    dataset->size = 0;
    dataset->__get_item__ = get_item;

    dataset->buff_capacity = 1;
    dataset->num_buffers = 0;
    dataset->buffers = malloc(dataset->buff_capacity * sizeof(Buffer));
}

void register_buff(Dataset *dataset, Buffer buffer) {
    if (dataset->buff_capacity == dataset->num_buffers) {
        dataset->buff_capacity *= 2;
        dataset->buffers =
            realloc(dataset->buffers, dataset->buff_capacity * sizeof(Buffer));
    }

    dataset->buffers[dataset->num_buffers++] = buffer;
}

typedef struct _TensorDataset {
    Dataset dataset;

    int num_tensors;
    Tensor **tensors;
} _TensorDataset;

void tensor_dataset_free_tensor(void *tensor) { free_tensor((Tensor *)tensor); }

void tensor_dataset_get_item(Dataset *dataset, ssize_t idx, void *buffer) {
    Tensor **tensors = (Tensor **)buffer;
    _TensorDataset *_dataset = (_TensorDataset *)dataset;

    char slice_str[MAX_BUFF_LEN];
    snprintf(slice_str, MAX_BUFF_LEN, "%zd", idx);

    for (int i = 0; i < _dataset->num_tensors; i++)
        tensors[i] = tensor_slice(_dataset->tensors[i], slice_str);
}

Dataset *TensorDataset(int num_tensors, Tensor **tensors) {
    _TensorDataset *dataset = calloc(1, sizeof(_TensorDataset));
    dataset_init(&dataset->dataset, tensor_dataset_get_item);

    dataset->num_tensors = num_tensors;
    dataset->tensors = tensors;

    size_t size = 0;
    for (int i = 0; i < num_tensors; i++) {
        int ndim = get_tensor_ndim(tensors[i]);
        const size_t *shape = get_tensor_shape(tensors[i]);

        if (ndim == 0)
            RUNTIME_ERROR(INVALID_ARRAY, "Expected all non-zero dim tensors");

        size_t tensor_size = shape[0];
        if (i > 0 && tensor_size != size)
            RUNTIME_ERROR(INVALID_ARRAY, "Expected same size for all tensors");

        size = tensor_size;
        Environment *env = get_tensor_environ(tensors[i]);
        if (env) {
            bool removed = env_remove(env, tensors[i]);
            if (!removed)
                RUNTIME_ERROR(ENV_RESOLVE_FAILURE, "Failed to remove tensor");

            set_tensor_environ(tensors[i], NULL);
        }

        register_buff(
            &dataset->dataset,
            (Buffer){.ptr = tensors[i], .free_fn = tensor_dataset_free_tensor});
    }

    dataset->dataset.size = size;
    return &dataset->dataset;
}

void get_index(Dataset *dataset, ssize_t idx, void *buffer) {
    dataset->__get_item__(dataset, idx, buffer);
}

void free_dataset(Dataset **dataset) {
    if (!dataset || !*dataset)
        return;

    for (size_t i = 0; i < (*dataset)->num_buffers; i++) {
        Buffer buffer = (*dataset)->buffers[i];
        buffer.free_fn(buffer.ptr);
    }

    free((*dataset)->buffers);
    free(*dataset);

    *dataset = NULL;
}
