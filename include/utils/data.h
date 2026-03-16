#ifndef DATA_H
#define DATA_H

#include "tensor.h"

#include <stddef.h>

typedef enum SplitType {
    TrainSplit,
    ValSplit,
    TestSplit,
} SplitType;

typedef struct Buffer {
    void *ptr;
    void (*free_fn)(void *);
} Buffer;

typedef struct Dataset Dataset;
typedef void *(*__get_item__)(Dataset *dataset, size_t idx);

struct Dataset {
    size_t size;
    __get_item__ __get_item__;

    Buffer *buffers;
    size_t num_buffers;
    size_t buff_capacity;
};

typedef struct DataLoader DataLoader;

void register_buff(Dataset *dataset, Buffer buffer);
void dataset_init(Dataset *dataset, __get_item__ __get_item__);

Dataset *TensorDataset(int num_tensors, Tensor **tensors);
DataLoader *dataloader_init(Dataset *dataset, size_t batch_size, bool shuffle);

void *get_index(Dataset *dataset, size_t idx);

void free_dataset(Dataset **dataset);

#endif // !DATA_H
