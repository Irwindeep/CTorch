#ifndef DATASETS_H
#define DATASETS_H

#include "utils/data.h"
#include "vision/transforms.h"

Dataset *MNISTDataset(const char *root, SplitType split, Transform *transform);
Dataset *CIFAR10Dataset(const char *root, SplitType split,
                        Transform *transform);

#endif // !DATASETS_H
