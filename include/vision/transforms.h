#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "tensor.h"

#include <vips/vips.h>

Tensor *ToTensor(VipsImage *image, Environment *env);

#endif // !TRANSFORMS_H
