#include "nn.h"
#include "tensor.h"

#include <stdbool.h>

Tensor *_linear(Tensor *input, Tensor *weight, Tensor *bias) {
    Tensor *output = tensor_matmul(input, weight);
    if (bias)
        output = tensor_add(output, bias);

    return output;
}

Tensor *_relu(Tensor *input) {
    bool requires_grad = get_requires_grad(input);
    Environment *env = get_tensor_environ(input);

    return tensor_max(zeros_like(input, requires_grad, env), input);
}
