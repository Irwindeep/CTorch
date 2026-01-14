#ifndef PRINT_H
#define PRINT_H

#include "array.h"
#include "tensor.h"

#define PRINT_EDGE_ITEMS 3

void print_array(const ndArray *array);
void print_shape(const ndArray *array);

void print_tensor(const Tensor *tensor);
void print_tensor_shape(const Tensor *tensor);

void print_grad_fn(const Tensor *tensor);
void print_next_functions(const BackwardFn *backward_fn);

#endif // !PRINT_H
