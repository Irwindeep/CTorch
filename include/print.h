#ifndef PRINT_H
#define PRINT_H

#include "array.h"
#include "tensor.h"

#define BASE_INDENT 7
#define PRINT_EDGE_ITEMS 3

void print_with_commas(size_t num);

void print_array(const ndArray *array, int base_indent);
void print_shape(const ndArray *array);

void print_tensor(const Tensor *tensor);
void print_tensor_shape(const Tensor *tensor);

void print_grad_fn(const Tensor *tensor);
void print_next_functions(const BackwardFn *backward_fn);

#endif // !PRINT_H
