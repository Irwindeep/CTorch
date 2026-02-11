# CTorch

Example usage:

```c
AutoEnvironment env = env_init();

Tensor *x = randn(SHAPE(1, 6), DTYPE_FLOAT, NO_GRAD, env);
Tensor *W = randn(SHAPE(6, 1), DTYPE_FLOAT, REQUIRES_GRAD, env);
Tensor *b = randn(SHAPE(), DTYPE_FLOAT, REQUIRES_GRAD, env);

Tensor *y = tensor_matmul(x, W);
y = tensor_mul(b, y);
y = tensor_sum(y);

print_tensor(x);
print_tensor(W);
print_tensor(b);
print_tensor(y);

Tensor *grads[2] = {0};
gradient(grads, TENSORS(W, b), TENSORS(y),
         TENSORS_(ones_like(y, NO_GRAD, env)), CREATE_GRAPH);

// higher order gradient backprop
Tensor *grad = tensor_mul(grads[0], grads[1]);
Tensor *grad_t = tensor_transpose(grad, NULL);
Tensor *norm2 = tensor_matmul(grad, grad_t);
Tensor *diff = tensor_sub(norm2, SCALAR_NG(1.0f, env));
Tensor *gp = tensor_mul(diff, diff);

print_tensor(grad);

backward(gp, ones_like(gp, REQUIRES_GRAD, env));

print_tensor(get_tensor_grad(x));
print_tensor(get_tensor_grad(W));
print_tensor(get_tensor_grad(b));
```

## TODO

- [x] Gradients as Tensors
- [x] Autograd for tensor arithmetic
- [x] Backpropagation loop
- [x] Random tensor initialization
- [ ] Tensor backward hooks
- [ ] nn Modules
  - [x] Parent Module
  - [x] Sequential Container
  - [x] Linear Module
  - [ ] Activations
  - [ ] Loss Functions
  - [ ] Module forward/backward hooks
- [ ] Optimizers
  - [ ] SGD, Adam
  - [ ] Optimizers with nesterov momentum
  - [ ] Explicit optimizers for with/without momentum
- [x] Autograd pipeline for higher order derivatives
- [x] Parallelize with OpenMP
- [ ] BLAS for matmul operations
- [ ] SIMD for elementwise ops
- [ ] Visualization Module
  - [ ] Line plots
  - [ ] Heatmaps
  - [ ] Imshow
  - [ ] Dot graphs for forward and backward mode computation graphs
  - [ ] Dot graphs for neural nets
- [x] Refactor autograd
  - [x] Callable gradient to a `void` function.
        So that, mallocs wouldn't be necessary for every grad_fn call.
  - [x] `gradient` function to take `grads` as a parameter
        instead of returning it.
- [ ] Prepare documentation
