# CTorch

# TODO

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
- [x] Autograd pipeline for higher order derivatives
- [x] Parallelize with OpenMP
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
