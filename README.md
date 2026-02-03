# CTorch

# TODO

- [x] Gradients as Tensors
- [x] Autograd for tensor arithmetic
- [x] Backpropagation loop
- [x] Random tensor initialization
- [ ] nn Modules
  - [ ] Parent Module
  - [ ] Sequential Container
  - [ ] Linear Module
  - [ ] Activations
  - [ ] Loss Functions
- [x] Autograd pipeline for higher order derivatives
- [x] Parallelize with OpenMP
- [ ] Visualization Module
  - [ ] Line plots
  - [ ] Heatmaps
  - [ ] Imshow
  - [ ] Dot graphs for forward and backward mode computation graphs
  - [ ] Dot graphs for neural nets
- [ ] Refactor autograd
  - [ ] Callable gradient to a `void` function.
        So that, mallocs wouldn't be necessary for every grad_fn call.
  - [ ] `gradient` function to take `grads` as a parameter
        instead of returning it.
- [ ] Prepare documentation
