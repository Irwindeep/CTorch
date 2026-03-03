#include "array.h"
#include "autograd.h"
#include "ctorch.h"
#include "nn.h"
#include "optim.h"
#include "pbar.h"
#include "print.h"
#include "random.h"
#include "tensor.h"

#include <stdio.h>

#define BATCH_SIZE 128

int main() {
    CTorchInit();
    ManualSeed(12);

    ScopedModule model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10));

    size_t num_trainable_vars = num_trainable_variables(model);
    size_t num_non_trainable_vars = num_non_trainable_variables(model);

    size_t num_params = num_parameters(model);
    Tensor *params[num_params];
    parameters(model, params);
    ScopedOptimizer optim = SGD(num_params, params, 1e-3);

    printf("# Trainable Variables: ");
    print_with_commas(num_trainable_vars);
    printf("# Non-Trainable Variables: ");
    print_with_commas(num_non_trainable_vars);

    printf("\n%s\n\n", model->repr);

    int num_batchs = 60000 / BATCH_SIZE;
    char desc[] = "Epoch [1/1]";
    char postfix[128] = "";
    ProgressBar *pbar = progress_init(num_batchs);
    for (int batch = 1; batch <= num_batchs; batch++) {
        ScopedEnvironment env = env_init();
        optim_zero_grad(optim);

        Tensor *x = randn(SHAPE(BATCH_SIZE, 784), DTYPE_FLOAT, NO_GRAD, env);
        Tensor *y = module_call(model, x);

        Tensor *loss = tensor_sub(y, SCALAR_NG(2.0f, env));
        loss = tensor_sum(tensor_mul(loss, loss));
        loss = tensor_div(loss, SCALAR_NG((float)BATCH_SIZE, env));

        backward(loss, NULL);
        optim_step(optim);

        snprintf(postfix, 128, "Train Loss: %.4f", item(loss).float_val);
        progress_update(pbar, batch, desc, postfix);
    }
    progress_finish(pbar);

    CTorchClose();
    return 0;
}
