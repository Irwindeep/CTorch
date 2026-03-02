#include "array.h"
#include "autograd.h"
#include "ctorch.h"
#include "nn.h"
#include "pbar.h"
#include "print.h"
#include "random.h"
#include "tensor.h"

#include <stdio.h>

void zero_grad_p(size_t num_params, Tensor **params) {
    for (size_t i = 0; i < num_params; i++)
        zero_grad(params[i]);
}

void sgd_step(size_t num_params, Tensor **params, ndArray *lr) {
    for (size_t i = 0; i < num_params; i++) {
        ndArray *param_data = get_tensor_data(params[i]);
        ndArray *params_grad =
            array_mul(get_tensor_data(get_tensor_grad(params[i])), lr);

        ndArray *new_data = array_sub(param_data, params_grad);
        replace_tensor_data(params[i], new_data);
        free_array(params_grad);
    }
}

int main() {
    CTorchInit();
    ManualSeed(12);

    Module *model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10));

    size_t num_trainable_vars = num_trainable_variables(model);
    size_t num_non_trainable_vars = num_non_trainable_variables(model);

    size_t num_params = num_parameters(model);
    Tensor *params[num_params];
    parameters(model, params);

    printf("# Trainable Variables: ");
    print_with_commas(num_trainable_vars);
    printf("# Non-Trainable Variables: ");
    print_with_commas(num_non_trainable_vars);

    printf("\n%s\n\n", model->repr);

    ndArray *lr = array_init(SHAPE(), DTYPE_FLOAT);
    populate_array(lr, (const float[]){1e-3});

    int num_batchs = 60000 / 128;
    char desc[] = "Epoch [1/1]";
    char postfix[128] = "";
    ProgressBar *pbar = progress_init(num_batchs);
    for (int batch = 1; batch <= num_batchs; batch++) {
        ScopedEnvironment env = env_init();
        zero_grad_p(num_params, params);

        Tensor *x = randn(SHAPE(128, 784), DTYPE_FLOAT, NO_GRAD, env);
        Tensor *y = module_call(model, x);

        Tensor *loss = tensor_sub(y, SCALAR_NG(2.0f, env));
        loss = tensor_sum(tensor_mul(loss, loss));
        loss = tensor_div(loss, SCALAR_NG(128.0f, env));

        backward(loss, NULL);
        sgd_step(num_params, params, lr);

        snprintf(postfix, 128, "Train Loss: %.4f", item(loss).float_val);
        progress_update(pbar, batch, desc, postfix);
    }
    progress_finish(pbar);

    free_array(lr);
    free_module(model);

    CTorchClose();
    return 0;
}
