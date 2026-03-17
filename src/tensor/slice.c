#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFF_LEN 256

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t')
        s++;

    char *end = s + strlen(s) - 1;
    while (end > s && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
    }

    return s;
}

static void parse_slice_str(Tensor *tensor, char *slice_str, Slice *slices) {
    const size_t *shape = get_tensor_shape(tensor);
    int i = 0, ndim = get_tensor_ndim(tensor);

    char *save = NULL;
    char *token = trim(strtok_r(slice_str, ",", &save));
    while (token && i < ndim) {
        token = trim(token);

        ssize_t start = 0;
        ssize_t end = (ssize_t)shape[i];
        ssize_t step = 1;
        bool is_index = false;

        if (strchr(token, ':')) {
            char *ptr = token;

            char *start_token = strsep(&ptr, ":");
            char *end_token = strsep(&ptr, ":");
            char *step_token = strsep(&ptr, ":");

            if (start_token && *start_token)
                start = strtol(start_token, NULL, 10);
            if (end_token && *end_token)
                end = strtol(end_token, NULL, 10);
            if (step_token && *step_token)
                step = strtol(step_token, NULL, 10);
        } else {
            start = strtol(token, NULL, 10);
            end = start + 1;
            is_index = true;
        }

        Slice s;
        if (start < 0)
            start = start + (ssize_t)shape[i];
        if (end < 0)
            end = end + (ssize_t)shape[i];

        s.start = (size_t)start;
        s.end = (size_t)end;
        s.step = (size_t)step;
        s.is_index = is_index;

        slices[i] = s;
        token = strtok_r(NULL, ",", &save);

        i++;
    }

    while (i < ndim) {
        slices[i].start = 0;
        slices[i].end = shape[i];
        slices[i].step = 1;
        slices[i].is_index = false;
        i++;
    }
}

Tensor *tensor_slice(Tensor *tensor, char *slice_str) {
    char str[MAX_BUFF_LEN];
    snprintf(str, sizeof(str), "%s", slice_str);

    Slice slices[MAX_NDIM];
    parse_slice_str(tensor, str, slices);

    ndArray *data_ = get_tensor_data(tensor);
    ndArray *data = array_slice(data_, slices);

    bool requires_grad = get_requires_grad(tensor);
    Environment *env = get_tensor_environ(tensor);

    Tensor *new_tensor = tensor_init(data, requires_grad, env);
    if (requires_grad) {
        BackwardFn *backward_fn = SelectBackward((Tensor *[]){new_tensor},
                                                 (Tensor *[]){tensor}, 1, 1);

        SliceCtx ctx = {.ndim = get_tensor_ndim(tensor),
                        .slice_str = slice_str,
                        .slices = slices};
        set_ctx(backward_fn, &ctx, SLICE_CTX);
        set_backward_fn(new_tensor, backward_fn);
    }

    return new_tensor;
}

Tensor *tensor_scatter_add(Tensor *t1, Tensor *t2, char *slice_str) {
    char str[MAX_BUFF_LEN];
    snprintf(str, sizeof(str), "%s", slice_str);

    Slice slices[MAX_NDIM];
    parse_slice_str(t2, str, slices);

    const ndArray *data1 = get_tensor_data(t1), *data2 = get_tensor_data(t2);

    ndArray *data = shallow_copy_array(data2);
    bool requires_grad = get_requires_grad(t1);
    Environment *env = resolve_environ(t1, t2);

    Tensor *tensor = tensor_init(data, requires_grad, env);
    scatter_add_slice(data1, data, slices);

    if (requires_grad) {
        BackwardFn *backward_fn = ScatterAddBackward(
            (Tensor *[]){tensor}, (Tensor *[]){t1, t2}, 1, 2);

        SliceCtx ctx = {.ndim = get_tensor_ndim(t2),
                        .slice_str = slice_str,
                        .slices = slices};
        set_ctx(backward_fn, &ctx, SLICE_CTX);
        set_backward_fn(tensor, backward_fn);
    }

    return tensor;
}
