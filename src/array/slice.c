#include "array.h"
#include "error_codes.h"

#include <stdbool.h>
#include <stddef.h>

ndArray *array_slice(ndArray *array, const Slice *slices) {
    int ndim = get_ndim(array);
    const size_t *shape = get_shape(array), *strides = get_strides(array);

    ndArray *new_array = shallow_copy_array(array);
    size_t *new_shape = get_shape(new_array),
           *new_strides = get_strides(new_array);

    char *data_ptr = (char *)get_array_data(array);
    bool any_index = false;

    for (int i = 0; i < ndim; i++) {
        size_t start = slices[i].start, end = slices[i].end,
               step = slices[i].step;

        if (start >= shape[i] || end > shape[i])
            RUNTIME_ERROR(INVALID_IDX, "Invalid slice");

        data_ptr += start * strides[i];

        new_shape[i] = (end - start + step - 1) / step;
        new_strides[i] = strides[i] * step;

        if (slices[i].is_index)
            any_index = true;
    }

    if (any_index) {
        int j = 0;
        for (int i = 0; i < ndim; i++) {
            if (!slices[i].is_index) {
                new_shape[j] = new_shape[i];
                new_strides[j] = new_strides[i];
                j++;
            }
        }
        set_ndim(new_array, j);
    }

    set_array_data(new_array, data_ptr);
    recompute_total_size(new_array);

    return new_array;
}

void scatter_add_slice(const ndArray *input, ndArray *output,
                       const Slice *slices) {
    int in_ndim = get_ndim(input);
    int out_ndim = get_ndim(output);

    DType dtype = get_dtype(input);

    const size_t *in_shape = get_shape(input);
    const size_t *in_strides = get_strides(input);

    const size_t *out_strides = get_strides(output);

    char *in_data = (char *)get_array_data(input);
    char *out_data = (char *)get_array_data(output);

    size_t idx[MAX_NDIM] = {0};
    size_t total = get_total_size(input);

    int in_dim = 0;

    for (size_t n = 0; n < total; n++) {

        size_t in_offset = 0;
        size_t out_offset = 0;

        in_dim = 0;

        for (int d = 0; d < out_ndim; d++) {

            if (slices[d].is_index) {
                size_t out_pos = slices[d].start;
                out_offset += out_pos * out_strides[d];
            } else {
                size_t i = idx[in_dim];
                in_offset += i * in_strides[in_dim];

                size_t out_pos = slices[d].start + i * slices[d].step;
                out_offset += out_pos * out_strides[d];
                in_dim++;
            }
        }

        switch (dtype) {
        case DTYPE_FLOAT:
            *(float *)(out_data + out_offset) +=
                *(float *)(in_data + in_offset);
            break;
        case DTYPE_DOUBLE:
            *(double *)(out_data + out_offset) +=
                *(double *)(in_data + in_offset);
            break;
        case DTYPE_INT:
            *(int *)(out_data + out_offset) += *(int *)(in_data + in_offset);
            break;
        case DTYPE_LONG:
            *(long *)(out_data + out_offset) += *(long *)(in_data + in_offset);
            break;
        }

        for (int d = in_ndim - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < in_shape[d])
                break;

            idx[d] = 0;
        }
    }
}
