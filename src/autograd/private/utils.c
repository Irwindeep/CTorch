#include "array.h"
#include "callable_grads.h"

ndArray *broadcast_grad_data(ndArray *data, int ndim, const size_t *shape) {
    int ndims_added = get_ndim(data) - ndim;
    for (int i = 0; i < ndims_added; i++) {
        ndArray *tmp = data;
        data = array_sum_dim(data, 0, false);
        free_array(tmp);
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] == 1) {
            ndArray *tmp = data;
            data = array_sum_dim(data, i, true);
            free_array(tmp);
        }
    }

    return data;
}
