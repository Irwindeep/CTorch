#include "print.h"
#include "array.h"
#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_indent(int n) {
    for (int i = 0; i < n; ++i)
        putchar(' ');
}

static int _print_trunc(const ndArray *array, bool truncated, size_t idx,
                        int dim, int indent) {
    if (truncated && idx == PRINT_EDGE_ITEMS) {
        if (idx > 0)
            printf(",");
        printf("\n");
        print_indent(indent + 2);
        printf("...");
        idx = get_shape(array)[dim] - PRINT_EDGE_ITEMS;
    }

    if (idx > 0)
        printf(",");

    if (dim < get_ndim(array) - 1) {
        printf("\n");
        print_indent(indent + 2);
    } else
        putchar(' ');

    return idx;
}

static void print_rec(const ndArray *array, int dim, size_t *idx, int indent) {
    if (array == NULL) {
        printf("NULL");
        return;
    }

    int ndim = get_ndim(array);
    if (ndim == 0 || dim == ndim) {
        ArrayVal value = get_value(array, idx);
        switch (get_dtype(array)) {
        case DTYPE_INT:
            printf("%d", value.int_val);
            break;
        case DTYPE_FLOAT:
            printf("%g", (double)value.float_val);
            break;
        case DTYPE_DOUBLE:
            printf("%g", value.double_val);
            break;
        case DTYPE_LONG:
            printf("%ld", value.long_val);
            break;
        }
        return;
    }
    size_t dim_len = get_shape(array)[dim];
    putchar('[');

    bool truncated = dim_len > 2 * PRINT_EDGE_ITEMS + 1;
    for (size_t i = 0; i < dim_len; ++i) {
        i = _print_trunc(array, truncated, i, dim, indent);
        idx[dim] = i;
        print_rec(array, dim + 1, idx, indent + 2);
    }

    if (dim < ndim - 1 && dim_len > 0) {
        printf("\n");
        print_indent(indent);
    } else if (dim == ndim - 1 && dim_len > 0)
        putchar(' ');

    putchar(']');
}

void print_array(const ndArray *array) {
    if (array == NULL) {
        printf("NULL\n");
        return;
    }

    if (get_ndim(array) < 0) {
        printf("Invalid ndim (%d)\n", get_ndim(array));
        return;
    }

    size_t *idx = malloc(get_ndim(array) * sizeof(size_t));
    if (!idx) {
        printf("Failed to create index buffer\n");
        exit(ARRAY_INIT_FAILURE);
    }

    for (int d = 0; d < get_ndim(array); d++) {
        idx[d] = 0;
        if (get_shape(array)[d] == 0) {
            printf("[]\n");
            return;
        }
    }
    print_rec(array, 0, idx, 0);

    free(idx);
}

void print_shape(const ndArray *array) {
    putchar('(');
    for (int i = 0; i < get_ndim(array); i++) {
        printf("%zu", get_shape(array)[i]);
        if (i < get_ndim(array) - 1)
            printf(", ");
    }
    putchar(')');
}

void print_tensor(const Tensor *tensor) {
    const ndArray *data = get_tensor_data(tensor);
    printf("Tensor(");
    print_array(data);
    printf("requires_grad=%s)", (get_requires_grad(tensor)) ? "True" : "False");
}

void print_tensor_shape(const Tensor *tensor) {
    const ndArray *data = get_tensor_data(tensor);
    print_shape(data);
}
