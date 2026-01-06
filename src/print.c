#include "print.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static void print_indent(int n) {
    for (int i = 0; i < n; ++i)
        putchar(' ');
}

static void _print_rec_base_case(const ndArray *array, size_t offset) {
    const char *base = (char *)array->data + offset;

    float v;
    memcpy(&v, base, sizeof(float));
    printf("%g", (double)v);
}

static int _print_trunc(const ndArray *array, bool truncated, size_t idx,
                        int dim, int indent) {
    if (truncated && idx == PRINT_EDGE_ITEMS) {
        if (idx > 0)
            printf(",");
        printf("\n");
        print_indent(indent + 2);
        printf("...");
        idx = array->shape[dim] - PRINT_EDGE_ITEMS;
    }

    if (idx > 0)
        printf(",");

    if (dim < array->ndim - 1) {
        printf("\n");
        print_indent(indent + 2);
    } else
        putchar(' ');

    return idx;
}

static void print_rec(const ndArray *array, int dim, size_t offset,
                      int indent) {
    if (array == NULL) {
        printf("NULL");
        return;
    }

    int ndim = array->ndim;
    if (ndim == 0 || dim == ndim) {
        _print_rec_base_case(array, offset);
        return;
    }

    size_t dim_len = array->shape[dim];
    size_t stride = array->strides[dim];

    putchar('[');

    bool truncated = dim_len > 2 * PRINT_EDGE_ITEMS + 1;
    for (size_t i = 0; i < dim_len; ++i) {
        i = _print_trunc(array, truncated, i, dim, indent);
        print_rec(array, dim + 1, offset + i * stride, indent + 2);
    }

    if (dim < ndim - 1 && dim_len > 0) {
        printf("\n");
        print_indent(indent);
    } else if (dim == ndim - 1 && dim_len > 0) {
        putchar(' ');
    }

    putchar(']');
}

void print_array(const ndArray *array) {
    if (array == NULL) {
        printf("NULL\n");
        return;
    }

    if (array->ndim < 0) {
        printf("Invalid ndim (%d)\n", array->ndim);
        return;
    }

    for (int d = 0; d < array->ndim; ++d) {
        if (array->shape[d] == 0) {
            printf("[]\n");
            return;
        }
    }
    print_rec(array, 0, 0, 0);
    putchar('\n');
}

void print_shape(const ndArray *array, char end) {
    printf("(");
    for (int i = 0; i < array->ndim; i++) {
        printf("%zu", array->shape[i]);
        if (i < array->ndim - 1)
            printf(", ");
    }
    printf(")%c", end);
}
