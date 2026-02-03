#include "print.h"
#include "array.h"
#include "autograd.h"
#include "tensor.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_indent(int n) {
    for (int i = 0; i < n; ++i)
        putchar(' ');
}

static int format_value(char *buf, size_t n, const ndArray *array,
                        const size_t *idx) {
    ArrayVal v = get_value(array, idx);

    switch (get_dtype(array)) {
    case DTYPE_INT:
        return snprintf(buf, n, "%d", v.int_val);
    case DTYPE_LONG:
        return snprintf(buf, n, "%ld", v.long_val);
    case DTYPE_FLOAT: {
        double x = (double)v.float_val;
        double ax = fabs(x);
        if ((ax != 0.0 && ax < 1e-4) || ax >= 1e6)
            return snprintf(buf, n, "%.4e", x);
        return snprintf(buf, n, "%.4f", x);
    }
    case DTYPE_DOUBLE: {
        double x = v.double_val;
        double ax = fabs(x);
        if ((ax != 0.0 && ax < 1e-4) || ax >= 1e6)
            return snprintf(buf, n, "%.4e", x);
        return snprintf(buf, n, "%.4f", x);
    }
    }
    return 0;
}

static inline void print_sep(bool multiline, int ndim, int dim, int base_indent,
                             int indent) {
    if (!multiline) {
        printf(", ");
        return;
    }

    printf(",\n");
    if (ndim - dim > 2)
        putchar('\n');

    print_indent(base_indent + indent + 1);
}

static void print_rec(const ndArray *array, int dim, size_t *idx, int indent,
                      int base_indent, int col_width) {
    int ndim = get_ndim(array);

    if (dim == ndim) {
        char buf[64];
        format_value(buf, sizeof(buf), array, idx);
        printf("%*s", col_width, buf);
        return;
    }

    size_t dim_len = get_shape(array)[dim];
    bool multiline = (ndim - dim) > 1;

    bool truncated = (dim_len > 2 * PRINT_EDGE_ITEMS + 1);

    putchar('[');

    size_t head = truncated ? PRINT_EDGE_ITEMS : dim_len;
    for (size_t i = 0; i < head; ++i) {
        if (i > 0)
            print_sep(multiline, ndim, dim, base_indent, indent);

        idx[dim] = i;
        print_rec(array, dim + 1, idx, indent + 1, base_indent, col_width);
    }

    if (truncated) {
        print_sep(multiline, ndim, dim, base_indent, indent);
        printf("...");
        for (size_t i = dim_len - PRINT_EDGE_ITEMS; i < dim_len; ++i) {
            print_sep(multiline, ndim, dim, base_indent, indent);

            idx[dim] = i;
            print_rec(array, dim + 1, idx, indent + 1, base_indent, col_width);
        }
    }

    putchar(']');
}

static void compute_max_width_rec(const ndArray *array, int dim, size_t *idx,
                                  int *max_width) {
    int ndim = get_ndim(array);

    if (dim == ndim) {
        char buf[64];
        int w = format_value(buf, sizeof(buf), array, idx);
        if (w > *max_width)
            *max_width = w;
        return;
    }

    size_t dim_len = get_shape(array)[dim];
    bool truncated = (dim_len > 2 * PRINT_EDGE_ITEMS + 1);

    size_t head = truncated ? PRINT_EDGE_ITEMS : dim_len;
    for (size_t i = 0; i < head; ++i) {
        idx[dim] = i;
        compute_max_width_rec(array, dim + 1, idx, max_width);
    }

    if (truncated) {
        for (size_t i = dim_len - PRINT_EDGE_ITEMS; i < dim_len; ++i) {
            idx[dim] = i;
            compute_max_width_rec(array, dim + 1, idx, max_width);
        }
    }
}

static int compute_max_width(const ndArray *array) {
    int ndim = get_ndim(array);
    size_t idx[ndim];
    int max_width = 0;

    for (int d = 0; d < ndim; ++d)
        idx[d] = 0;

    compute_max_width_rec(array, 0, idx, &max_width);
    return max_width;
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

    size_t idx[get_ndim(array)];

    for (int d = 0; d < get_ndim(array); d++) {
        idx[d] = 0;
        if (get_shape(array)[d] == 0) {
            printf("[]\n");
            return;
        }
    }
    int max_width = compute_max_width(array);
    print_rec(array, 0, idx, 0, BASE_INDENT, max_width);
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
    if (!tensor) {
        printf("NULL\n");
        return;
    }

    const ndArray *data = get_tensor_data(tensor);
    printf("Tensor(");
    print_array(data);

    bool requires_grad = get_requires_grad(tensor);
    BackwardFn *backward_fn = get_backward_fn(tensor);

    if (backward_fn)
        printf(", grad_fn=<%s>, size=", get_backward_name(backward_fn));
    else if (requires_grad)
        printf(", requires_grad=True, size=");
    else
        printf(", size=");

    print_shape(get_tensor_data(tensor));
    printf(")\n");
}

void print_tensor_shape(const Tensor *tensor) {
    const ndArray *data = get_tensor_data(tensor);
    print_shape(data);
    printf("\n");
}

void print_grad_fn(const Tensor *tensor) {
    BackwardFn *backward_fn = get_backward_fn(tensor);
    if (!backward_fn) {
        printf("NULL\n");
        return;
    }

    printf("<%s at %p>\n", get_backward_name(backward_fn), backward_fn);
}

void print_next_functions(const BackwardFn *backward_fn) {
    BackwardFn **next_functions = get_next_functions(backward_fn);
    size_t num_functions = get_backward_outputs(backward_fn);

    printf("(");
    for (size_t i = 0; i < num_functions; i++) {
        printf("<%s at %p>", get_backward_name(next_functions[i]),
               next_functions[i]);
        if (i < num_functions - 1)
            printf(", ");
    }

    printf(")\n");
}
