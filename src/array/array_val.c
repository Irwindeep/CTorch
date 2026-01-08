#include "array.h"

ArrayVal array_val_one(DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = 1;
        break;
    case DTYPE_FLOAT:
        value.float_val = 1.0f;
        break;
    case DTYPE_DOUBLE:
        value.double_val = 1.0;
        break;
    case DTYPE_LONG:
        value.long_val = 1;
        break;
    }

    return value;
}

ArrayVal array_val_zero(DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = 0;
        break;
    case DTYPE_FLOAT:
        value.float_val = 0.0f;
        break;
    case DTYPE_DOUBLE:
        value.double_val = 0.0;
        break;
    case DTYPE_LONG:
        value.long_val = 0;
        break;
    }

    return value;
}

ArrayVal array_val_add(ArrayVal v1, ArrayVal v2, DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = v1.int_val + v2.int_val;
        break;
    case DTYPE_FLOAT:
        value.float_val = v1.float_val + v2.float_val;
        break;
    case DTYPE_DOUBLE:
        value.double_val = v1.double_val + v2.double_val;
        break;
    case DTYPE_LONG:
        value.long_val = v1.long_val + v2.long_val;
        break;
    }

    return value;
}

ArrayVal array_val_sub(ArrayVal v1, ArrayVal v2, DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = v1.int_val - v2.int_val;
        break;
    case DTYPE_FLOAT:
        value.float_val = v1.float_val - v2.float_val;
        break;
    case DTYPE_DOUBLE:
        value.double_val = v1.double_val - v2.double_val;
        break;
    case DTYPE_LONG:
        value.long_val = v1.long_val - v2.long_val;
        break;
    }

    return value;
}

ArrayVal array_val_mul(ArrayVal v1, ArrayVal v2, DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = v1.int_val * v2.int_val;
        break;
    case DTYPE_FLOAT:
        value.float_val = v1.float_val * v2.float_val;
        break;
    case DTYPE_DOUBLE:
        value.double_val = v1.double_val * v2.double_val;
        break;
    case DTYPE_LONG:
        value.long_val = v1.long_val * v2.long_val;
        break;
    }

    return value;
}

ArrayVal array_val_div(ArrayVal v1, ArrayVal v2, DType dtype) {
    ArrayVal value;

    switch (dtype) {
    case DTYPE_INT:
        value.int_val = v1.int_val / v2.int_val;
        break;
    case DTYPE_FLOAT:
        value.float_val = v1.float_val / v2.float_val;
        break;
    case DTYPE_DOUBLE:
        value.double_val = v1.double_val / v2.double_val;
        break;
    case DTYPE_LONG:
        value.long_val = v1.long_val / v2.long_val;
        break;
    }

    return value;
}

static float float_abs(float x) { return (x > 0) ? x : -x; }
static double double_abs(double x) { return (x > 0) ? x : -x; }

bool array_val_equal(ArrayVal v1, ArrayVal v2, DType dtype) {
    bool is_equal;

    switch (dtype) {
    case DTYPE_INT:
        is_equal = v1.int_val == v2.int_val;
        break;
    case DTYPE_FLOAT:
        is_equal = float_abs(v1.float_val - v2.float_val) < FLOAT_EQ_TOL;
        break;
    case DTYPE_DOUBLE:
        is_equal = double_abs(v1.double_val - v2.double_val) < DOUBLE_EQ_TOL;
        break;
    case DTYPE_LONG:
        is_equal = v1.long_val == v2.long_val;
        break;
    }

    return is_equal;
}
