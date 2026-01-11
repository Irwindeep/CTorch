#include "array.h"
#include "array_tests.h"
#include <CUnit/CUnit.h>

void test_array_val_one() {
    ArrayVal value = array_val_one(DTYPE_INT);
    CU_ASSERT(value.int_val == 1);

    value = array_val_one(DTYPE_FLOAT);
    CU_ASSERT(value.float_val == 1.0f);

    value = array_val_one(DTYPE_DOUBLE);
    CU_ASSERT(value.double_val == 1.0);

    value = array_val_one(DTYPE_LONG);
    CU_ASSERT(value.long_val == 1);
}

void test_array_val_zero() {
    ArrayVal value = array_val_zero(DTYPE_INT);
    CU_ASSERT(value.int_val == 0);

    value = array_val_zero(DTYPE_FLOAT);
    CU_ASSERT(value.float_val == 0.0f);

    value = array_val_zero(DTYPE_DOUBLE);
    CU_ASSERT(value.double_val == 0.0);

    value = array_val_zero(DTYPE_LONG);
    CU_ASSERT(value.long_val == 0);
}

void test_array_val_add() {
    ArrayVal v1, v2;

    v1.double_val = 0.4;
    v2.double_val = 1.7;
    ArrayVal result = array_val_add(v1, v2, DTYPE_DOUBLE);
    CU_ASSERT(result.double_val == v1.double_val + v2.double_val);
}

void test_array_val_sub() {
    ArrayVal v1, v2;

    v1.int_val = 4;
    v2.int_val = 17;
    ArrayVal result = array_val_sub(v1, v2, DTYPE_INT);
    CU_ASSERT(result.int_val == v1.int_val - v2.int_val);
}

void test_array_val_mul() {
    ArrayVal v1, v2;

    v1.float_val = 0.4f;
    v2.float_val = 1.7f;
    ArrayVal result = array_val_mul(v1, v2, DTYPE_FLOAT);
    CU_ASSERT(result.float_val == v1.float_val * v2.float_val);
}

void test_array_val_div() {
    ArrayVal v1, v2;

    v1.long_val = 17;
    v2.long_val = 4;
    ArrayVal result = array_val_div(v1, v2, DTYPE_LONG);
    CU_ASSERT(result.long_val == v1.long_val / v2.long_val);
}

void test_array_val_neg() {
    ArrayVal v;

    v.int_val = 4;
    CU_ASSERT(array_val_neg(v, DTYPE_INT).int_val == -4);
    v.float_val = 4.0f;
    CU_ASSERT(array_val_neg(v, DTYPE_FLOAT).float_val == -4.0f);
    v.double_val = 4.0;
    CU_ASSERT(array_val_neg(v, DTYPE_DOUBLE).double_val == -4.0);
    v.long_val = 4;
    CU_ASSERT(array_val_neg(v, DTYPE_LONG).long_val == -4);
}

void test_array_val_equal() {
    ArrayVal v1, v2;
    v1.float_val = 1.0f;
    v2.float_val = 1.0f;
    CU_ASSERT(array_val_equal(v1, v2, DTYPE_FLOAT));

    v2.float_val = 3.7f;
    CU_ASSERT(!array_val_equal(v1, v2, DTYPE_FLOAT));
}
