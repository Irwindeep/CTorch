#include "array_tests/array_tests.h"

#include <CUnit/Basic.h>
#include <CUnit/TestDB.h>

void ArrayUnitTests(CU_pSuite array_tests);

int main() {
    CU_initialize_registry();

    CU_pSuite array_tests = CU_add_suite("ArrayTestSuite", 0, 0);
    ArrayUnitTests(array_tests);

    CU_basic_set_mode(CU_BRM_NORMAL);
    CU_basic_run_tests();
    CU_cleanup_registry();

    return 0;
}

void ArrayUnitTests(CU_pSuite array_tests) {
    CU_add_test(array_tests, "Array Value One", test_array_val_one);
    CU_add_test(array_tests, "Array Value Zero", test_array_val_zero);
    CU_add_test(array_tests, "Array Value Addition", test_array_val_add);
    CU_add_test(array_tests, "Array Value Subtraction", test_array_val_sub);
    CU_add_test(array_tests, "Array Value Multiplication", test_array_val_mul);
    CU_add_test(array_tests, "Array Value Division", test_array_val_div);
    CU_add_test(array_tests, "Array Value Negation", test_array_val_neg);
    CU_add_test(array_tests, "Array Value Equal", test_array_val_equal);

    CU_add_test(array_tests, "Array Initialization", test_array_init);
    CU_add_test(array_tests, "Getters and Setters", test_getters_and_setters);
    CU_add_test(array_tests, "Populate Array", test_populate_array);
    CU_add_test(array_tests, "Identity Array", test_eye_array);
    CU_add_test(array_tests, "Zeros Array", test_zeroes_array);
    CU_add_test(array_tests, "Ones Array", test_ones_array);

    CU_add_test(array_tests, "Array Equality", test_array_equal);
    CU_add_test(array_tests, "Array Addition", test_array_add);
    CU_add_test(array_tests, "Array Subtraction", test_array_sub);
    CU_add_test(array_tests, "Array Multiplication", test_array_mul);
    CU_add_test(array_tests, "Array Division", test_array_div);
    CU_add_test(array_tests, "Array Matrix Multiplication", test_array_matmul);
    CU_add_test(array_tests, "Array Transposition", test_array_transpose);
    CU_add_test(array_tests, "Array Sum", test_array_sum);
    CU_add_test(array_tests, "Array Sum Across a Dimension",
                test_array_sum_dim);
}
