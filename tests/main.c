#include "array_tests/array_ops_tests.h"
#include "array_tests/array_tests.h"

#include <CUnit/Basic.h>
#include <CUnit/TestDB.h>

int main() {
    CU_initialize_registry();

    CU_pSuite array_tests = CU_add_suite("ArrayTestSuite", 0, 0);
    CU_add_test(array_tests, "Array Initialization", test_array_init);
    CU_add_test(array_tests, "Getters and Setters", test_getters_and_setters);
    CU_add_test(array_tests, "Populate Array", test_populate_array);
    CU_add_test(array_tests, "Identity Array", test_eye_array);
    CU_add_test(array_tests, "Zeros Array", test_zeroes_array);
    CU_add_test(array_tests, "Ones Array", test_ones_array);

    CU_pSuite array_ops = CU_add_suite("ArrayOperationsSuite", 0, 0);
    CU_add_test(array_ops, "Array Equality", test_array_equal);
    CU_add_test(array_ops, "Array Addition", test_array_add);
    CU_add_test(array_ops, "Array Subtraction", test_array_sub);
    CU_add_test(array_ops, "Array Multiplication", test_array_mul);
    CU_add_test(array_ops, "Array Matrix Multiplication", test_array_matmul);

    CU_basic_set_mode(CU_BRM_NORMAL);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}
