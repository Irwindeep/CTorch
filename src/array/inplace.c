#include "array.h"

void array_addi(ndArray **arr1, ndArray *arr2) {
    ndArray *tmp = *arr1;
    *arr1 = array_add(tmp, arr2);
    free_array(tmp);
}

void array_subi(ndArray **arr1, ndArray *arr2) {
    ndArray *tmp = *arr1;
    *arr1 = array_sub(tmp, arr2);
    free_array(tmp);
}

void array_muli(ndArray **arr1, ndArray *arr2) {
    ndArray *tmp = *arr1;
    *arr1 = array_mul(tmp, arr2);
    free_array(tmp);
}

void array_divi(ndArray **arr1, ndArray *arr2) {
    ndArray *tmp = *arr1;
    *arr1 = array_div(tmp, arr2);
    free_array(tmp);
}

void array_sumi(ndArray **array) {
    ndArray *tmp = *array;
    *array = array_sum(tmp);
    free_array(tmp);
}

void array_sum_dimi(ndArray **array, int dim, bool keepdims) {
    ndArray *tmp = *array;
    *array = array_sum_dim(tmp, dim, keepdims);
    free_array(tmp);
}

void negativei(ndArray **array) {
    ndArray *tmp = *array;
    *array = negative(tmp);
    free_array(tmp);
}

void inversei(ndArray **array) {
    ndArray *tmp = *array;
    *array = inverse(tmp);
    free_array(tmp);
}
