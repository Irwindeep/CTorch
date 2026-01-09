#ifndef ARRAY_H
#define ARRAY_H

#include <stdbool.h>
#include <stddef.h>

#define ARRAY_INIT_FAILURE 1
#define INVALID_IDX 2
#define NON_BROADCASTABLE_ARRAYS 3
#define SHAPE_MISMATCH 4
#define INVALID_ARRAY 5
#define INVALID_DTYPE 6
#define REPEATED_ARRAY_DIMS 7
#define INVALID_DIM 8

typedef enum DType {
    DTYPE_INT,
    DTYPE_FLOAT,
    DTYPE_DOUBLE,
    DTYPE_LONG,
} DType;

extern const char *DTypeNames[];

typedef union ArrayVal {
    int int_val;
    float float_val;
    double double_val;
    long int long_val;
} ArrayVal;

#define FLOAT_EQ_TOL 1e-6
#define DOUBLE_EQ_TOL 1e-9

ArrayVal array_val_one(DType dtype);
ArrayVal array_val_zero(DType dtype);
ArrayVal array_val_add(ArrayVal v1, ArrayVal v2, DType dtype);
ArrayVal array_val_sub(ArrayVal v1, ArrayVal v2, DType dtype);
ArrayVal array_val_mul(ArrayVal v1, ArrayVal v2, DType dtype);
ArrayVal array_val_div(ArrayVal v1, ArrayVal v2, DType dtype);

bool array_val_equal(ArrayVal v1, ArrayVal v2, DType dtype);

typedef struct ndArray ndArray;

// array initialization
ndArray *array_init(int ndim, const size_t *shape, DType dtype);
void free_array(ndArray *array);

// getters and setters
int get_ndim(const ndArray *array);
size_t get_itemsize(const ndArray *array);
size_t get_total_size(const ndArray *array);
DType get_dtype(const ndArray *array);
size_t *get_shape(const ndArray *array);
size_t *get_strides(const ndArray *array);

ArrayVal get_value(const ndArray *array, const size_t *indices);
void set_value(ndArray *array, const size_t *indices, ArrayVal value);

void populate_array(ndArray *array, const void *data);
void offset_to_index(size_t offset, size_t *idx, const size_t *shape, int ndim);

// some important arrays
ndArray *eye(size_t m, size_t n, DType dtype);
ndArray *zeros(int ndim, const size_t *shape, DType dtype);
ndArray *ones(int ndim, const size_t *shape, DType dtype);

// array comparators
bool array_equal(ndArray *arr1, ndArray *arr2);

// array operations
bool broadcastable(const size_t *shape1, const size_t *shape2, int ndim1,
                   int ndim2);
size_t *broadcast_shape(const size_t *shape1, const size_t *shape2, int ndim1,
                        int ndim2);
ndArray *add(ndArray *arr1, ndArray *arr2);
ndArray *sub(ndArray *arr1, ndArray *arr2);
ndArray *mul(ndArray *arr1, ndArray *arr2);
ndArray *matmul(ndArray *arr1, ndArray *arr2);

void transpose(ndArray *array, const int *dims);
ndArray *array_sum(ndArray *array);
ndArray *array_sum_dim(ndArray *array, int dim, bool keepdims);

#endif // !ARRAY_H
