#ifndef ARRAY_H
#define ARRAY_H

#include <stdbool.h>
#include <stddef.h>

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
ArrayVal array_val_neg(ArrayVal value, DType dtype);
bool array_val_equal(ArrayVal v1, ArrayVal v2, DType dtype);

#define MAX_NDIM 32

typedef struct ndArray ndArray;

ndArray *array_init(int ndim, const size_t *shape, DType dtype);
void free_array(ndArray *array);

int get_ndim(const ndArray *array);
size_t get_itemsize(const ndArray *array);
size_t get_total_size(const ndArray *array);
DType get_dtype(const ndArray *array);
size_t *get_shape(const ndArray *array);
size_t *get_strides(const ndArray *array);
void *get_array_data(const ndArray *array);

ArrayVal get_value(const ndArray *array, const size_t *indices);
void set_value(ndArray *array, const size_t *indices, ArrayVal value);
void set_strides(ndArray *array, const size_t *strides);

void populate_array(ndArray *array, const void *data);
void offset_to_index(size_t offset, size_t *idx, const size_t *shape, int ndim);

size_t index_to_offset(const size_t *idx, const size_t *strides, int ndim);

ndArray *copy_array(const ndArray *array);
bool is_array_contiguous(const ndArray *array);

ndArray *eye(size_t m, size_t n, DType dtype);
ndArray *zeros(int ndim, const size_t *shape, DType dtype);
ndArray *ones(int ndim, const size_t *shape, DType dtype);

bool array_equal(const ndArray *arr1, const ndArray *arr2);

bool broadcastable(const size_t *shape1, const size_t *shape2, int ndim1,
                   int ndim2);

void broadcast_shape(const size_t *shape1, const size_t *shape2, size_t *shape,
                     int ndim1, int ndim2, int ndim);

void broadcasted_strides(size_t *strides, const size_t *src_strides,
                         const size_t *src_shape, int src_ndim,
                         const size_t *dst_shape, int dst_ndim);

void get_broadcasted_indices(const size_t *shape1, const size_t *shape2,
                             const size_t *shape, int ndim1, int ndim2,
                             int ndim, size_t *idx1, size_t *idx2, size_t *idx,
                             size_t batch);

ndArray *array_add(ndArray *arr1, ndArray *arr2);
ndArray *array_sub(ndArray *arr1, ndArray *arr2);
ndArray *array_mul(ndArray *arr1, ndArray *arr2);
ndArray *array_div(ndArray *arr1, ndArray *arr2);

ndArray *negative(ndArray *array);
ndArray *inverse(ndArray *array);
ndArray *matmul(ndArray *arr1, ndArray *arr2);

ndArray *transpose(ndArray *array, const int *dims);
ndArray *array_sum(ndArray *array);
ndArray *array_sum_dim(ndArray *array, int dim, bool keepdims);

void array_addi(ndArray **arr1, ndArray *arr2);
void array_subi(ndArray **arr1, ndArray *arr2);
void array_muli(ndArray **arr1, ndArray *arr2);
void array_divi(ndArray **arr1, ndArray *arr2);
void array_sumi(ndArray **array);
void array_sum_dimi(ndArray **array, int dim, bool keepdims);

void negativei(ndArray **array);
void inversei(ndArray **array);

ndArray *array_max(ndArray *arr1, ndArray *arr2);
ndArray *array_min(ndArray *arr1, ndArray *arr2);

ndArray *array_gt(ndArray *arr1, ndArray *arr2);
ndArray *array_ge(ndArray *arr1, ndArray *arr2);
ndArray *array_lt(ndArray *arr1, ndArray *arr2);
ndArray *array_le(ndArray *arr1, ndArray *arr2);
ndArray *array_eq(ndArray *arr1, ndArray *arr2);

#endif // !ARRAY_H
