#include "array.h"
#include "ops.h"

#include <cblas.h>
#include <stddef.h>
#include <stdlib.h>

static inline bool is_contig(size_t rows, size_t cols, size_t sR, size_t sC) {
    return (sR == cols && sC == 1);
}

static void contig_array_f(const float *src, float *dst, size_t rows,
                           size_t cols, size_t sR, size_t sC) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[i * cols + j] = src[i * sR + j * sC];
        }
    }
}

static void contig_array_d(const double *src, double *dst, size_t rows,
                           size_t cols, size_t sR, size_t sC) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[i * cols + j] = src[i * sR + j * sC];
        }
    }
}

void matmul_kernel(const ndArray *arr1, const ndArray *arr2, ndArray *result,
                   const size_t *idx1, const size_t *idx2, const size_t *idx) {
    size_t m = get_shape(result)[get_ndim(result) - 2],
           n = get_shape(result)[get_ndim(result) - 1],
           k = get_shape(arr1)[get_ndim(arr1) - 1];

    DType dtype = get_dtype(result);
    int ndim1 = get_ndim(arr1), ndim2 = get_ndim(arr2), ndim = get_ndim(result);

    size_t sAr = get_strides(arr1)[ndim1 - 2] / get_itemsize(arr1),
           sAc = get_strides(arr1)[ndim1 - 1] / get_itemsize(arr1);
    size_t sBr = get_strides(arr2)[ndim2 - 2] / get_itemsize(arr2),
           sBc = get_strides(arr2)[ndim2 - 1] / get_itemsize(arr2);
    size_t sCr = get_strides(result)[ndim - 2] / get_itemsize(result),
           sCc = get_strides(result)[ndim - 1] / get_itemsize(result);

    size_t offsetA = index_to_offset(idx1, get_strides(arr1), ndim1 - 2),
           offsetB = index_to_offset(idx2, get_strides(arr2), ndim2 - 2),
           offsetC = index_to_offset(idx, get_strides(result), ndim - 2);

    bool A_contig = is_contig(m, k, sAr, sAc),
         B_contig = is_contig(k, n, sBr, sBc);

    switch (dtype) {
    default:
        break;
    case DTYPE_FLOAT: {
        const float *A0 = (float *)(get_array_data(arr1) + offsetA);
        const float *B0 = (float *)(get_array_data(arr2) + offsetB);
        float *C = (float *)(get_array_data(result) + offsetC);

        float *A = (float *)A0;
        float *B = (float *)B0;

        if (!A_contig) {
            A = malloc(sizeof(float) * m * k);
            contig_array_f(A0, A, m, k, sAr, sAc);
        }

        if (!B_contig) {
            B = malloc(sizeof(float) * k * n);
            contig_array_f(B0, B, k, n, sBr, sBc);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n,
                    (int)k, 1.0f, A, (int)k, B, (int)n, 0.0f, C, (int)n);

        if (!A_contig)
            free(A);
        if (!B_contig)
            free(B);
    } break;
    case DTYPE_DOUBLE: {
        const double *A0 = (double *)(get_array_data(arr1) + offsetA);
        const double *B0 = (double *)(get_array_data(arr2) + offsetB);
        double *C = (double *)(get_array_data(result) + offsetC);

        double *A = (double *)A0;
        double *B = (double *)B0;

        if (!A_contig) {
            A = malloc(sizeof(double) * m * k);
            contig_array_d(A0, A, m, k, sAr, sAc);
        }

        if (!B_contig) {
            B = malloc(sizeof(double) * k * n);
            contig_array_d(B0, B, k, n, sBr, sBc);
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n,
                    (int)k, 1.0f, A, (int)k, B, (int)n, 0.0f, C, (int)n);

        if (!A_contig)
            free(A);
        if (!B_contig)
            free(B);
    } break;
    }
}
