#include "array.h"
#include "ops.h"

#include <cblas.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

static int num_threads = -1;

static inline int decide_threads(int M, int N, int K) {
    long flops = 2L * M * N * K;

    if (flops < 5e7)
        return 1;
    if (flops < 2e8)
        return 2;
    if (flops < 1e9)
        return 4;
    return 8;
}

static inline bool is_contig(size_t rows, size_t cols, size_t sR, size_t sC) {
    return (sR == cols && sC == 1);
}

void matmul_kernel(const ndArray *arr1, const ndArray *arr2, ndArray *result,
                   const size_t *idx1, const size_t *idx2, const size_t *idx) {
    size_t m = get_shape(result)[get_ndim(result) - 2],
           n = get_shape(result)[get_ndim(result) - 1],
           k = get_shape(arr1)[get_ndim(arr1) - 1];

    int threads = decide_threads(m, n, k);
    if (threads != num_threads) {
        omp_set_num_threads(threads);
        openblas_set_num_threads(threads);
        num_threads = threads;
    }

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

    CBLAS_TRANSPOSE transA = A_contig ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB = B_contig ? CblasNoTrans : CblasTrans;

    int lda = A_contig ? (int)k : (int)m;
    int ldb = B_contig ? (int)n : (int)k;

    switch (dtype) {
    default:
        break;
    case DTYPE_FLOAT: {
        const float *A = (float *)(get_array_data(arr1) + offsetA);
        const float *B = (float *)(get_array_data(arr2) + offsetB);
        float *C = (float *)(get_array_data(result) + offsetC);

        cblas_sgemm(CblasRowMajor, transA, transB, (int)m, (int)n, (int)k, 1.0f,
                    A, lda, B, ldb, 0.0f, C, (int)n);
    } break;
    case DTYPE_DOUBLE: {
        const double *A = (double *)(get_array_data(arr1) + offsetA);
        const double *B = (double *)(get_array_data(arr2) + offsetB);
        double *C = (double *)(get_array_data(result) + offsetC);

        cblas_dgemm(CblasRowMajor, transA, transB, (int)m, (int)n, (int)k, 1.0f,
                    A, lda, B, ldb, 0.0f, C, (int)n);
    } break;
    }
}
