#include "random.h"
#include "array.h"
#include "tensor.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static inline double prng_uniform_d(PRNG *rng) {
    return (rng_rand(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static inline float prng_uniform_f(PRNG *rng) {
    return (rng_rand(rng) >> 40) * (1.0f / 16777216.0f);
}

Tensor *uniform(PRNG *rng, int ndim, const size_t *shape, DType dtype,
                bool requires_grad, Environment *environ) {
    if (dtype != DTYPE_DOUBLE && dtype != DTYPE_FLOAT) {
        printf("Invalid dtype for uniform tensor\n");
        exit(INVALID_DTYPE);
    }

    ndArray *data = array_init(ndim, shape, dtype);
    size_t total_size = get_total_size(data);

    switch (dtype) {
    default:
        break;
    case DTYPE_DOUBLE: {
        double *arr_data = malloc(total_size * sizeof(double));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = prng_uniform_d(rng);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    case DTYPE_FLOAT: {
        float *arr_data = malloc(total_size * sizeof(float));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = prng_uniform_f(rng);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    }

    Tensor *tensor = tensor_init(data, requires_grad, environ);
    return tensor;
}

static inline double prng_randn_d(PRNG *rng) {
    static int has_spare_d = 0;
    static double spare_d;

    if (has_spare_d) {
        has_spare_d = 0;
        return spare_d;
    }

    double u, v, s;
    do {
        u = prng_uniform_d(rng) * 2.0 - 1.0;
        v = prng_uniform_d(rng) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare_d = v * s;
    has_spare_d = 1;
    return u * s;
}

static inline float prng_randn_f(PRNG *rng) {
    static int has_spare_f = 0;
    static double spare_f;

    if (has_spare_f) {
        has_spare_f = 0;
        return spare_f;
    }

    double u, v, s;
    do {
        u = prng_uniform_d(rng) * 2.0 - 1.0;
        v = prng_uniform_d(rng) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare_f = v * s;
    has_spare_f = 1;
    return u * s;
}

Tensor *randn(PRNG *rng, int ndim, const size_t *shape, DType dtype,
              bool requires_grad, Environment *environ) {
    if (dtype != DTYPE_DOUBLE && dtype != DTYPE_FLOAT) {
        printf("Invalid dtype for randn tensor\n");
        exit(INVALID_DTYPE);
    }

    ndArray *data = array_init(ndim, shape, dtype);
    size_t total_size = get_total_size(data);

    switch (dtype) {
    default:
        break;
    case DTYPE_DOUBLE: {
        double *arr_data = malloc(total_size * sizeof(double));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = prng_randn_d(rng);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    case DTYPE_FLOAT: {
        float *arr_data = malloc(total_size * sizeof(float));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = prng_randn_f(rng);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    }

    Tensor *tensor = tensor_init(data, requires_grad, environ);
    return tensor;
}

static inline uint64_t pcg64_bounded(PRNG *rng, uint64_t bound) {
    if (bound == 0)
        return 0;

    uint64_t threshold = (UINT64_MAX - bound + 1) % bound;
    for (;;) {
        uint64_t r = rng_rand(rng);
        if (r >= threshold)
            return r % bound;
    }
}

static inline int pcg64_randint_i(PRNG *rng, int low, int high) {
    if (low > high) {
        int tmp = low;
        low = high;
        high = tmp;
    }

    if (low == high)
        return low;

    int64_t range = (int64_t)high - (int64_t)low;

    uint64_t r = pcg64_bounded(rng, (uint64_t)range);
    return low + (int)r;
}

static inline long pcg64_randint_l(PRNG *rng, long low, long high) {
    if (low > high) {
        long tmp = low;
        low = high;
        high = tmp;
    }

    if (low == high)
        return low;

    int64_t range = (int64_t)high - (int64_t)low;

    uint64_t r = pcg64_bounded(rng, (uint64_t)range);
    return low + (long)r;
}

Tensor *randint(PRNG *rng, int ndim, const size_t *shape, long int low,
                long int high, DType dtype, Environment *environ) {
    if (dtype != DTYPE_INT && dtype != DTYPE_LONG) {
        printf("Invalid dtype for randint tensor\n");
        exit(INVALID_DTYPE);
    }

    ndArray *data = array_init(ndim, shape, dtype);
    size_t total_size = get_total_size(data);

    switch (dtype) {
    default:
        break;
    case DTYPE_INT: {
        int *arr_data = malloc(total_size * sizeof(int));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = pcg64_randint_i(rng, low, high);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    case DTYPE_LONG: {
        long int *arr_data = malloc(total_size * sizeof(long int));
        if (!arr_data) {
            printf("Array Initialization failed in uniform\n");
            break;
        }
        for (size_t i = 0; i < total_size; i++)
            arr_data[i] = pcg64_randint_l(rng, low, high);

        populate_array(data, arr_data);
        free(arr_data);
        break;
    }
    }

    Tensor *tensor = tensor_init(data, false, environ);
    return tensor;
}
