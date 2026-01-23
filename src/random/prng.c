#include "random.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// PCG64 pseudo random number
struct PRNG {
    __uint128_t state;
    __uint128_t inc; // must be odd
};

static inline void pcg64_seed(PRNG *rng, __uint128_t initstate,
                              __uint128_t initseq) {
    rng->state = 0;
    rng->inc = (initseq << 1u) | 1u;
    (void)rng_rand(rng);
    rng->state += initstate;
    (void)rng_rand(rng);
}

static inline void prng_seed(PRNG *rng, uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;

    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    uint64_t lo = z ^ (z >> 27);

    z = (z ^ (z >> 30)) * 0x94d049bb133111ebULL;
    uint64_t hi = z ^ (z >> 31);

    __uint128_t initstate = ((__uint128_t)hi << 64) | lo;
    __uint128_t initseq = ((__uint128_t)lo << 64) | hi;

    pcg64_seed(rng, initstate, initseq);
}

PRNG *rng_init(uint64_t seed) {
    PRNG *rng = malloc(sizeof(PRNG));
    if (!rng) {
        printf("Failure to allocate PRNG\n");
        exit(PRNG_INIT_FAILURE);
    }
    prng_seed(rng, seed);

    return rng;
}

uint64_t rng_rand(PRNG *rng) {
    __uint128_t old = rng->state;

    /* LCG step */
    rng->state = old * (((__uint128_t)6364136223846793005ULL << 64) |
                        1442695040888963407ULL) +
                 rng->inc;

    /* XSL RR output transform */
    uint64_t xorshifted = (uint64_t)(((old >> 64) ^ old) >> 64);

    uint64_t rot = old >> 122;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 63));
}

void free_rng(PRNG *rng) { free(rng); }
