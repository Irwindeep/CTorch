#include "ctorch.h"
#include "random.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

PRNG       *global_rng;
const char *DTypeNames[] = {
    "DTYPE_INT",
    "DTYPE_FLOAT",
    "DTYPE_DOUBLE",
    "DTYPE_LONG",
};

void CTorchInit(void) { global_rng = rng_init((uint64_t)time(NULL)); }

void ManualSeed(uint64_t seed) {
    free_rng(global_rng);
    global_rng = rng_init(seed);
}

void CTorchClose(void) { free_rng(global_rng); }

void ct_exit(int status) { exit(status); }
