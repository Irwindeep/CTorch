#include "ctorch.h"
#include "random.h"
#include <stdint.h>
#include <time.h>

PRNG *global_rng;

void CTorchInit() { global_rng = rng_init(time(NULL)); }

void ManualSeed(uint64_t seed) {
    free_rng(global_rng);
    global_rng = rng_init(seed);
}

void CTorchClose() { free_rng(global_rng); };
