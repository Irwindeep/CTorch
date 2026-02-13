#include "ctorch.h"
#include "random.h"
#include "tensor.h"

#include <omp.h>
#include <stdint.h>
#include <time.h>

PRNG *global_rng;

void CTorchInit() {
    global_rng = rng_init(time(NULL));

    int ncores = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(ncores);
}

void ManualSeed(uint64_t seed) {
    free_rng(global_rng);
    global_rng = rng_init(seed);
}

void CTorchClose() { free_rng(global_rng); };

void auto_free_env(Environment **env) {
    if (*env == NULL)
        return;

    free_env(*env);
}
