#ifndef CTORCH_H
#define CTORCH_H

#include <stdint.h>

void CTorchInit(void);
void ManualSeed(uint64_t seed);
void CTorchClose(void);

#if defined(__GNUC__) || defined(__clang__)
#define ScopedEnvironment __attribute__((cleanup(free_env))) Environment *
#define ScopedOptimizer __attribute__((cleanup(free_optim))) Optimizer *
#define ScopedModule __attribute__((cleanup(free_module))) Module *
#else
#define ScopedEnvironment Environment *
#define ScopedOptimizer Optimizer *
#define ScoScopedModule Module *
#endif

#endif // !CTORCH_H
