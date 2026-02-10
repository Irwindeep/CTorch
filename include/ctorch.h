#ifndef CTORCH_H
#define CTORCH_H

#include "tensor.h"
#include <stdint.h>

void CTorchInit();
void ManualSeed(uint64_t seed);
void CTorchClose();

void auto_free_env(Environment **env);
#define AutoEnvironment __attribute__((cleanup(auto_free_env))) Environment *

#endif // !CTORCH_H
