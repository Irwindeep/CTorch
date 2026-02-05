#ifndef CTORCH_H
#define CTORCH_H

#include <stdint.h>

void CTorchInit();
void ManualSeed(uint64_t seed);
void CTorchClose();

#endif // !CTORCH_H
