#ifndef __MODEL_H
#define __MODEL_H

#include "arm_math.h"

void init_model(void);
int model_inference(float32_t *x, uint32_t *y);

#endif // __MODEL_H
