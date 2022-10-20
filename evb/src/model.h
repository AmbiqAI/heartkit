#ifndef __MODEL_H
#define __MODEL_H

#include "arm_math.h"

int init_model(void);
int model_inference(float32_t *x, float32_t *y);

#endif // __MODEL_H
