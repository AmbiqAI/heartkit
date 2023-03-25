/**
 * @file model.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Performs inference using TFLM
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __MODEL_H
#define __MODEL_H

#include "arm_math.h"

int
init_models(void);
int
arrhythmia_inference(float32_t *x, float32_t threshold);
int
segmentation_inference(float32_t *data, int32_t *segMask, uint32_t padLen);
int
beat_inference(float32_t *pBeat, float32_t *beat, float32_t *nBeat);
#endif // __MODEL_H
