/**
 * @file model.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Performs inference using TFLM
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __MODEL_H
#define __MODEL_H

#include "arm_math.h"

uint32_t
init_models(void);
int
arrhythmia_inference(float32_t *x, float32_t threshold);
int
segmentation_inference(float32_t *data, uint8_t *segMask, uint32_t padLen);
int
beat_inference(float32_t *pBeat, float32_t *beat, float32_t *nBeat);
#endif // __MODEL_H
