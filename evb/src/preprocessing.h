/**
 * @file preprocessing.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __PREPROCESSING_H
#define __PREPROCESSING_H

#include "arm_math.h"

uint32_t
init_preprocess(void);
uint32_t
standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
bandpass_filter(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
resample_signal(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample);

#endif // __PREPROCESSING_H
