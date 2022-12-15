/**
 * @file preprocessing.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __PREPROCESSING_H
#define __PREPROCESSING_H

#include "arm_math.h"

int
init_preprocess(void);
int
standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
int
bandpass_filter(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);

#endif // __PREPROCESSING_H
