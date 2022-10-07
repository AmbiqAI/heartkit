#ifndef __PREPROCESSING_H
#define __PREPROCESSING_H

#include "arm_math.h"

void init_preprocess(void);
int standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
int bandpass_filter(float32_t* pSrc, float32_t *pResult, uint32_t blockSize);
int preprocess(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);


#endif // __PREPROCESSING_H
