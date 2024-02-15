#ifndef __HK_MODEL_H
#define __HK_MODEL_H

#include "tensorflow/lite/micro/micro_common.h"
#include <stdint.h>

/**
 * @brief Initialize the model
 *
 * @return uint32_t
 */
uint32_t
model_init();

/**
 * @brief Setup the model
 *
 * @param modelBuffer Pointer to the model buffer
 * @param inputs Pointer to the input tensor
 * @param outputs Pointer to the output tensor
 * @return uint32_t
 */
uint32_t
model_setup(const void *modelBuffer, TfLiteTensor *inputs, TfLiteTensor *outputs);

/**
 * @brief Run the model
 *
 * @return uint32_t
 */
uint32_t
model_run();

#endif // __HK_MODEL_H
