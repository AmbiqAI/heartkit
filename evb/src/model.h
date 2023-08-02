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
#ifndef __HK_MODEL_H
#define __HK_MODEL_H

#include "arm_math.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

typedef struct {
    int arenaSize;
    uint8_t *arena;
    const unsigned char *buffer;
    const tflite::Model *model;
    TfLiteTensor *input;
    TfLiteTensor *output;
    tflite::MicroInterpreter *interpreter;
} tf_model_config_t;

uint32_t
init_models(void);
uint32_t
arrhythmia_inference(float32_t *x, float32_t *yVal, uint32_t *yIdx);
uint32_t
segmentation_inference(float32_t *data, uint8_t *segMask, uint32_t padLen);
uint32_t
beat_inference(float32_t *pBeat, float32_t *beat, float32_t *nBeat, float32_t *yVal, uint32_t *yIdx);

#endif // __HK_MODEL_H
