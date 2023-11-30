#ifndef __HK_MODEL_H
#define __HK_MODEL_H

#include "tensorflow/lite/micro/micro_common.h"
#include <stdint.h>

uint32_t
init_model();
uint32_t
setup_model(const void *modelBuffer, TfLiteTensor *inputs, TfLiteTensor *outputs);
uint32_t
run_model();

#endif // __HK_MODEL_H
