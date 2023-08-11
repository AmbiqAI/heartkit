/**
 * @file sensor.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Initializes and collects sensor data from MAX86150
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __SENSOR_H
#define __SENSOR_H

#include "arm_math.h"
#include "max86150_addons.h"
#include "ns_max86150_driver.h"
#include <stdint.h>

typedef struct {
    max86150_context_t *maxCtx;
    max86150_config_t *maxCfg;
} hk_sensor_t;

uint32_t
init_sensor(hk_sensor_t *ctx);
void
start_sensor(hk_sensor_t *ctx);
uint32_t
capture_sensor_data(hk_sensor_t *ctx, float32_t *slot0, float32_t *slot1, float32_t *slot2, float32_t *slot3, uint32_t maxSamples,
                    uint32_t *numSamples);
void
stop_sensor(hk_sensor_t *ctx);

#endif // __SENSOR_H
