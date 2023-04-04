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
#include <stdint.h>

uint32_t
init_sensor(void);
void
start_sensor(void);
uint32_t
capture_sensor_data(float32_t *buffer);
void
stop_sensor(void);

#endif // __SENSOR_H
