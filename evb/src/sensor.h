/**
 * @file sensor.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Initializes and collects sensor data from MAX86150
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __SENSOR_H
#define __SENSOR_H

#include <stdint.h>
#include "arm_math.h"

int init_sensor(void);
void start_sensor(void);
uint32_t capture_sensor_data(float32_t* buffer);
void stop_sensor(void);


#endif // __SENSOR_H
