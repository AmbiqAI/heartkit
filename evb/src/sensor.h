#ifndef __SENSOR_H
#define __SENSOR_H

#include <stdint.h>
#include "arm_math.h"

void init_sensor(void);
void start_sensor(void);
uint32_t capture_sensor_data(float32_t* buffer);
void stop_sensor(void);


#endif // __SENSOR_H
