/**
 * @file max86150_addons.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief MAX86150 addons
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __MAX86150_ADDONS_H
#define __MAX86150_ADDONS_H

#include "ns_max86150_driver.h"

typedef struct {
    uint8_t numSlots;
    max86150_slot_type *fifoSlotConfigs;
    uint8_t fifoRolloverFlag;
    uint8_t ppgSampleAvg;
    uint8_t ppgAdcRange;
    uint8_t ppgSampleRate;
    uint8_t ppgPulseWidth;
    uint8_t led0CurrentRange;
    uint8_t led1CurrentRange;
    uint8_t led2CurrentRange;
    uint8_t led0PulseAmplitude;
    uint8_t led1PulseAmplitude;
    uint8_t led2PulseAmplitude;
    uint8_t ecgSampleRate;
    uint8_t ecgIaGain;
    uint8_t ecgPgaGain;
} max86150_config_t;

#endif // __MAX86150_ADDONS_H
