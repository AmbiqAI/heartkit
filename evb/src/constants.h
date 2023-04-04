/**
 * @file constants.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Store global app constants
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __HK_CONSTANTS_H
#define __HK_CONSTANTS_H

// #define EMULATION (1)
#define SAMPLE_RATE (250)
#define MAX86150_ADDR (0x5E)

#define ARRHTYHMIA_ENABLE
#define SEGMENTATION_ENABLE
#define BEAT_ENABLE

#define DISPLAY_LEN_USEC (2000000)

#define HK_DATA_LEN (10 * SAMPLE_RATE)
#define HK_PEAK_LEN (120)
#define HK_ARR_LEN (1000)
#define HK_BEAT_LEN (120)
#define HK_SEG_LEN (624)
#define HK_SEG_OLP (25)
#define HK_SEG_STEP (HK_SEG_LEN - 2 * HK_SEG_OLP)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif // __HK_CONSTANTS_H
