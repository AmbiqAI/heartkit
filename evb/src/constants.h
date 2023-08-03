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

// Sensor block
#define I2C_IOM (1)
#define CAPTURE_SECS (10)
#define SENSOR_RATE (200)
#define SAMPLE_RATE (200)
#define MAX86150_ADDR (0x5E)
#define SENSOR_LEN (CAPTURE_SECS * SENSOR_RATE)

// Preprocess block
#define RPC_BUF_LEN (50)
#define NORM_STD_EPS (0.1)
#define ECG_SOS_LEN (3)
#define QRS_SOS_LEN (3)

// Model block
#define ARRHTYHMIA_ENABLE 1
#define ARR_MODEL_SIZE_KB (65)
#define ARR_FRAME_LEN (800)
#define ARR_THRESHOLD (0.75)

#define SEGMENTATION_ENABLE 1
#define SEG_MODEL_SIZE_KB (85)
#define SEG_FRAME_LEN (512)
#define SEG_OVERLAP_LEN (20)
#define SEG_STEP_SIZE (SEG_FRAME_LEN - 2 * SEG_OVERLAP_LEN)
#define SEG_THRESHOLD (0.70)

#define BEAT_ENABLE 1
#define BEAT_MODEL_SIZE_KB (60)
#define BEAT_FRAME_LEN (160)
#define BEAT_THRESHOLD (0.45)

#define DISPLAY_LEN_USEC (2000000)

#define HK_DATA_LEN (CAPTURE_SECS * SAMPLE_RATE)
#define HK_PEAK_LEN (10 * CAPTURE_SECS)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif // __HK_CONSTANTS_H
