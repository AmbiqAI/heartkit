/**
 * @file constants.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Store global app constants
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __ECG_CONSTANTS_H
#define __ECG_CONSTANTS_H

// #define EMULATION (1)
#define SAMPLE_RATE (250)
#define MAX86150_ADDR (0x5E)
#define INF_WINDOW_LEN (SAMPLE_RATE*5) // 5 seconds
#define PAD_WINDOW_LEN (SAMPLE_RATE*1) // 1 seconds
#define COLLECT_LEN (INF_WINDOW_LEN+PAD_WINDOW_LEN)
#define SENSOR_BUFFER_LEN (COLLECT_LEN+SAMPLE_RATE)
#define NUM_CLASSES (2)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static const char *heart_rhythm_labels[] = { "normal", "afib", "aflut", "noise" };
// static const char *heart_beat_labels[] = { "normal", "pac", "aberrated", "pvc", "noise" };
// static const char *hear_rate_labels[] = { "normal", "tachycardia", "bradycardia", "noise" };

#endif // __ECG_CONSTANTS_H
