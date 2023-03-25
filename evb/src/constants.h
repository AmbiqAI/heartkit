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
#define INF_WINDOW_LEN (SAMPLE_RATE * 10) // 5 seconds
#define PAD_WINDOW_LEN (SAMPLE_RATE * 1)  // 1 seconds
#define COLLECT_LEN (INF_WINDOW_LEN + PAD_WINDOW_LEN)
#define SENSOR_BUFFER_LEN (COLLECT_LEN + SAMPLE_RATE)

#define NUM_CLASSES (2)

#define HK_DATA_LEN (SAMPLE_RATE * 10) // 10 seconds
#define HK_PEAK_LEN (120)
#define HK_ARR_LEN (1000)
#define HK_SEG_LEN (624)
#define HK_SEG_OLP (25)
#define HK_SEG_STEP (HK_SEG_LEN - 2 * HK_SEG_OLP)
#define HK_BEAT_LEN (120)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// const char *heart_rhythm_labels[] = {"NSR", "AFIB/AFL"};
// const char *heart_beat_labels[] = { "NORMAL", "PAC", "PVC" };
// const char *hear_rate_labels[] = { "NORMAL", "TACHYCARDIA", "BRADYCARDIA" };
// const char *heart_seg_labels[] = { "NONE", "P-WAVE", "QRS", "T-WAVE" };

#endif // __ECG_CONSTANTS_H
