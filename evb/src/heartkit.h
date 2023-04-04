/**
 * @file heartkit.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __HEARTKIT_H
#define __HEARTKIT_H

typedef struct {
    uint32_t heartRate;
    uint32_t heartRhythm;
    uint32_t numNormBeats;
    uint32_t numPacBeats;
    uint32_t numPvcBeats;
    uint32_t arrhythmia;
} hk_result_t;

enum HeartRhythm { HeartRhythmNormal, HeartRhythmAfib, HeartRhythmAfut };
typedef enum HeartRhythm HeartRhythm;

enum HeartBeat { HeartBeatNormal, HeartBeatPac, HeartBeatPvc };
typedef enum HeartBeat HeartBeat;

enum HeartRate { HeartRateNormal, HeartRateTachycardia, HeartRateBradycardia };
typedef enum HeartRate HeartRate;

enum HeartSegment { HeartSegmentNormal, HeartSegmentPWave, HeartSegmentQrs, HeartSegmentTWave };
typedef enum HeartSegment HeartSegment;

extern const char *HK_RHYTHM_LABELS[3];
extern const char *HK_BEAT_LABELS[3];
extern const char *HK_HEART_RATE_LABELS[3];
extern const char *HK_SEGMENT_LABELS[4];

uint32_t
init_heartkit();
uint32_t
hk_preprocess(float32_t *data);
uint32_t
ecg_rate(int32_t *peaks, uint32_t dataLen, int32_t *rrIntervals);
uint32_t
find_peaks_from_segments(float32_t *data, uint8_t *segMask, uint32_t dataLen, int32_t *peaks);
uint32_t
hk_run(float32_t *data, uint8_t *segMask, hk_result_t *result);
uint32_t
hk_print_result(hk_result_t *result);

#endif // __HEARTKIT_H
