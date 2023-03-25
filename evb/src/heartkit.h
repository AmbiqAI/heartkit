/**
 * @file heartkit.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __HEARTKIT_H
#define __HEARTKIT_H

enum HeartRhythm { HeartRhythmNormal, HeartRhythmAfib, HeartRhythmAfut };
typedef enum HeartRhythm HeartRhythm;

enum HeartBeat { HeartBeatNormal, HeartBeatPac, HeartBeatPvc };
typedef enum HeartBeat HeartBeat;

enum HeartRate { HeartRateNormal, HeartRateTachycardia, HeartRateBradycardia };
typedef enum HeartRate HeartRate;

enum HeartSegment { HeartSegmentNormal, HeartSegmentPWave, HeartSegmentQrs, HeartSegmentTWave };
typedef enum HeartSegment HeartSegment;

// const char *heart_rhythm_labels[] = {"NSR", "AFIB/AFL"};
// const char *heart_beat_labels[] = { "NORMAL", "PAC", "PVC" };
// const char *hear_rate_labels[] = { "NORMAL", "TACHYCARDIA", "BRADYCARDIA" };
// const char *heart_seg_labels[] = { "NONE", "P-WAVE", "QRS", "T-WAVE" };

int
init_heartkit();
void
hk_preprocess(float32_t *data);
int
hk_run(float32_t *data, int32_t *segMask, int32_t *results);

#endif // __HEARTKIT_H
