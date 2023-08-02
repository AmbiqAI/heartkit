/**
 * @file physiokit.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Pre-processing of physiological signals in ambulatory settings
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __PHYSIOKIT_H
#define __PHYSIOKIT_H

#include "arm_math.h"

typedef struct {
    arm_biquad_casd_df1_inst_f32 *inst;
    uint8_t numSecs;
    const float32_t *sos;
    float32_t *state;
} biquad_filt_f32_t;

typedef struct {
    float32_t peakWin;      // 0.111
    float32_t beatWin;      // 0.667
    float32_t beatOffset;   // 0.02
    float32_t peakDelayWin; // 0.3
    uint32_t sampleRate;
    // State requires 4*ppgLen
    float32_t *state;
    uint32_t *peaks;
} ppg_peak_f32_t;

typedef struct {
    float32_t qrsWin;          // 0.1
    float32_t avgWin;          // 1.0
    float32_t qrsPromWeight;   // 1.5
    float32_t qrsMinLenWeight; // 0.4
    float32_t qrsDelayWin;     // 0.3
    uint32_t sampleRate;
    // State requires 3*ecgLen
    float32_t *state;
} ecg_peak_f32_t;

typedef struct {
    // Deviation-based
    float32_t meanNN;
    float32_t sdNN;
    // Difference-based
    float32_t rmsSD;
    float32_t sdSD;
    // Normalized
    float32_t cvNN;
    float32_t cvSD;
    // Robust
    float32_t medianNN;
    float32_t madNN;
    float32_t mcvNN;
    float32_t iqrNN;
    float32_t prc20NN;
    float32_t prc80NN;

    // Extrema
    uint32_t nn50;
    uint32_t nn20;
    float32_t pnn50;
    float32_t pnn20;
    float32_t minNN;
    float32_t maxNN;
} hrv_td_metrics_t;

uint32_t
pk_mean(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
pk_std(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
pk_gradient(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
pk_rms(float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
pk_standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t epsilon);
uint32_t
pk_resample_signal(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample);
uint32_t
pk_linear_downsample(float32_t *pSrc, uint32_t srcSize, uint32_t srcFs, float32_t *pRst, uint32_t rstSize, uint32_t rstFs);
uint32_t
pk_init_biquad_filter(arm_biquad_casd_df1_inst_f32 *ctx);
uint32_t
pk_apply_biquad_filter(arm_biquad_casd_df1_inst_f32 *ctx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize);
uint32_t
pk_apply_biquad_filtfilt(arm_biquad_casd_df1_inst_f32 *ctx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *state);
uint32_t
pk_quotient_filter_mask(uint32_t *data, uint8_t *mask, uint32_t dataLen, uint32_t iterations, float32_t lowcut, float32_t highcut);
uint32_t
smooth_signal(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *wBuffer, uint32_t windowSize);
uint32_t
pk_ppg_find_peaks(ppg_peak_f32_t *ctx, float32_t *ppg, uint32_t ppgLen, uint32_t *peaks);
uint32_t
pk_ecg_find_peaks(ecg_peak_f32_t *ctx, float32_t *ecgg, uint32_t ecgLen, uint32_t *peaks);
float32_t
pk_compute_spo2_from_perfusion(float32_t dc1, float32_t ac1, float32_t dc2, float32_t ac2, float32_t *coefs);
float32_t
pk_compute_spo2_in_time(float32_t *ppg1, float32_t *ppg2, float32_t ppg1Mean, float32_t ppg2Mean, uint32_t blockSize, float32_t *coefs,
                        float32_t sampleRate);
uint32_t
pk_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals);
uint32_t
pk_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate);
uint32_t
pk_compute_hrv_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_td_metrics_t *metrics);

#endif // __PHYSIOKIT_H
