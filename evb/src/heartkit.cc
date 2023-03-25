
#include "arm_math.h"
#include <stdbool.h>

#include "constants.h"
#include "heartkit.h"
#include "model.h"
#include "preprocessing.h"

// static float32_t hkData[HK_DATA_LEN];
// static int32_t hkSegMask[HK_DATA_LEN];
static int32_t hkPeaks[HK_PEAK_LEN];
static int32_t hkRRIntervals[HK_PEAK_LEN];

int
init_heartkit() {
    // Initialize all three models
    // Initialize preprocess?
    int err = 0;
    err = init_preprocess();
    err = init_models();
    return err;
}

void
hk_preprocess(float32_t *data) {
    /**
     * @brief Preprocess by bandpass filtering and standardizing
     *
     */
    bandpass_filter(data, data, HK_DATA_LEN);
    standardize(data, data, HK_DATA_LEN);
}

int
find_peaks_from_segments(float32_t *data, int32_t *segMask, uint32_t dataLen, int32_t *peaks) {
    int numPeaks = 0;
    uint32_t qrsStartIdx = 0, qrsEndIdx = 0;
    for (size_t i = 1; i < dataLen; i++) {
        // QRS start case
        if (segMask[i] == HeartSegmentQrs && segMask[i - i] == HeartSegmentNormal) {
            qrsStartIdx = i - 1;
        }
        // QRS end case
        if (segMask[i - 1] == HeartSegmentQrs && segMask[i] == HeartSegmentNormal) {
            qrsEndIdx = i - 1;
            peaks[numPeaks++] = (qrsEndIdx + qrsStartIdx) >> 1;
        }
    }
    return numPeaks;
}

int
ecg_rate(int32_t *peaks, uint32_t dataLen, int32_t *rrIntervals) {
    // Make sure we have at least 2 peaks
    for (size_t i = 1; i < dataLen; i++) {
        rrIntervals[i] = peaks[i] - peaks[i - 1];
    }
    rrIntervals[0] = rrIntervals[1];
    return 0;
}

float32_t
ecg_bpm(int32_t *rrIntervals, uint32_t dataLen, float32_t sampling_rate, float32_t min_rate, float32_t max_rate) {
    float32_t bpm = 0;
    float32_t val;
    uint32_t bpmLen = 0;
    for (size_t i = 0; i < dataLen; i++) {
        val = rrIntervals[i] / sampling_rate;
        if (min_rate >= 0 && val < min_rate) {
            continue;
        }
        if (max_rate >= 0 && val > max_rate) {
            continue;
        }
        bpm += val;
        bpmLen += 1;
    }
    bpm = 60.0f * (bpm / bpmLen);
    return bpm;
}

int
hk_run(float32_t *data, int32_t *segMask, int32_t *results) {
    // Pre-process data

    // Apply arrhythmia model
    for (size_t i = 0; i < HK_DATA_LEN - HK_ARR_LEN + 1; i += HK_ARR_LEN) {
        arrhythmia_inference(&data[i], 0);
    }

    // Apply segmentatiom model
    // We dont predict on first and last overlap size so set to normal
    memset(&segMask[0], 0, HK_SEG_OLP);
    memset(&segMask[HK_DATA_LEN - HK_SEG_OLP], HeartSegmentNormal, HK_SEG_OLP);

    for (size_t i = 0; i < HK_DATA_LEN - HK_SEG_LEN + 1; i += HK_SEG_STEP) {
        segmentation_inference(&data[i], &segMask[i + HK_SEG_OLP], HK_SEG_OLP);
    }
    segmentation_inference(&data[HK_DATA_LEN - HK_SEG_LEN], &segMask[HK_DATA_LEN - HK_SEG_LEN], HK_SEG_OLP);

    // Apply HRV
    int numPeaks = find_peaks_from_segments(data, segMask, HK_DATA_LEN, hkPeaks);
    ecg_rate(hkPeaks, numPeaks, hkRRIntervals);
    float32_t bpm = ecg_bpm(hkRRIntervals, numPeaks, SAMPLE_RATE, -1, -1);
    uint32_t avgRR = (uint32_t)(SAMPLE_RATE / (bpm / 60));
    uint32_t rhythm = bpm < 60 ? HeartRateBradycardia : bpm <= 100 ? HeartRateNormal : HeartRateTachycardia;

    // Apply beat head
    uint32_t bIdx;
    uint32_t beatLabel;
    uint32_t numPac = 0;
    uint32_t numPvc = 0;
    for (int i = 0; i < numPeaks; i++) {
        bIdx = hkPeaks[i] - (HK_BEAT_LEN >> 1);
        if (bIdx < avgRR || bIdx + avgRR + HK_BEAT_LEN > HK_DATA_LEN) {
            beatLabel = HeartBeatNormal;
        } else {
            beatLabel = beat_inference(&data[bIdx - avgRR], &data[bIdx], &data[bIdx + avgRR]);
        }
        segMask[bIdx] += beatLabel;
        if (beatLabel == HeartBeatPac) {
            numPac += 1;
        } else if (beatLabel == HeartBeatPvc) {
            numPvc += 1;
        }
    }
    return 0;
}
