
#include "arm_math.h"
#include <stdbool.h>

#include "ns_ambiqsuite_harness.h"

#include "constants.h"
#include "heartkit.h"
#include "model.h"
#include "preprocessing.h"

static int32_t hkPeaks[HK_PEAK_LEN];
static int32_t hkRRIntervals[HK_PEAK_LEN];

const char *HK_RHYTHM_LABELS[] = {"NSR", "AFIB/AFL", "AFIB/AFL"};
const char *HK_BEAT_LABELS[] = {"NORMAL", "PAC", "PVC"};
const char *HK_HEART_RATE_LABELS[] = {"NORMAL", "TACHYCARDIA", "BRADYCARDIA"};
const char *HK_SEGMENT_LABELS[] = {"NONE", "P-WAVE", "QRS", "T-WAVE"};

uint32_t
init_heartkit() {
    uint32_t err = 0;
    err = init_preprocess();
    err |= init_models();
    return err;
}

uint32_t
hk_preprocess(float32_t *data) {
    /**
     * @brief Preprocess by bandpass filtering and standardizing
     *
     */
    uint32_t err = 0;
    err = bandpass_filter(data, data, HK_DATA_LEN);
    err |= standardize(data, data, HK_DATA_LEN);
    return err;
}

uint32_t
find_peaks_from_segments(float32_t *data, uint8_t *segMask, uint32_t dataLen, int32_t *peaks) {
    uint32_t numPeaks = 0;
    uint32_t qrsStartIdx = 0, qrsEndIdx = 0, qrsLen = 0;
    for (size_t i = 1; i < dataLen; i++) {
        // QRS start case
        if (segMask[i] == HeartSegmentQrs && segMask[i - i] == HeartSegmentNormal) {
            qrsStartIdx = i - 1;
        }
        // QRS end case
        if (segMask[i - 1] == HeartSegmentQrs && segMask[i] == HeartSegmentNormal) {
            qrsEndIdx = i - 1;
            qrsLen = qrsEndIdx - qrsStartIdx + 1;
            peaks[numPeaks++] = (qrsEndIdx + qrsStartIdx) >> 1;
            // QRS width must be within limits and # QRS points must be at least 70%

            // Avoid compiler warning
            (void)qrsLen;
        }
    }
    return numPeaks;
}

uint32_t
ecg_rate(int32_t *peaks, uint32_t dataLen, int32_t *rrIntervals) {
    // Make sure we have at least 2 peaks
    for (size_t i = 1; i < dataLen; i++) {
        rrIntervals[i] = peaks[i] - peaks[i - 1];
        // ns_printf("rrIntervals[%lu] = %d\n", i, rrIntervals[i]);
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
    bpm = 60.0f / (bpm / bpmLen);
    return bpm;
}

uint32_t
hk_run(float32_t *data, uint8_t *segMask, hk_result_t *result) {
    uint32_t err = 0;
    int val = 0;

    result->heartRate = 0;
    result->heartRhythm = HeartRateNormal;
    result->numPacBeats = 0;
    result->numPvcBeats = 0;
    result->numNormBeats = 0;
    result->arrhythmia = HeartRhythmNormal;

    // Apply arrhythmia model
    for (size_t i = 0; i < HK_DATA_LEN - HK_ARR_LEN + 1; i += HK_ARR_LEN) {
        val = arrhythmia_inference(&data[i], 0);
        if (val == -1) {
            err = 1;
        } else if (val == HeartRhythmAfib || val == HeartRhythmAfut) {
            result->arrhythmia = HeartRhythmAfib;
        }
    }

    // Apply segmentatiom model
    // We dont predict on first and last overlap size so set to normal
    memset(segMask, HeartSegmentNormal, HK_DATA_LEN);

    for (size_t i = 0; i < HK_DATA_LEN - HK_SEG_LEN + 1; i += HK_SEG_STEP) {
        val = segmentation_inference(&data[i], &segMask[i], HK_SEG_OLP);
        if (val == -1) {
            err = 1;
        }
    }
    val = segmentation_inference(&data[HK_DATA_LEN - HK_SEG_LEN], &segMask[HK_DATA_LEN - HK_SEG_LEN], HK_SEG_OLP);
    if (val == -1) {
        err = 1;
    }

    // Apply HRV
    int numPeaks = find_peaks_from_segments(data, segMask, HK_DATA_LEN, hkPeaks);
    ecg_rate(hkPeaks, numPeaks, hkRRIntervals);
    float32_t bpm = ecg_bpm(hkRRIntervals, numPeaks, SAMPLE_RATE, -1, -1);
    uint32_t avgRR = (uint32_t)(SAMPLE_RATE / (bpm / 60));
    result->heartRhythm = bpm < 60 ? HeartRateBradycardia : bpm <= 100 ? HeartRateNormal : HeartRateTachycardia;
    result->heartRate = (uint32_t)bpm;
    ns_printf("avgRR=%lu\n", avgRR);

    // Apply beat head
    uint32_t bIdx;
    uint32_t beatLabel;

    result->numPacBeats = 0;
    result->numPvcBeats = 0;
    result->numNormBeats = 0;

    uint32_t bOffset = (HK_BEAT_LEN >> 1);
    uint32_t bStart = 0;
    for (int i = 1; i < numPeaks - 1; i++) {
        bIdx = hkPeaks[i];
        bStart = bIdx - bOffset;
        if (bIdx < bOffset || bStart < avgRR || bStart + avgRR + HK_BEAT_LEN > HK_DATA_LEN) {
            beatLabel = HeartBeatNormal;
        } else {
            ns_printf("beats (%lu, %lu, %lu)\n", bStart - avgRR, bStart, bStart + avgRR);
            beatLabel = beat_inference(&data[bStart - avgRR], &data[bStart], &data[bStart + avgRR]);
        }
        // Place beat label in upper nibble
        segMask[bIdx] |= ((beatLabel + 1) << 4);
        if (beatLabel == HeartBeatPac) {
            result->numPacBeats += 1;
        } else if (beatLabel == HeartBeatPvc) {
            result->numPvcBeats += 1;
        } else {
            result->numNormBeats += 1;
        }
    }
    return err;
}

uint32_t
hk_print_result(hk_result_t *result) {
    uint32_t numBeats = result->numNormBeats + result->numPacBeats + result->numPvcBeats;
    const char *rhythm = HK_HEART_RATE_LABELS[result->heartRhythm];
    ns_printf("----------------------\n");
    ns_printf("** HeartKit Results **\n");
    ns_printf("----------------------\n");
    ns_printf("  Heart Rate: %lu\n", result->heartRate);
    ns_printf("Heart Rhythm: %s\n", rhythm);
    ns_printf(" Total Beats: %lu\n", numBeats);
    ns_printf("  Norm Beats: %lu\n", result->numNormBeats);
    ns_printf("   PAC Beats: %lu\n", result->numPacBeats);
    ns_printf("   PVC Beats: %lu\n", result->numPvcBeats);
    ns_printf("  Arrhythmia: %lu\n", result->arrhythmia);
    ns_printf("----------------------\n");
    return 0;
}
