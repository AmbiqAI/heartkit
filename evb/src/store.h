
/**
 * @file store.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Act as central store for app
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __HK_STORE_H
#define __HK_STORE_H

#include "arm_math.h"

// neuralSPOT
#include "ns_ambiqsuite_harness.h"
#include "ns_i2c.h"
#include "ns_malloc.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_rpc_generic_data.h"
#include "ns_usb.h"

#include "constants.h"
#include "physiokit.h"
#include "sensor.h"
#include "usb_handler.h"

typedef struct {
    uint32_t heartRate;
    uint32_t heartRhythm;
    uint32_t numNormBeats;
    uint32_t numPacBeats;
    uint32_t numPvcBeats;
    uint32_t numNoiseBeats;
    uint32_t arrhythmia;
} hk_result_t;

enum HeartRhythm { HeartRhythmNormal, HeartRhythmAfib, HeartRhythmAfut };
typedef enum HeartRhythm HeartRhythm;

enum HeartBeat { HeartBeatNormal, HeartBeatPac, HeartBeatPvc, HeartBeatNoise };
typedef enum HeartBeat HeartBeat;

enum HeartRate { HeartRateNormal, HeartRateTachycardia, HeartRateBradycardia };
typedef enum HeartRate HeartRate;

enum HeartSegment { HeartSegmentNormal, HeartSegmentPWave, HeartSegmentQrs, HeartSegmentTWave };
typedef enum HeartSegment HeartSegment;

enum AppState {
    IDLE_STATE,
    START_COLLECT_STATE,
    COLLECT_STATE,
    STOP_COLLECT_STATE,
    PREPROCESS_STATE,
    INFERENCE_STATE,
    DISPLAY_STATE,
    FAIL_STATE
};
typedef enum AppState AppState;

enum DataCollectMode { NO_DATA_COLLECT, SENSOR_DATA_COLLECT, STIMULUS_DATA_COLLECT };
typedef enum DataCollectMode DataCollectMode;

typedef struct {
    ecg_peak_f32_t *qrsFindPeakCtx;
    int32_t numQrsPeaks;
    uint32_t *qrsPeaks;
    uint32_t *rrIntervals;
    uint8_t *rrMask;
} qrs_context_t;

typedef struct {
    uint32_t numSamples;
    AppState state;
    DataCollectMode collectMode;
    float32_t *rawData;
    float32_t *ecgData;
    float32_t *qrsData;
    float32_t *bufData;
    uint8_t *segMask;
    hrv_td_metrics_t *hrvMetrics;
    hk_result_t *results;
    uint32_t errorCode;
} hk_app_store_t;

extern int volatile sensorCollectBtnPressed;
extern int volatile clientCollectBtnPressed;
extern const ns_power_config_t nsPwrCfg;
extern ns_button_config_t nsBtnCfg;
extern ns_rpc_config_t nsRpcCfg;
extern ns_i2c_config_t nsI2cCfg;
extern ns_core_config_t nsCoreCfg;
extern usb_config_t usbCfg;

extern const char *HK_RHYTHM_LABELS[3];
extern const char *HK_BEAT_LABELS[4];
extern const char *HK_HEART_RATE_LABELS[3];
extern const char *HK_SEGMENT_LABELS[4];

extern hk_sensor_t sensorCtx;
extern arm_biquad_casd_df1_inst_f32 ecgFilterCtx;
extern arm_biquad_casd_df1_inst_f32 qrsFilterCtx;
extern qrs_context_t qrsCtx;

extern hk_app_store_t hkStore;

#endif // __HK_STORE_H
