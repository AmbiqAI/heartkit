/**
 * @file main.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Main application
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "arm_math.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// neuralSPOT
#include "ns_ambiqsuite_harness.h"
#include "ns_malloc.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_rpc_generic_data.h"
#include "ns_usb.h"
// Locals
#include "constants.h"
#include "main.h"
#include "model.h"
#include "physiokit.h"
#include "sensor.h"
#include "stimulus.h"
#include "store.h"
#include "usb_handler.h"

uint32_t
print_hk_result(hk_result_t *result) {
    uint32_t numBeats = result->numNormBeats + result->numPacBeats + result->numPvcBeats;
    const char *rhythm = HK_HEART_RATE_LABELS[result->heartRhythm];
    ns_printf("----------------------\n");
    ns_printf("** HeartKit Results **\n");
    ns_printf("----------------------\n");
    ns_printf("  Heart Rate: %lu\n", result->heartRate);
    ns_printf(" Heart Rhythm: %s\n", rhythm);
    ns_printf("  Total Beats: %lu\n", numBeats);
    ns_printf("   Norm Beats: %lu\n", result->numNormBeats);
    ns_printf("    PAC Beats: %lu\n", result->numPacBeats);
    ns_printf("    PVC Beats: %lu\n", result->numPvcBeats);
    ns_printf("Unknown Beats: %lu\n", result->numNoiseBeats);
    ns_printf("   Arrhythmia: %lu\n", result->arrhythmia);
    ns_printf("----------------------\n");
    return 0;
}

uint32_t
apply_arrhythmia_model() {
    uint32_t err;
    uint32_t yIdx;
    float32_t yVal;
    for (size_t i = 0; i < HK_DATA_LEN - ARR_FRAME_LEN + 1; i += ARR_FRAME_LEN) {
        err = arrhythmia_inference(&hkStore.ecgData[i], &yVal, &yIdx);
        ns_printf("Arrhythmia Detection: class=%lu confidence=%0.2f \n", yIdx, yVal);
        if (err) {
        } else if (yVal >= ARR_THRESHOLD && (yIdx == HeartRhythmAfib || yIdx == HeartRhythmAfut)) {
            hkStore.results->arrhythmia = HeartRhythmAfib;
            hkStore.results->heartRhythm = HeartRateTachycardia;
        }
    }
    return err;
}

uint32_t
apply_segmentation_model() {
    uint32_t err = 0;
    for (size_t i = 0; i < HK_DATA_LEN - SEG_FRAME_LEN + 1; i += SEG_FRAME_LEN) {
        err = segmentation_inference(&hkStore.ecgData[i], &hkStore.segMask[i], SEG_OVERLAP_LEN, SEG_THRESHOLD);
    }
    err |= segmentation_inference(&hkStore.ecgData[HK_DATA_LEN - SEG_FRAME_LEN], &hkStore.segMask[HK_DATA_LEN - SEG_FRAME_LEN],
                                  SEG_OVERLAP_LEN, SEG_THRESHOLD);
    return err;
}

uint32_t
apply_hrv_model() {
    uint32_t err = 0;
    for (size_t i = 0; i < HK_DATA_LEN; i++) {
        hkStore.qrsData[i] = hkStore.segMask[i] == HeartSegmentQrs ? 10 * hkStore.qrsData[i] : hkStore.qrsData[i];
    }
    // Find QRS peaks
    qrsCtx.numQrsPeaks = pk_ecg_find_peaks(qrsCtx.qrsFindPeakCtx, hkStore.qrsData, HK_DATA_LEN, qrsCtx.qrsPeaks);
    pk_compute_rr_intervals(qrsCtx.qrsPeaks, qrsCtx.numQrsPeaks, qrsCtx.rrIntervals);
    pk_filter_rr_intervals(qrsCtx.rrIntervals, qrsCtx.numQrsPeaks, qrsCtx.rrMask, SAMPLE_RATE);
    pk_compute_hrv_from_rr_intervals(qrsCtx.rrIntervals, qrsCtx.numQrsPeaks, qrsCtx.rrMask, hkStore.hrvMetrics);
    // Apply HRV head
    float32_t bpm = 60 / (hkStore.hrvMetrics->meanNN / SAMPLE_RATE);
    hkStore.results->heartRhythm = bpm < 60 ? HeartRateBradycardia : bpm <= 100 ? HeartRateNormal : HeartRateTachycardia;
    hkStore.results->heartRate = (uint32_t)bpm;
    return err;
}

uint32_t
apply_beat_model() {
    uint32_t err = 0;
    uint32_t bIdx;
    uint32_t beatLabel;
    float32_t beatValue;

    uint32_t avgRR = (uint32_t)(hkStore.hrvMetrics->meanNN);
    uint32_t bOffset = (BEAT_FRAME_LEN >> 1);
    uint32_t bStart = 0;
    for (int i = 1; i < qrsCtx.numQrsPeaks - 1; i++) {
        bIdx = qrsCtx.qrsPeaks[i];
        bStart = bIdx - bOffset;
        if (bIdx < bOffset || bStart < avgRR || bStart + avgRR + BEAT_FRAME_LEN > HK_DATA_LEN) {
            beatLabel = HeartBeatNormal;
            beatValue = 1;
        } else {
            err |= beat_inference(&hkStore.ecgData[bStart - avgRR], &hkStore.ecgData[bStart], &hkStore.ecgData[bStart + avgRR], &beatValue,
                                  &beatLabel);
        }
        ns_printf("Beat Detection: loc=%lu class=%lu confidence=%0.2f \n", bIdx, beatLabel, beatValue);
        // If confidence is too low, skip
        if (beatValue < BEAT_THRESHOLD) {
            beatLabel = HeartBeatNoise;
        }
        // Place beat label in upper nibble
        hkStore.segMask[bIdx] |= ((beatLabel + 1) << 4);
        switch (beatLabel) {
        case HeartBeatPac:
            hkStore.results->numPacBeats += 1;
            break;
        case HeartBeatPvc:
            hkStore.results->numPvcBeats += 1;
            break;
        case HeartBeatNormal:
            hkStore.results->numNormBeats += 1;
            break;
        case HeartBeatNoise:
            hkStore.results->numNoiseBeats += 1;
            break;
        }
    }
    return err;
}

DataCollectMode
get_collect_mode() {
    if (sensorCollectBtnPressed) {
        return SENSOR_DATA_COLLECT;
    }
    if (clientCollectBtnPressed) {
        return STIMULUS_DATA_COLLECT;
    }
    return NO_DATA_COLLECT;
}

void
clear_collect_mode() {
    sensorCollectBtnPressed = false;
    clientCollectBtnPressed = false;
}

void
print_to_pc(const char *msg) {
    /**
     * @brief Print to PC over RPC
     */
    if (usbCfg.available) {
        ns_rpc_data_remotePrintOnPC(msg);
    }
    ns_printf(msg);
}

uint32_t
fetch_samples_from_stimulus(float32_t *samples, uint32_t numSamples) {
    /**
     * @brief Fetch stimulus samples
     * @param samples Buffer to store samples
     * @param numSamples # requested samples
     * @return # samples actually fetched
     */
    static uint32_t stimulusOffset = 0;
    uint32_t newSamples = MIN(test_stimulus_len - stimulusOffset, numSamples);
    memcpy(samples, &test_stimulus[stimulusOffset], newSamples * sizeof(float32_t));
    stimulusOffset = (stimulusOffset + newSamples) % test_stimulus_len;
    return newSamples;
}

uint32_t
fetch_samples_from_pc(float32_t *samples, uint32_t numSamples, uint32_t *reqSamples) {
    /**
     * @brief Fetch samples from PC over RPC
     * @param samples Buffer to store samples
     * @param numSamples # requested samples
     * @return # samples actually fetched
     */
    static char rpcFetchSamplesDesc[] = "FETCH_SAMPLES";
    int err;
    if (!usbCfg.available) {
        return 0;
    }

    *reqSamples = MIN(numSamples, RPC_BUF_LEN);
    dataBlock resultBlock = {.length = *reqSamples,
                             .dType = float32_e,
                             .description = rpcFetchSamplesDesc,
                             .cmd = generic_cmd,
                             .buffer = {
                                 .data = (uint8_t *)(samples),
                                 .dataLength = *reqSamples * sizeof(float32_t),
                             }};

    err = ns_rpc_data_computeOnPC(&resultBlock, &resultBlock);
    if (resultBlock.description != rpcFetchSamplesDesc) {
        ns_free(resultBlock.description);
    }
    if (resultBlock.buffer.data != (uint8_t *)samples) {
        ns_free(resultBlock.buffer.data);
    }
    if (err) {
        ns_printf("Failed fetching from PC w/ error: %x\n", err);
        *reqSamples = 0;
        return 1;
    }
    memcpy(samples, resultBlock.buffer.data, resultBlock.buffer.dataLength);
    *reqSamples = resultBlock.buffer.dataLength / sizeof(float32_t);
    return 0;
}

void
send_samples_to_pc() {
    /**
     * @brief Send sensor samples to PC
     */
    static char rpcSendSamplesDesc[] = "SEND_SAMPLES";
    if (!usbCfg.available) {
        return;
    }
    dataBlock commandBlock = {.length = 0,
                              .dType = float32_e,
                              .description = rpcSendSamplesDesc,
                              .cmd = generic_cmd,
                              .buffer = {
                                  .data = NULL,
                                  .dataLength = 0,
                              }};
    for (size_t i = 0; i < HK_DATA_LEN; i += RPC_BUF_LEN) {
        uint32_t numSamples = MIN(HK_DATA_LEN - i, RPC_BUF_LEN);
        commandBlock.length = i;
        commandBlock.buffer.data = (uint8_t *)(&hkStore.ecgData[i]);
        commandBlock.buffer.dataLength = numSamples * sizeof(float32_t);
        ns_rpc_data_sendBlockToPC(&commandBlock);
        ns_delay_us(200);
    }
}

void
send_mask_to_pc() {
    /**
     * @brief Send mask to PC
     */
    static char rpcSendMaskDesc[] = "SEND_MASK";
    if (!usbCfg.available) {
        return;
    }
    dataBlock commandBlock = {.length = 0,
                              .dType = uint8_e,
                              .description = rpcSendMaskDesc,
                              .cmd = generic_cmd,
                              .buffer = {
                                  .data = NULL,
                                  .dataLength = 0,
                              }};
    for (size_t i = 0; i < HK_DATA_LEN; i += RPC_BUF_LEN) {
        uint32_t numSamples = MIN(HK_DATA_LEN - i, RPC_BUF_LEN);
        commandBlock.length = i;
        commandBlock.buffer.data = (uint8_t *)(&hkStore.segMask[i]);
        commandBlock.buffer.dataLength = numSamples * sizeof(uint8_t);
        ns_rpc_data_sendBlockToPC(&commandBlock);
    }
}

void
send_results_to_pc() {
    /**
     * @brief Send results to PC
     */
    static char rpcSendResultsDesc[] = "SEND_RESULTS";
    print_hk_result(hkStore.results);
    if (!usbCfg.available) {
        return;
    }
    dataBlock commandBlock = {.length = 1,
                              .dType = uint32_e,
                              .description = rpcSendResultsDesc,
                              .cmd = generic_cmd,
                              .buffer = {
                                  .data = (uint8_t *)hkStore.results,
                                  .dataLength = sizeof(hk_result_t),
                              }};
    ns_rpc_data_sendBlockToPC(&commandBlock);
}

uint32_t
clear_results() {
    /**
     * @brief Clear results
     */
    hkStore.results->heartRate = 0;
    hkStore.results->heartRhythm = HeartRateNormal;
    hkStore.results->arrhythmia = HeartRhythmNormal;
    hkStore.results->numPacBeats = 0;
    hkStore.results->numPvcBeats = 0;
    hkStore.results->numNormBeats = 0;
    hkStore.results->numNoiseBeats = 0;
    qrsCtx.numQrsPeaks = 0;
    memset(hkStore.segMask, HeartSegmentNormal, HK_DATA_LEN);
    return 0;
}

void
start_collecting(void) {
    /**
     * @brief Setup sensor for collecting
     *
     */
    uint32_t newSamples = 0;
    if (hkStore.collectMode == SENSOR_DATA_COLLECT) {
        start_sensor(&sensorCtx);
        // Discard first second for sensor warm up time
        for (size_t i = 0; i < 100; i++) {
            capture_sensor_data(&sensorCtx, hkStore.rawData, NULL, NULL, NULL, SENSOR_LEN, &newSamples);
            ns_delay_us(10000);
        }
    }
    clear_results();
    hkStore.numSamples = 0;
}

uint32_t
collect_samples() {
    /**
     * @brief Collect samples from sensor or PC
     * @return # new samples collected
     */
    uint32_t err = 0;
    uint32_t newSamples = 0;
    uint32_t reqSamples = HK_DATA_LEN - hkStore.numSamples;
    float32_t *data = &hkStore.rawData[hkStore.numSamples];
    if (hkStore.numSamples >= HK_DATA_LEN) {
        return newSamples;
    }
    if (hkStore.collectMode == STIMULUS_DATA_COLLECT) {
        // newSamples = fetch_samples_from_stimulus(data, reqSamples);
        err = fetch_samples_from_pc(data, reqSamples, &newSamples);
        ns_delay_us(200);
    } else if (hkStore.collectMode == SENSOR_DATA_COLLECT) {
        err = capture_sensor_data(&sensorCtx, data, NULL, NULL, NULL, reqSamples, &newSamples);
    }
    hkStore.numSamples += newSamples;
    ns_delay_us(5000);
    return err;
}

void
stop_collecting(void) {
    /**
     * @brief Stop collecting sensor data
     */
    if (hkStore.collectMode == SENSOR_DATA_COLLECT) {
        stop_sensor(&sensorCtx);
    }
}

uint32_t
preprocess() {
    /**
     * @brief Run preprocess on data
     */
    uint32_t err = 0;
    err |= pk_standardize(hkStore.rawData, hkStore.rawData, SENSOR_LEN, NORM_STD_EPS);
    err |= pk_apply_biquad_filtfilt(&ecgFilterCtx, hkStore.rawData, hkStore.ecgData, HK_DATA_LEN, hkStore.bufData);
    err |= pk_apply_biquad_filtfilt(&qrsFilterCtx, hkStore.rawData, hkStore.qrsData, HK_DATA_LEN, hkStore.bufData);
    err |= pk_standardize(hkStore.ecgData, hkStore.ecgData, HK_DATA_LEN, NORM_STD_EPS);
    err |= pk_standardize(hkStore.qrsData, hkStore.qrsData, HK_DATA_LEN, NORM_STD_EPS);
    return err;
}

uint32_t
inference() {
    /**
     * @brief Run inference on data
     */
    uint32_t err = 0;
    err = apply_arrhythmia_model();
    // If arrhythmia detected, skip other models
    if (hkStore.results->arrhythmia == HeartRhythmNormal) {
        err |= apply_segmentation_model();
        err |= apply_hrv_model();
        err |= apply_beat_model();
    }
    return err;
}

void
wakeup() {
    /**
     * @brief Wakeup SoC from sleep
     */
    am_bsp_itm_printf_enable();
    am_bsp_debug_printf_enable();
    ns_delay_us(5000);
    usb_update_state();
}

void
deepsleep() {
    /**
     * @brief Put SoC into deep sleep
     */
    usbCfg.available = false;
    am_bsp_itm_printf_disable();
    am_bsp_debug_printf_disable();
    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}

void
setup() {
    /**
     * @brief Application setup
     */
    // Power configuration (mem, cache, peripherals, clock)
    NS_TRY(ns_core_init(&nsCoreCfg), "NS Core init failed");
    NS_TRY(ns_power_config(&nsPwrCfg), "NS Power config failed");
    NS_TRY(init_usb_handler(&usbCfg), "USB init failed");
    am_hal_interrupt_master_enable();
    wakeup();
    ns_i2c_interface_init(&nsI2cCfg, AM_HAL_IOM_400KHZ);
    ns_rpc_genericDataOperations_init(&nsRpcCfg);
    NS_TRY(ns_peripheral_button_init(&nsBtnCfg), "NS Button init failed");
    NS_TRY(init_sensor(&sensorCtx), "Sensor init failed");
    NS_TRY(init_models(), "Model init failed");
    ns_printf("♥️ HeartKit Demo\n\n");
    ns_printf("Please select data collection options:\n\n\t1. BTN1=sensor\n\t2. BTN2=stimulus\n");
}

void
loop() {
    /**
     * @brief Application loop
     */
    switch (hkStore.state) {
    case IDLE_STATE:
        hkStore.collectMode = get_collect_mode();
        if (hkStore.collectMode != NO_DATA_COLLECT) {
            hkStore.state = START_COLLECT_STATE;
            wakeup();
        } else {
            ns_printf("\n\nIDLE_STATE\n");
            hkStore.state = IDLE_STATE;
            deepsleep();
        }
        break;

    case START_COLLECT_STATE:
        print_to_pc("COLLECT_STATE\n");
        start_collecting();
        clear_collect_mode();
        hkStore.state = COLLECT_STATE;
        break;

    case COLLECT_STATE:
        hkStore.errorCode = collect_samples();
        if (hkStore.errorCode != 0) {
            hkStore.state = STOP_COLLECT_STATE;
        } else if (hkStore.numSamples >= HK_DATA_LEN) {
            hkStore.state = STOP_COLLECT_STATE;
        }
        break;

    case STOP_COLLECT_STATE:
        stop_collecting();
        hkStore.state = hkStore.errorCode != 0 ? FAIL_STATE : PREPROCESS_STATE;
        break;

    case PREPROCESS_STATE:
        print_to_pc("PREPROCESS_STATE\n");
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
        hkStore.errorCode = preprocess();
        hkStore.state = INFERENCE_STATE;
        break;

    case INFERENCE_STATE:
        print_to_pc("INFERENCE_STATE\n");
        hkStore.errorCode = inference();
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
        hkStore.state = hkStore.errorCode != 0 ? FAIL_STATE : DISPLAY_STATE;
        break;

    case DISPLAY_STATE:
        send_samples_to_pc();
        send_mask_to_pc();
        send_results_to_pc();
        print_to_pc("DISPLAY_STATE\n");
        ns_delay_us(DISPLAY_LEN_USEC);
        hkStore.state = IDLE_STATE;
        break;

    case FAIL_STATE:
        ns_printf("FAIL_STATE err=%d\n", hkStore.errorCode);
        hkStore.state = IDLE_STATE;
        hkStore.errorCode = 0;
        break;

    default:
        hkStore.state = IDLE_STATE;
        break;
    }
}

int
main(void) {
    /**
     * @brief Main function
     * @return int
     */
    setup();
    while (1) {
        loop();
    }
}
