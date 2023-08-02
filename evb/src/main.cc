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
    ns_printf("Heart Rhythm: %s\n", rhythm);
    ns_printf(" Total Beats: %lu\n", numBeats);
    ns_printf("  Norm Beats: %lu\n", result->numNormBeats);
    ns_printf("   PAC Beats: %lu\n", result->numPacBeats);
    ns_printf("   PVC Beats: %lu\n", result->numPvcBeats);
    ns_printf(" Noise Beats: %lu\n", result->numNoiseBeats);
    ns_printf("  Arrhythmia: %lu\n", result->arrhythmia);
    ns_printf("----------------------\n");
    return 0;
}

uint32_t
apply_arrhythmia_model() {
    uint32_t err = 0;
    uint32_t yIdx;
    float32_t yVal;
    hkResults.arrhythmia = HeartRhythmNormal;
    for (size_t i = 0; i < HK_DATA_LEN - ARR_FRAME_LEN + 1; i += ARR_FRAME_LEN) {
        err = arrhythmia_inference(&hkEcgData[i], &yVal, &yIdx);
        if (err) {
        } else if (yIdx == HeartRhythmAfib || yIdx == HeartRhythmAfut) {
            hkResults.arrhythmia = HeartRhythmAfib;
        }
    }
    return err;
}

uint32_t
apply_segmentation_model() {
    uint32_t err = 0;
    // We dont predict on first and last overlap size so set to normal
    memset(hkSegMask, HeartSegmentNormal, HK_DATA_LEN);
    for (size_t i = 0; i < HK_DATA_LEN - SEG_FRAME_LEN + 1; i += SEG_FRAME_LEN) {
        err = segmentation_inference(&hkEcgData[i], &hkSegMask[i], SEG_OVERLAP_LEN);
    }
    err |= segmentation_inference(&hkEcgData[HK_DATA_LEN - SEG_FRAME_LEN], &hkSegMask[HK_DATA_LEN - SEG_FRAME_LEN], SEG_OVERLAP_LEN);
    return err;
}

uint32_t
apply_hrv_model() {
    uint32_t err = 0;
    uint32_t numQrsPeaks;
    // Find QRS peaks
    numQrsPeaks = pk_ecg_find_peaks(&qrsFindPeakCtx, hkEcgData, HK_DATA_LEN, hkQrsPeaks);
    pk_compute_rr_intervals(hkQrsPeaks, numQrsPeaks, hkRRIntervals);
    pk_filter_rr_intervals(hkRRIntervals, numQrsPeaks, hkQrsMask, SAMPLE_RATE);
    pk_compute_hrv_from_rr_intervals(hkRRIntervals, numQrsPeaks, hkQrsMask, &hkHrvMetrics);

    // Apply HRV head
    float32_t bpm = 60 / (hkHrvMetrics.meanNN / SAMPLE_RATE);
    uint32_t avgRR = (uint32_t)hkHrvMetrics.meanNN;
    hkResults.heartRhythm = bpm < 60 ? HeartRateBradycardia : bpm <= 100 ? HeartRateNormal : HeartRateTachycardia;
    hkResults.heartRate = (uint32_t)bpm;
}

uint32_t
apply_beat_model() {
    // Apply beat head
    uint32_t err = 0;
    uint32_t bIdx;
    uint32_t beatLabel;
    float32_t beatValue;

    hkResults.numPacBeats = 0;
    hkResults.numPvcBeats = 0;
    hkResults.numNormBeats = 0;
    hkResults.numNoiseBeats = 0;

    uint32_t avgRR = (uint32_t)(hkHrvMetrics.meanNN);
    uint32_t bOffset = (BEAT_FRAME_LEN >> 1);
    uint32_t bStart = 0;
    for (int i = 1; i < numQrsPeaks - 1; i++) {
        bIdx = hkQrsPeaks[i];
        bStart = bIdx - bOffset;
        if (hkQrsMask[i]) {
            beatLabel = HeartBeatNoise;
        } else if (bIdx < bOffset || bStart < avgRR || bStart + avgRR + BEAT_FRAME_LEN > HK_DATA_LEN) {
            beatLabel = HeartBeatNormal;
        } else {
            ns_printf("beats (%lu, %lu, %lu)\n", bStart - avgRR, bStart, bStart + avgRR);
            err |= beat_inference(&hkEcgData[bStart - avgRR], &hkEcgData[bStart], &hkEcgData[bStart + avgRR], &beatValue, &beatLabel);
        }
        // Place beat label in upper nibble
        hkSegMask[bIdx] |= ((beatLabel + 1) << 4);
        switch (beatLabel) {
        case HeartBeatPac:
            hkResults.numPacBeats += 1;
            break;
        case HeartBeatPvc:
            hkResults.numPvcBeats += 1;
            break;
        case HeartBeatNormal:
            hkResults.numNormBeats += 1;
            break;
        case HeartBeatNoise:
            hkResults.numNoiseBeats += 1;
            break;
        }
    }
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
fetch_samples_from_pc(float32_t *samples, uint32_t offset, uint32_t numSamples) {
    /**
     * @brief Fetch samples from PC over RPC
     * @param samples Buffer to store samples
     * @param offset Buffer offset
     * @param numSamples # requested samples
     * @return # samples actually fetched
     */
    static char rpcFetchSamplesDesc[] = "FETCH_SAMPLES";
    int err;
    if (!usbCfg.available) {
        return 0;
    }
    binary_t binaryBlock = {
        .data = (uint8_t *)(&samples[offset]),
        .dataLength = numSamples * sizeof(float32_t),
    };
    dataBlock resultBlock = {
        .length = numSamples, .dType = float32_e, .description = rpcFetchSamplesDesc, .cmd = generic_cmd, .buffer = binaryBlock};
    err = ns_rpc_data_computeOnPC(&resultBlock, &resultBlock);
    if (resultBlock.description != rpcFetchSamplesDesc) {
        ns_free(resultBlock.description);
    }
    if (resultBlock.buffer.data != (uint8_t *)&samples[offset]) {
        ns_free(resultBlock.buffer.data);
    }
    if (err) {
        ns_printf("Failed fetching from PC w/ error: %x\n", err);
        return 0;
    }
    memcpy(&samples[offset], resultBlock.buffer.data, resultBlock.buffer.dataLength);
    return resultBlock.buffer.dataLength / sizeof(float32_t);
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
    for (size_t i = 0; i < HK_DATA_LEN; i += 200) {
        uint32_t numSamples = MIN(HK_DATA_LEN - i, 200);
        commandBlock.length = i;
        commandBlock.buffer.data = (uint8_t *)(&hkEcgData[i]);
        commandBlock.buffer.dataLength = numSamples * sizeof(float32_t);
        ns_rpc_data_sendBlockToPC(&commandBlock);
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
    for (size_t i = 0; i < HK_DATA_LEN; i += 200) {
        uint32_t numSamples = MIN(HK_DATA_LEN - i, 200);
        commandBlock.length = i;
        commandBlock.buffer.data = (uint8_t *)(&hkSegMask[i]);
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
    print_hk_result(&hkResults);
    if (!usbCfg.available) {
        return;
    }
    dataBlock commandBlock = {.length = 1,
                              .dType = uint32_e,
                              .description = rpcSendResultsDesc,
                              .cmd = generic_cmd,
                              .buffer = {
                                  .data = (uint8_t *)&hkResults,
                                  .dataLength = sizeof(hk_result_t),
                              }};
    ns_rpc_data_sendBlockToPC(&commandBlock);
}

void
start_collecting(void) {
    /**
     * @brief Setup sensor for collecting
     *
     */
    if (appStore.collectMode == SENSOR_DATA_COLLECT) {
        start_sensor(&sensorCtx);
        // Discard first second for sensor warm up time
        for (size_t i = 0; i < 100; i++) {
            capture_sensor_data(&sensorCtx, hkRawData, NULL, NULL, NULL, SENSOR_LEN);
            ns_delay_us(10000);
        }
    }
    appStore.numSamples = 0;
}

uint32_t
collect_samples() {
    /**
     * @brief Collect samples from sensor or PC
     * @return # new samples collected
     */
    uint32_t newSamples = 0;
    uint32_t reqSamples = HK_DATA_LEN - appStore.numSamples;
    float32_t *data = &hkRawData[appStore.numSamples];
    if (appStore.numSamples >= HK_DATA_LEN) {
        return newSamples;
    }
    if (appStore.collectMode == STIMULUS_DATA_COLLECT) {
        newSamples = fetch_samples_from_stimulus(data, reqSamples);
    } else if (appStore.collectMode == SENSOR_DATA_COLLECT) {
        newSamples = capture_sensor_data(&sensorCtx, data, NULL, NULL, NULL, reqSamples);
    }
    appStore.numSamples += newSamples;
    ns_delay_us(5000);
    return newSamples;
}

void
stop_collecting(void) {
    /**
     * @brief Stop collecting sensor data
     */
    if (appStore.collectMode == SENSOR_DATA_COLLECT) {
        stop_sensor(&sensorCtx);
    }
}

uint32_t
preprocess() {
    /**
     * @brief Run preprocess on data
     */
    uint32_t err = 0;
    err |= pk_apply_biquad_filter(&ecgFilterCtx, hkRawData, hkEcgData, HK_DATA_LEN);
    err |= pk_standardize(hkEcgData, hkEcgData, HK_DATA_LEN, 0.1);
    send_samples_to_pc();
    return err;
}

uint32_t
inference() {
    /**
     * @brief Run inference on data
     */
    uint32_t err = 0;
    err |= apply_arrhythmia_model();
    err |= apply_segmentation_model();
    err |= apply_hrv_model();
    err |= apply_beat_model();
    return err;
}

void
wakeup() {
    /**
     * @brief Wakeup SoC from sleep
     */
    am_bsp_itm_printf_enable();
    am_bsp_debug_printf_enable();
}

void
deepsleep() {
    /**
     * @brief Put SoC into deep sleep
     */
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
    am_hal_interrupt_master_enable();
    wakeup();
    ns_i2c_interface_init(&nsI2cCfg, AM_HAL_IOM_400KHZ);
    NS_TRY(init_usb_handler(&usbCfg), "USB init failed");
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
    switch (appStore.state) {
    case IDLE_STATE:
        appStore.collectMode = get_collect_mode();
        if (appStore.collectMode != NO_DATA_COLLECT) {
            appStore.state = START_COLLECT_STATE;
            wakeup();
        } else {
            ns_printf("IDLE_STATE\n");
            appStore.state = IDLE_STATE;
            deepsleep();
        }
        break;

    case START_COLLECT_STATE:
        print_to_pc("COLLECT_STATE\n");
        start_collecting();
        clear_collect_mode();
        appStore.state = COLLECT_STATE;
        break;

    case COLLECT_STATE:
        collect_samples();
        if (appStore.numSamples >= HK_DATA_LEN) {
            appStore.state = STOP_COLLECT_STATE;
        }
        break;

    case STOP_COLLECT_STATE:
        stop_collecting();
        appStore.state = PREPROCESS_STATE;
        break;

    case PREPROCESS_STATE:
        print_to_pc("PREPROCESS_STATE\n");
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
        appStore.errorCode = preprocess();
        appStore.state = INFERENCE_STATE;
        break;

    case INFERENCE_STATE:
        print_to_pc("INFERENCE_STATE\n");
        appStore.errorCode = inference();
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
        appStore.state = appStore.errorCode == 1 ? FAIL_STATE : DISPLAY_STATE;
        break;

    case DISPLAY_STATE:
        print_to_pc("DISPLAY_STATE\n");
        send_mask_to_pc();
        send_results_to_pc();
        ns_delay_us(DISPLAY_LEN_USEC);
        appStore.state = IDLE_STATE;
        break;

    case FAIL_STATE:
        ns_printf("FAIL_STATE err=%d\n", appStore.errorCode);
        appStore.state = IDLE_STATE;
        appStore.errorCode = 0;
        break;

    default:
        appStore.state = IDLE_STATE;
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
