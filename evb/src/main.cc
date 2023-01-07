/**
 * @file main.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Main application
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
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
#include "preprocessing.h"
#include "sensor.h"

static const char *heart_rhythm_labels[] = {"NSR", "AFIB/AFL"};
// const char *heart_beat_labels[] = { "normal", "pac", "aberrated", "pvc", "noise" };
// const char *hear_rate_labels[] = { "normal", "tachycardia", "bradycardia", "noise" };

// Application globals
static uint32_t numSamples = 0;
static float32_t sensorBuffer[SENSOR_BUFFER_LEN];
static float32_t modelResults[NUM_CLASSES] = {0};
static int modelResult = -1;
static bool usbAvailable = false;
static int volatile sensorCollectBtnPressed = false;
static int volatile clientCollectBtnPressed = false;
static AppState state = IDLE_STATE;
static DataCollectMode collectMode = SENSOR_DATA_COLLECT;

const ns_power_config_t ns_pwr_config = {.api = &ns_power_V1_0_0,
                                         .eAIPowerMode = NS_MINIMUM_PERF,
                                         .bNeedAudAdc = false,
                                         .bNeedSharedSRAM = false,
                                         .bNeedCrypto = true,
                                         .bNeedBluetooth = false,
                                         .bNeedUSB = true,
                                         .bNeedIOM = false, // We will manually enable IOM0
                                         .bNeedAlternativeUART = false,
                                         .b128kTCM = false};

//*****************************************************************************
//*** Peripheral Configs
ns_button_config_t button_config = {.api = &ns_button_V1_0_0,
                                    .button_0_enable = true,
                                    .button_1_enable = true,
                                    .button_0_flag = &sensorCollectBtnPressed,
                                    .button_1_flag = &clientCollectBtnPressed};

// Handle TinyUSB events
void
tud_mount_cb(void) {
    usbAvailable = true;
}
void
tud_resume_cb(void) {
    usbAvailable = true;
}
void
tud_umount_cb(void) {
    usbAvailable = false;
}
void
tud_suspend_cb(bool remote_wakeup_en) {
    usbAvailable = false;
}

void
background_task() {
    /**
     * @brief Run background tasks
     *
     */
}

void
sleep_us(uint32_t time) {
    /**
     * @brief Enable longer sleeps while also running background tasks on interval
     * @param time Sleep duration in microseconds
     */
    uint32_t chunk;
    while (time > 0) {
        chunk = MIN(10000, time);
        ns_delay_us(chunk);
        time -= chunk;
        background_task();
    }
}

void
init_rpc(void) {
    /**
     * @brief Initialize RPC and USB
     *
     */
    ns_rpc_config_t rpcConfig = {.api = &ns_rpc_gdo_V1_0_0,
                                 .mode = NS_RPC_GENERICDATA_CLIENT,
                                 .sendBlockToEVB_cb = NULL,
                                 .fetchBlockFromEVB_cb = NULL,
                                 .computeOnEVB_cb = NULL};
    ns_rpc_genericDataOperations_init(&rpcConfig);
}

void
print_to_pc(const char *msg) {
    /**
     * @brief Print to PC over RPC
     *
     */
    if (usbAvailable) {
        ns_rpc_data_remotePrintOnPC(msg);
    }
    ns_printf(msg);
}

void
start_collecting(void) {
    /**
     * @brief Setup sensor for collecting
     *
     */
    if (collectMode == SENSOR_DATA_COLLECT) {
        start_sensor();
        // Discard first second- this will give sensor and user warm up time
        for (size_t i = 0; i < 100; i++) {
            capture_sensor_data(sensorBuffer);
            sleep_us(10000);
        }
    }
    numSamples = 0;
}

void
stop_collecting(void) {
    /**
     * @brief Disable sensor
     *
     */
    if (collectMode == SENSOR_DATA_COLLECT) {
        stop_sensor();
    }
}

uint32_t
fetch_samples_from_pc(float32_t *samples, uint32_t numSamples) {
    /**
     * @brief Fetch samples from PC over RPC
     * @param samples Buffer to store samples
     * @param numSamples # requested samples
     * @return # samples actually fetched
     */
    static char rpcFetchSamplesDesc[] = "FETCH_SAMPLES";
    int err;
    if (!usbAvailable) {
        return 0;
    }
    binary_t binaryBlock = {
        .data = (uint8_t *)samples,
        .dataLength = numSamples * sizeof(float32_t),
    };
    dataBlock resultBlock = {
        .length = numSamples, .dType = float32_e, .description = rpcFetchSamplesDesc, .cmd = generic_cmd, .buffer = binaryBlock};
    err = ns_rpc_data_computeOnPC(&resultBlock, &resultBlock);
    if (resultBlock.description != rpcFetchSamplesDesc) {
        ns_free(resultBlock.description);
    }
    if (resultBlock.buffer.data != (uint8_t *)samples) {
        ns_free(resultBlock.buffer.data);
    }
    if (err) {
        ns_printf("Failed fetching from PC w/ error: %x\n", err);
        return 0;
    }
    memcpy(samples, resultBlock.buffer.data, resultBlock.buffer.dataLength);
    return resultBlock.length;
}

void
send_samples_to_pc(float32_t *samples, uint32_t numSamples) {
    /**
     * @brief Send sensor samples to PC
     * @param samples Samples to send
     * @param numSamples # samples to send
     */
    static char rpcSendSamplesDesc[] = "SEND_SAMPLES";
    if (!usbAvailable) {
        return;
    }
    binary_t binaryBlock = {
        .data = (uint8_t *)samples,
        .dataLength = numSamples * sizeof(float32_t),
    };
    dataBlock commandBlock = {
        .length = numSamples, .dType = float32_e, .description = rpcSendSamplesDesc, .cmd = generic_cmd, .buffer = binaryBlock};
    ns_rpc_data_sendBlockToPC(&commandBlock);
}

void
send_results_to_pc(float32_t *results, uint32_t numResults) {
    /**
     * @brief Send classification results to PC
     * @param results Buffer with model outputs (logits)
     * @param numResults # model ouputs
     */
    static char rpcSendResultsDesc[] = "SEND_RESULTS";
    if (!usbAvailable) {
        return;
    }
    binary_t binaryBlock = {
        .data = (uint8_t *)results,
        .dataLength = numResults * sizeof(float32_t),
    };
    dataBlock commandBlock = {
        .length = numResults, .dType = float32_e, .description = rpcSendResultsDesc, .cmd = generic_cmd, .buffer = binaryBlock};
    ns_rpc_data_sendBlockToPC(&commandBlock);
}

uint32_t
collect_samples() {
    /**
     * @brief Collect samples from sensor or PC
     * @return # new samples collected
     */
    uint32_t newSamples = 0;
    if (collectMode == CLIENT_DATA_COLLECT) {
        newSamples = fetch_samples_from_pc(&sensorBuffer[numSamples], 10);
        numSamples += newSamples;
        sleep_us(20000);
    } else if (collectMode == SENSOR_DATA_COLLECT) {
        newSamples = capture_sensor_data(&sensorBuffer[numSamples]);
        if (newSamples) {
            send_samples_to_pc(&sensorBuffer[numSamples], newSamples);
        }
        numSamples += newSamples;
        sleep_us(10000);
    }
    return newSamples;
}

void
preprocess_samples() {
    /**
     * @brief Preprocess by bandpass filtering and standardizing
     *
     */
    bandpass_filter(sensorBuffer, sensorBuffer, COLLECT_LEN);
    standardize(&sensorBuffer[PAD_WINDOW_LEN], &sensorBuffer[PAD_WINDOW_LEN], INF_WINDOW_LEN);
}

void
wakeup() {
    am_bsp_itm_printf_enable();
    am_bsp_debug_printf_enable();
    ns_delay_us(50);
}

void
deepsleep() {
    am_bsp_itm_printf_disable();
    am_bsp_debug_printf_disable();
    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}

void
setup() {
    /**
     * @brief Application setup
     *
     */
    // Power configuration (mem, cache, peripherals, clock)
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    ns_core_init(&ns_core_cfg);
    ns_power_config(&ns_pwr_config);
    am_hal_pwrctrl_periph_enable(AM_HAL_PWRCTRL_PERIPH_IOM0);
    // Enable Interrupts
    am_hal_interrupt_master_enable();
    // Enable SWO/USB
    wakeup();
    // Initialize blocks
    init_rpc();
    init_sensor();
    init_preprocess();
    init_model();
    ns_peripheral_button_init(&button_config);
    ns_printf("♥️ Heart Arrhythmia Classifier Demo\n\n");
    ns_printf("Please select data collection options:\n\n\t1. BTN1=sensor\n\t2. BTN2=client\n");
}

void
loop() {
    /**
     * @brief Application loop
     *
     */
    static int err = 0;
    switch (state) {
    case IDLE_STATE:
        if (sensorCollectBtnPressed | clientCollectBtnPressed) {
            collectMode = sensorCollectBtnPressed ? SENSOR_DATA_COLLECT : CLIENT_DATA_COLLECT;
            wakeup();
            state = START_COLLECT_STATE;
        } else {
            deepsleep();
        }
        break;

    case START_COLLECT_STATE:
        print_to_pc("COLLECT_STATE\n");
        start_collecting();
        state = COLLECT_STATE;
        break;

    case COLLECT_STATE:
        collect_samples();
        if (numSamples >= COLLECT_LEN) {
            state = STOP_COLLECT_STATE;
        }
        break;

    case STOP_COLLECT_STATE:
        stop_collecting();
        sensorCollectBtnPressed = false; // DEBOUNCE
        clientCollectBtnPressed = false; // DEBOUNCE
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
        state = PREPROCESS_STATE;
        break;

    case PREPROCESS_STATE:
        print_to_pc("PREPROCESS_STATE\n");
        preprocess_samples();
        state = INFERENCE_STATE;
        break;

    case INFERENCE_STATE:
        print_to_pc("INFERENCE_STATE\n");
        modelResult = model_inference(&sensorBuffer[PAD_WINDOW_LEN], modelResults);
        am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
        state = modelResult == -1 ? FAIL_STATE : DISPLAY_STATE;
        break;

    case DISPLAY_STATE:
        print_to_pc("DISPLAY_STATE\n");
        state = IDLE_STATE;
        ns_printf("\tLabel=%s [%d,%f]\n", heart_rhythm_labels[modelResult], modelResult, modelResults[modelResult]);
        send_results_to_pc(modelResults, NUM_CLASSES);
        break;

    case FAIL_STATE:
        ns_printf("FAIL_STATE (err=%d)\n", err);
        state = IDLE_STATE;
        err = 0;
        break;

    default:
        state = IDLE_STATE;
        break;
    }
    background_task();
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
