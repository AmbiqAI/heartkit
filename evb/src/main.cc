//*****************************************************************************
//
// Copyright (c) 2022, Ambiq Micro, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// Third party software included in this distribution is subject to the
// additional license terms as defined in the /docs/licenses directory.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is part of revision R4.1.0 of the AmbiqSuite
// NeuralSPOT
//
//*****************************************************************************

//*****************************************************************************
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// Tensorflow Lite for Microcontroller includes (somewhat boilerplate)
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
//***
//*****************************************************************************

//*****************************************************************************
//*** NeuralSPOT Includes
#include "ns_ambiqsuite_harness.h"
#ifdef RINGBUFFER_MODE
    #include "ns_ipc_ring_buffer.h"
#endif
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
//***
//*****************************************************************************

#include "constants.h"
#include "model.h"
#include "SEGGER_RTT.h"
#include "max86150.h"
#include "ns_io_i2c.h"

//*****************************************************************************
//*** Assorted Configs and helpers
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// #define EMULATION
#define SAMPLE_RATE 250
#define INF_WINDOW_LEN (SAMPLE_RATE*5) // 5 seconds
#define NUM_ELEMENTS (3)
#define ECG_BUFFER_LEN (INF_WINDOW_LEN+SAMPLE_RATE)
#define MAX86150_ADDR (0x5E)
//*****************************************************************************

#define RTT_PORT 1
#define RTT_BUFFER_LEN (2*INF_WINDOW_LEN)

enum AppState {
    IDLE_STATE,
    START_CAPTURE_STATE,
    CAPTURING_STATE,
    STOP_CAPTURE_STATE,
    INFERENCE_STATE,
    DISPLAY_STATE,
    FAIL_STATE
}; typedef enum AppState AppState;

//*****************************************************************************
//*** Application globals
static const char *heart_rhythm_labels[] = { "normal", "afib", "aflut", "noise" };
static const char *heart_beat_labels[] = { "normal", "pac", "aberrated", "pvc", "noise" };
static const char *hear_rate_labels[] = { "normal", "tachycardia", "bradycardia", "noise" };


uint8_t rttBuffer[RTT_BUFFER_LEN];
static uint32_t sensorBuffer[MAX86150_FIFO_DEPTH*NUM_ELEMENTS] = { 0 };
static uint32_t ecgBuffer[ECG_BUFFER_LEN];
int captureButtonPressed = 0;
AppState state = IDLE_STATE;
uint32_t numSamples;

//***
//*****************************************************************************

//*****************************************************************************
//*** Tensorflow Globals
tflite::ErrorReporter *errorReporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *modelInput = nullptr;
TfLiteTensor *modelOutput = nullptr;
constexpr int kTensorArenaSize = 1024 * 300;
alignas(16) uint8_t tensorArena[kTensorArenaSize];
//***
//*****************************************************************************

//*****************************************************************************
//*** Peripheral Configs
ns_button_config_t button_config = {
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &captureButtonPressed,
    .button_1_flag = NULL
};

ns_i2c_config_t i2cConfig = {
    .i2cBus = 0,
    .device = 1,
    .speed = 100000,
};

static int max86150_write_read(uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    return ns_io_i2c_write_read(&i2cConfig, addr, write_buf, num_write, read_buf, num_read);
}
static int max86150_read(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_io_i2c_read(&i2cConfig, buf, num_bytes, addr);
}
static int max86150_write(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_io_i2c_write(&i2cConfig, buf, num_bytes, addr);
}
max86150_context_t maxCtx = {
    .addr = MAX86150_ADDR,
    .i2c_write_read = max86150_write_read,
    .i2c_read = max86150_read,
    .i2c_write = max86150_write,
};
//***
//*****************************************************************************


void init_ecg_sensor(void) {
    /**
     * @brief Initialize and configure ECG sensor (MAX86150)
     *
     */
#ifdef EMULATION
    // Do nothing
#else
    ns_io_i2c_init(&i2cConfig);
    max86150_reset(&maxCtx);
    ns_delay_us(10000);
    max86150_set_fifo_slots(
        &maxCtx,
        Max86150SlotEcg, Max86150SlotPpgLed1,
        Max86150SlotPpgLed2, Max86150SlotOff
    );
    max86150_set_almost_full_rollover(&maxCtx, 1);      // !FIFO rollover: should decide
    max86150_set_ppg_sample_average(&maxCtx, 2);        // Avg 4 samples
    max86150_set_ppg_adc_range(&maxCtx, 2);             // 16,384 nA Scale
    max86150_set_ppg_sample_rate(&maxCtx, 4);           // 100 Samples/sec
    max86150_set_ppg_pulse_width(&maxCtx, 3);           // 400 us
    // max86150_set_proximity_threshold(&i2c_dev, MAX86150_ADDR, 0x1F); // Disabled
    max86150_set_led_current_range(&maxCtx, 0, 0);      // IR LED 50 mA
    max86150_set_led_current_range(&maxCtx, 1, 0);      // RED LED 50 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 0, 0x64); // IR LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 1, 0x64); // RED LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 2, 0x64); // RED LED 20 mA
    max86150_set_ecg_sample_rate(&maxCtx, 3);           // Fs = 200 Hz
    max86150_set_ecg_ia_gain(&maxCtx, 1);               // 9.5 V/V
    max86150_set_ecg_pga_gain(&maxCtx, 2);              // 4 V/V
#endif
}

void start_ecg_sensor(void) {
    /**
     * @brief Takes ECG sensor out of low-power mode and enables FIFO
     *
     */
#ifdef EMULATION
    // Do nothing
#else
    max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(&maxCtx, 1);
#endif
}

void stop_ecg_sensor(void) {
    /**
     * @brief Puts ECG sensor in low-power mode
     *
     */
#ifdef EMULATION
    // Do nothing
#else
    max86150_set_fifo_enable(&maxCtx, 0);
    max86150_shutdown(&maxCtx);
#endif
}

uint32_t capture_ecg_sensor(uint32_t* buffer, uint32_t size) {
    uint32_t numSamples;
    uint32_t val;
#ifdef EMULATION
    numSamples = 8;
    for (size_t i = 0; i < numSamples; i++) {
        buffer[i] = (rand() % (255 - 0 + 1)) + 0;
        SEGGER_RTT_Write(RTT_PORT, (uint8_t *)&val, 4);
        ns_delay_us(4000);
    }

#else
    numSamples = max86150_read_fifo_samples(&maxCtx, sensorBuffer, NUM_ELEMENTS);
    if (numSamples == 0) {
        ns_delay_us(2500);
    }
    for (size_t i = 0; i < numSamples; i++) {
        buffer[i] = sensorBuffer[NUM_ELEMENTS*i];
        SEGGER_RTT_Write(RTT_PORT, &buffer[i], 4);
        ns_delay_us(5000);
    }
#endif
    return numSamples;
}

void model_init(void) {
    /**
     * @brief Initialize TF model
     *
     */
    static tflite::MicroErrorReporter micro_error_reporter;
    errorReporter = &micro_error_reporter;

    tflite::InitializeTarget();

    // Map the model into a usable data structure.
    model = tflite::GetModel(g_afib_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter,
            "Model provided is schema version %d not equal to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION
        );
        return;
    }

    // This pulls in all the operation implementations we need.
    // static tflite::MicroMutableOpResolver<1> resolver;
    static tflite::AllOpsResolver resolver;
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensorArena, kTensorArenaSize, errorReporter
    );
    interpreter = &static_interpreter;

    // Allocate memory from the tensorArena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return;
    }

    size_t bytesUsed = interpreter->arena_used_bytes();
    if (bytesUsed > kTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter,
            "Model requires %d bytes for arena but given only %d bytes.",
            bytesUsed, kTensorArenaSize
        );
    }

    // Obtain pointers to the model's input and output tensors.
    modelInput = interpreter->input(0);
    modelOutput = interpreter->output(0);
}

int model_run() {
    // Copy sensor data to input buffer
    int8_t y;
    TfLiteStatus invokeStatus = interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    } else {
        for (size_t i = 0; i < modelOutput->dims->data[0]; i++) {
            y = modelOutput->data.int8[i];
        }
        return y;
    }
}

static void setup() {
    /**
     * @brief Application setup
     *
     */
    ns_itm_printf_enable();
    am_hal_interrupt_master_enable();

#ifdef ENERGYMODE
    ns_uart_printf_enable();
    ns_init_power_monitor_state();
    ns_power_set_monitor_state(&am_ai_audio_default);
#else
    ns_debug_printf_enable(); // Leave crypto on for ease of debugging
    ns_power_config(&ns_development_default);
#endif
    SEGGER_RTT_Init();
    SEGGER_RTT_ConfigUpBuffer(RTT_PORT, "DATA", rttBuffer, RTT_BUFFER_LEN, SEGGER_RTT_MODE_NO_BLOCK_SKIP);
    ns_delay_us(20);
    // am_bsp_itm_printf_disable();
    init_ecg_sensor();
    model_init();
    ns_peripheral_button_init(&button_config);
    ns_printf("Press button to capture ECG...\n");

#ifdef ENERGYMODE
    ns_power_set_monitor_state(AM_AI_DATA_COLLECTION);
#endif
}

static void loop() {
    /**
     * @brief Application loop
     *
     */
    int result;
    switch (state) {
    case IDLE_STATE:
        if (captureButtonPressed) {
            state = START_CAPTURE_STATE;
        } else {
            ns_delay_us(1000);
            state = IDLE_STATE;
        }
        break;

    case START_CAPTURE_STATE:
        ns_printf("Started ECG capture.\n");
        numSamples = 0;
        start_ecg_sensor();
        state = CAPTURING_STATE;
        break;

    case CAPTURING_STATE:
        numSamples += capture_ecg_sensor(&ecgBuffer[numSamples], ECG_BUFFER_LEN-numSamples);
        if (numSamples >= INF_WINDOW_LEN) {
            state = STOP_CAPTURE_STATE;
        } else {
            state = CAPTURING_STATE;
        }
        break;

    case STOP_CAPTURE_STATE:
        ns_printf("Finished ECG capture.\n");
        stop_ecg_sensor();
        state = INFERENCE_STATE;
        break;

    case INFERENCE_STATE:
        ns_printf("Running inference\n");
        result = model_run();
        if (result == -1) {
            state = FAIL_STATE;
        } else {
            state = DISPLAY_STATE;
        }
        break;

    case DISPLAY_STATE:
        captureButtonPressed = 0;
        state = START_CAPTURE_STATE;
        ns_printf("Done\n");
        break;

    case FAIL_STATE:
        // Report error and reset state
        state = IDLE_STATE;
        break;

    default:
        state = IDLE_STATE;
        break;
    }
}

int main(void) {
    /**
     * @brief Main function
     * @return int
     */
    setup();
    while (1) { loop(); }
}
