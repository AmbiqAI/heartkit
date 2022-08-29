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

#include "model.h"
#include "max86150.h"
#include "ns_io_i2c.h"

//*****************************************************************************
//*** Assorted Configs and helpers
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define SAMPLE_RATE 250
#define INF_WINDOW_LEN (SAMPLE_RATE*5) // 5 seconds
#define NUM_ELEMENTS (3)
//*****************************************************************************

//*****************************************************************************
//*** Model-specific Stuff
static const char *rhythm_labels[] = { "normal", "afib", "aflut" };
static const char *beat_labels[] = { "normal", "pac", "aberrated", "pvc" };
static uint8_t sensorBuffer[MAX86150_FIFO_DEPTH*NUM_ELEMENTS*3] = { 0 };
static uint32_t ecgData[INF_WINDOW_LEN];
static const uint16_t MAX86150_ADDR = 0x5E;
int captureButtonPressed = 0;
//***
//*****************************************************************************


//*****************************************************************************
//*** Tensorflow Globals
tflite::ErrorReporter *errorReporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *modelInput = nullptr;
TfLiteTensor *modelOutput = nullptr;

constexpr int kTensorArenaSize = 1024 * 70; // TODO: Determine based of model size
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
    .device = 0,
    .speed = 100000,
};

static inline int max86150_write_read(uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    ns_io_i2c_write_read(&i2cConfig, addr, write_buf, num_write, read_buf, num_read);
}
static inline int max86150_read(uint8_t *buf, uint32_t num_bytes, uint16_t addr) {
    ns_io_i2c_read(&i2cConfig, buf, num_bytes, addr);
}
static inline int max86150_write(const uint8_t *buf, uint32_t num_bytes, uint16_t addr) {
    ns_io_i2c_write(&i2cConfig, buf, num_bytes, addr);
}
max86150_context_t maxCtx = {
    .addr = MAX86150_ADDR,
    .i2c_write_read = max86150_write_read,
    .i2c_read = max86150_read,
    .i2c_write = max86150_write,
};
//***
//*****************************************************************************

enum AppState {
    IdleState,
    StartCaptureState,
    CapturingState,
    StopCaptureState,
    InferenceState,
    DisplayState,
    FailState
}; typedef enum AppState AppState;

void init_ecg_sensor(void) {
    /**
     * @brief Initialize and configure ECG sensor (MAX86150)
     *
     */
    max86150_reset(&maxCtx);
    ns_delay_us(10000);
    max86150_set_fifo_slots(
        &maxCtx,
        Max86150SlotEcg, Max86150SlotPpgLed1,
        Max86150SlotPpgLed2, Max86150SlotOff
    );
    max86150_set_ppg_sample_average(&maxCtx, 2);        // Avg 4 samples
    max86150_set_ppg_adc_range(&maxCtx, 3);             // 16,384 nA Scale
    max86150_set_ppg_sample_rate(&maxCtx, 4);           // 100 Samples/sec
    max86150_set_ppg_pulse_width(&maxCtx, 1);           // 100 us
    // max86150_set_proximity_threshold(&i2c_dev, MAX86150_ADDR, 0x1F); // Disabled
    max86150_set_led_current_range(&maxCtx, 0, 0);      // IR LED 50 mA
    max86150_set_led_current_range(&maxCtx, 1, 0);      // RED LED 50 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 0, 0x64); // IR LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 1, 0x64); // RED LED 20 mA
    max86150_set_ecg_sample_rate(&maxCtx, 3);           // Fs = 200 Hz
    max86150_set_ecg_ia_gain(&maxCtx, 1);               // 9.5 V/V
    max86150_set_ecg_pga_gain(&maxCtx, 3);              // 8 V/V
}

void start_ecg_sensor(void) {
    /**
     * @brief Takes ECG sensor out of low-power mode and enables FIFO
     *
     */
    max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(&maxCtx, 1);
}

void stop_ecg_sensor(void) {
    /**
     * @brief Puts ECG sensor in low-power mode
     *
     */
    max86150_set_fifo_enable(&maxCtx, 0);
    max86150_shutdown(&maxCtx);
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
    model = tflite::GetModel(slu_model_tflite);
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
    static tflite::MicroInterpreter staticInterpreter(
        model, resolver, tensorArena, kTensorArenaSize, errorReporter
    );
    interpreter = &staticInterpreter;

    // Allocate memory from the tensorArena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    modelInput = interpreter->input(0);
    modelOutput = interpreter->output(0);
}


int main(void) {
    /**
     * @brief Main function - infinite loop listening and inferring
     * @return int
     */

    TfLiteStatus invokeStatus;
    uint32_t numSamples = 0;
    uint32_t numSensorSamples = 0;
    AppState state = IdleState;
    ns_itm_printf_enable();

#ifdef ENERGYMODE
    ns_uart_printf_enable();
    ns_init_power_monitor_state();
    ns_power_set_monitor_state(&am_ai_audio_default);
#else
    ns_debug_printf_enable(); // Leave crypto on for ease of debugging
    ns_power_config(&ns_development_default);
#endif

    // Initialize everything else
    // init_ecg_sensor();
    model_init();
    ns_peripheral_button_init(&button_config);
    ns_printf("Press button to capture ECG...\n");

#ifdef ENERGYMODE
    ns_power_set_monitor_state(AM_AI_DATA_COLLECTION);
#endif

    while (1) {
        switch (state) {
        case IdleState:
            if (captureButtonPressed) {
                state = StartCaptureState;
            } else {
                ns_delay_us(1000);
                state = IdleState;
            }
            break;
        case StartCaptureState:
            ns_printf("Starting ECG capture.\n");
            numSamples = 0;
            start_ecg_sensor();
            state = CapturingState;
            break;
        case CapturingState:
            numSensorSamples = max86150_read_fifo_samples(&maxCtx, &sensorBuffer[0], 3);
            for (size_t i = 0; i < numSensorSamples && numSamples < INF_WINDOW_LEN; i++) {
                memcpy(&ecgData[numSamples++], &sensorBuffer[3*NUM_ELEMENTS*i], 3);
            }
            if (numSamples >= INF_WINDOW_LEN) {
                state = StopCaptureState;
            } else {
                state = CapturingState;
                ns_delay_us(100);
            }
            break;
        case StopCaptureState:
            ns_printf("Finished ECG capture.\n");
            stop_ecg_sensor();
            state = InferenceState;
            break;
        case InferenceState:
            ns_printf("Running inference\n");
            // Copy sensor data to input buffer
            invokeStatus = interpreter->Invoke();
            if (invokeStatus != kTfLiteOk) {
                ns_printf("Invoke failed\n");
                state = FailState;
            }
            state = DisplayState;
            break;
        case DisplayState:
            captureButtonPressed = 0;
            state = IdleState;
            ns_printf("Done\n");
            break;
        case FailState:
            // Report error and reset state
            state = IdleState;
            break;
        default:
            break;
        }
    }
}
