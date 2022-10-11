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

#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include "arm_math.h"
// neuralSPOT
#include "ns_ambiqsuite_harness.h"
#ifdef RINGBUFFER_MODE
    #include "ns_ipc_ring_buffer.h"
#endif
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_usb.h"
#include "ns_malloc.h"
#include "ns_rpc_generic_data.h"
// Locals
#include "constants.h"
#include "sensor.h"
#include "preprocessing.h"
#include "model.h"
#include "main.h"


// Application globals
static float32_t sensorBuffer[SENSOR_BUFFER_LEN];
uint32_t numSamples;
int volatile static sensorCollectBtnPressed = false;
int volatile static clientCollectBtnPressed = false;

AppState state = IDLE_STATE;
DataCollectMode collectMode = SENSOR_DATA_COLLECT;

bool usbAvailable = false;
uint32_t modelResult = 0;
char rpcSendSamplesDesc[]  = "SEND_SAMPLES";
char rpcFetchSamplesDesc[] = "FETCH_SAMPLES";

//*****************************************************************************
//*** Peripheral Configs
ns_button_config_t button_config = {
    .button_0_enable = true,
    .button_1_enable = true,
    .button_0_flag = &sensorCollectBtnPressed,
    .button_1_flag = &clientCollectBtnPressed
};

// Handle TinyUSB events
void tud_mount_cb(void) { usbAvailable = true; }
void tud_resume_cb(void) { usbAvailable = true; }
void tud_umount_cb(void) { usbAvailable = false; }
void tud_suspend_cb(bool remote_wakeup_en) { usbAvailable = false; }

void background_task() {
    tud_task();
}

void sleep_us(uint32_t time) {
    uint32_t chunk;
    while (time > 0){
        chunk = MIN(10000, time);
        ns_delay_us(chunk);
        time -= chunk;
        background_task();
    }
}

void init_rpc(void) {
    ns_rpc_config_t rpcConfig = {
        .mode = NS_RPC_GENERICDATA_CLIENT,
        .sendBlockToEVB_cb = NULL,
        .fetchBlockFromEVB_cb = NULL,
        .computeOnEVB_cb = NULL
    };
    ns_rpc_genericDataOperations_init(&rpcConfig);
}

void start_collecting(void) {
    numSamples = 0;
    if (collectMode == SENSOR_DATA_COLLECT) {
        start_sensor();
    }
}

void stop_collecting(void) {
    if (collectMode == SENSOR_DATA_COLLECT) {
        stop_sensor();
    }
}

uint32_t fetch_samples_from_pc(float32_t *samples, uint32_t numSamples) {
    binary_t binaryBlock = {
        .data = (uint8_t *)samples,
        .dataLength = numSamples*sizeof(float32_t),
    };
    dataBlock resultBlock = {
        .length = numSamples,
        .dType = float32_e,
        .description = rpcFetchSamplesDesc,
        .cmd = generic_cmd,
        .buffer = binaryBlock
    };
    ns_rpc_data_computeOnPC(&resultBlock, &resultBlock);
    memcpy(samples, resultBlock.buffer.data, resultBlock.buffer.dataLength);
    ns_free(resultBlock.description);
    ns_free(resultBlock.buffer.data);
    return resultBlock.length;
}

void send_samples_to_pc(float32_t *samples, uint32_t numSamples) {
    binary_t binaryBlock = {
        .data = (uint8_t *)samples,
        .dataLength = numSamples*sizeof(float32_t),
    };
    dataBlock commandBlock = {
        .length = numSamples,
        .dType = float32_e,
        .description = rpcSendSamplesDesc,
        .cmd = generic_cmd,
        .buffer = binaryBlock
    };
    ns_rpc_data_sendBlockToPC(&commandBlock);
}

void send_results_to_pc() {
}

uint32_t collect_samples() {
    uint32_t newSamples = 0;
    if (collectMode == CLIENT_DATA_COLLECT) {
        newSamples = fetch_samples_from_pc(&sensorBuffer[numSamples], 10);
        numSamples += newSamples;
        sleep_us(40000); // Tweak for overhead
    } else if (collectMode == SENSOR_DATA_COLLECT) {
        newSamples = capture_sensor_data(&sensorBuffer[numSamples]);
        if (newSamples) {
            // for (size_t i = 0; i < newSamples; i++) {
            //     sensorBuffer[numSamples+i] = i;
            // }
            send_samples_to_pc(&sensorBuffer[numSamples], newSamples);
        }
        numSamples += newSamples;
        sleep_us(1000); // Tweak for overhead
    }
    return newSamples;
}


void setup() {
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
    ns_delay_us(50);
    init_rpc();
    init_sensor();
    init_preprocess();
    init_model();
    ns_peripheral_button_init(&button_config);
    ns_printf("Heart Arrhythmia Classifier Demo\n\n");
    ns_printf("Please select data collection options:\n\t1. BTN0=sensor\n\t2. BTN1=client\n");
#ifdef ENERGYMODE
    ns_power_set_monitor_state(AM_AI_DATA_COLLECTION);
#endif
}

void loop() {
    /**
     * @brief Application loop
     *
     */
    static int err = 0;
    switch (state) {
    case IDLE_STATE:
        if (sensorCollectBtnPressed | clientCollectBtnPressed) {
            collectMode = sensorCollectBtnPressed ? SENSOR_DATA_COLLECT : CLIENT_DATA_COLLECT;
            sensorCollectBtnPressed = false;
            clientCollectBtnPressed = false;
            state = START_COLLECT_STATE;
        } else {
            sleep_us(10000);
        }
        break;

    case START_COLLECT_STATE:
        ns_rpc_data_remotePrintOnPC("COLLECT STAGE\n");
        ns_printf("COLLECT STAGE\n");
        start_collecting();
        state = COLLECT_STATE;
        break;

    case COLLECT_STATE:
        collect_samples();
        if (numSamples >= (INF_WINDOW_LEN)) {
            state = STOP_COLLECT_STATE;
        }
        break;

    case STOP_COLLECT_STATE:
        stop_collecting();
        state = PREPROCESS_STATE;
        break;

    case PREPROCESS_STATE:
        ns_printf("PREPROCESS STAGE\n");
        preprocess(sensorBuffer, sensorBuffer, INF_WINDOW_LEN);
        state = INFERENCE_STATE;
        break;

    case INFERENCE_STATE:
        ns_printf("INFERENCE STAGE\n");
        err = model_inference(&sensorBuffer[0], &modelResult);
        state = err == 0 ? DISPLAY_STATE : FAIL_STATE;
        break;

    case DISPLAY_STATE:
        ns_printf("DISPLAY STAGE\n");
        state = START_COLLECT_STATE;
        // TODO: Convert logits to probs and add inconclusive label
        ns_printf("\tLabel=%s [%d] \n", heart_rhythm_labels[modelResult], modelResult);
        break;

    case FAIL_STATE:
        ns_printf("FAIL STAGE (err=%d)\n", err);
        state = IDLE_STATE;
        err = 0;
        break;

    default:
        state = IDLE_STATE;
        break;
    }
    background_task();
}

int main(void) {
    /**
     * @brief Main function
     * @return int
     */
    setup();
    while (1) { loop(); }
}
