/**
 * @file sensor.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Initializes and collects sensor data from MAX86150
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "sensor.h"
#include "arm_math.h"
#include "constants.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_i2c.h"
#include "ns_max86150_driver.h"

#define MAX86150_NUM_SLOTS (4)
#define RESET_DELAY_US (10000)

uint32_t
init_sensor(hk_sensor_t *ctx) {
    /**
     * @brief Initialize and configure sensor block (MAX86150)
     *
     */
    max86150_powerup(ctx->maxCtx);
    ns_delay_us(RESET_DELAY_US);
    max86150_reset(ctx->maxCtx);
    ns_delay_us(RESET_DELAY_US);
    max86150_set_fifo_slots(ctx->maxCtx, ctx->maxCfg->fifoSlotConfigs);
    max86150_set_almost_full_rollover(ctx->maxCtx, ctx->maxCfg->fifoRolloverFlag);
    max86150_set_ppg_sample_average(ctx->maxCtx, ctx->maxCfg->ppgSampleAvg);
    max86150_set_ppg_adc_range(ctx->maxCtx, ctx->maxCfg->ppgAdcRange);
    max86150_set_ppg_sample_rate(ctx->maxCtx, ctx->maxCfg->ppgSampleRate);
    max86150_set_ppg_pulse_width(ctx->maxCtx, ctx->maxCfg->ppgPulseWidth);
    max86150_set_prox_int_flag(ctx->maxCtx, 0);
    max86150_set_led_current_range(ctx->maxCtx, 0, ctx->maxCfg->led0CurrentRange);
    max86150_set_led_current_range(ctx->maxCtx, 1, ctx->maxCfg->led1CurrentRange);
    max86150_set_led_current_range(ctx->maxCtx, 2, ctx->maxCfg->led2CurrentRange);
    max86150_set_led_pulse_amplitude(ctx->maxCtx, 0, ctx->maxCfg->led0PulseAmplitude);
    max86150_set_led_pulse_amplitude(ctx->maxCtx, 1, ctx->maxCfg->led1PulseAmplitude);
    max86150_set_led_pulse_amplitude(ctx->maxCtx, 2, ctx->maxCfg->led2PulseAmplitude);
    max86150_set_ecg_sample_rate(ctx->maxCtx, ctx->maxCfg->ecgSampleRate);
    max86150_set_ecg_ia_gain(ctx->maxCtx, ctx->maxCfg->ecgIaGain);
    max86150_set_ecg_pga_gain(ctx->maxCtx, ctx->maxCfg->ecgPgaGain);
    max86150_powerup(ctx->maxCtx);
    stop_sensor(ctx);
    return 0;
}

void
start_sensor(hk_sensor_t *ctx) {
    /**
     * @brief Takes sensor out of low-power mode and enables FIFO
     *
     */
    // max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(ctx->maxCtx, 1);
}

void
stop_sensor(hk_sensor_t *ctx) {
    /**
     * @brief Puts sensor in low-power mode
     *
     */
    max86150_set_fifo_enable(ctx->maxCtx, 0);
    // max86150_shutdown(&maxCtx);
}

uint32_t
capture_sensor_data(hk_sensor_t *ctx, float32_t *slot0, float32_t *slot1, float32_t *slot2, float32_t *slot3, uint32_t maxSamples) {
    uint32_t numSamples;
    int32_t val;
    float32_t *slots[4] = {slot0, slot1, slot2, slot3};
    uint32_t maxFifoBuffer[MAX86150_FIFO_DEPTH * MAX86150_NUM_SLOTS];
    numSamples = max86150_read_fifo_samples(ctx->maxCtx, maxFifoBuffer, ctx->maxCfg->fifoSlotConfigs, ctx->maxCfg->numSlots);
    numSamples = MIN(maxSamples, numSamples);
    for (size_t i = 0; i < numSamples; i++) {
        for (size_t j = 0; j < ctx->maxCfg->numSlots; j++) {
            val = maxFifoBuffer[ctx->maxCfg->numSlots * i + j];
            if (ctx->maxCfg->fifoSlotConfigs[j] == Max86150SlotEcg) {
                // ECG data is 18-bit 2's complement. If MSB=1 then make negative
                if (val & (1 << 17)) {
                    val -= (1 << 18);
                }
                slots[j][i] = (float32_t)(val) * (0.012247 / 9.5 / 8); // 12.247μV/IA_GAIN/PGA_GAIN
            } else {
                slots[j][i] = (float32_t)(val);
            }
        }
    }
    return numSamples;
}
