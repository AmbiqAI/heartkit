#include "arm_math.h"
#include "ns_ambiqsuite_harness.h"
#include "max86150.h"
#include "ns_io_i2c.h"
#include "constants.h"
#include "sensor.h"

uint32_t maxFifoBuffer[MAX86150_FIFO_DEPTH*NUM_ELEMENTS];

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

void init_sensor(void) {
    /**
     * @brief Initialize and configure sensor block (MAX86150)
     *
     */
    ns_io_i2c_init(&i2cConfig);
    max86150_powerup(&maxCtx);
    ns_delay_us(10000);
    max86150_reset(&maxCtx);
    ns_delay_us(10000);
    max86150_set_fifo_slots(
        &maxCtx,
        Max86150SlotEcg, Max86150SlotOff,
        Max86150SlotOff, Max86150SlotOff
    );
    max86150_set_almost_full_rollover(&maxCtx, 1);      // !FIFO rollover: should decide
    max86150_set_ppg_sample_average(&maxCtx, 2);        // Avg 4 samples
    max86150_set_ppg_adc_range(&maxCtx, 2);             // 16,384 nA Scale
    max86150_set_ppg_sample_rate(&maxCtx, 5);           // 200 Samples/sec
    max86150_set_ppg_pulse_width(&maxCtx, 1);           // 100 us
    // max86150_set_proximity_threshold(&i2c_dev, MAX86150_ADDR, 0x1F); // Disabled

    max86150_set_led_current_range(&maxCtx, 0, 0);      // IR LED 50 mA
    max86150_set_led_current_range(&maxCtx, 1, 0);      // RED LED 50 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 0, 0xFF); // IR LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 1, 0xFF); // RED LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 2, 0x64); // AMB LED 20 mA

    max86150_set_ecg_sample_rate(&maxCtx, 3);           // Fs = 200 Hz
    max86150_set_ecg_ia_gain(&maxCtx, 1);               // 9.5 V/V
    max86150_set_ecg_pga_gain(&maxCtx, 3);              // 8 V/V
    max86150_clear_fifo(&maxCtx);
}


void start_sensor(void) {
    /**
     * @brief Takes sensor out of low-power mode and enables FIFO
     *
     */
    // max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(&maxCtx, 1);
    max86150_clear_fifo(&maxCtx);
}

void stop_sensor(void) {
    /**
     * @brief Puts sensor in low-power mode
     *
     */
    max86150_set_fifo_enable(&maxCtx, 0);
    // max86150_shutdown(&maxCtx);
}


uint32_t capture_sensor_data(float32_t* buffer) {
    uint32_t numSamples;
    numSamples = max86150_read_fifo_samples(&maxCtx, maxFifoBuffer, NUM_ELEMENTS);
    for (size_t i = 0; i < numSamples; i++) {
        buffer[i] = (float32_t)maxFifoBuffer[NUM_ELEMENTS*i+0];
    }
    return numSamples;
}
