#include "arm_math.h"
#include "ns_ambiqsuite_harness.h"
#include "max86150.h"
#include "ns_io_i2c.h"
#include "constants.h"
#include "sensor.h"

#define NUM_SLOTS (1)
Max86150SlotType maxSlotsConfig[] = {
    Max86150SlotEcg, Max86150SlotOff,
    Max86150SlotOff, Max86150SlotOff
};

uint32_t maxFifoBuffer[MAX86150_FIFO_DEPTH*NUM_SLOTS];

ns_i2c_config_t i2cConfig = {
    .i2cBus = 0,
    .device = 1,
    .speed = 400000,
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

void generate_synthetic_data(float32_t *buffer, int len) {
#ifdef EMULATION
  float32_t dom_amp = 1000;
  float32_t dom_freq = 3;
  float32_t dom_offset = 100;
  float32_t dom_phi = (0.0f/180.0f)*PI;
  float32_t sec_amp = 500;
  float32_t sec_freq = 40;
  float32_t sec_offset = 80;
  float32_t sec_phi = (0.0f/180.0f)*PI;
  static float32_t t_step = 0.0;
  for (int i = 0; i < len; i++) {
    buffer[i] =  dom_amp*arm_cos_f32(2*PI*dom_freq*t_step + dom_phi) + dom_offset;
    buffer[i] += sec_amp*arm_cos_f32(2*PI*sec_freq*t_step + sec_phi) + sec_offset;
    t_step += 1.0/SAMPLE_RATE;
  }
#endif
}

int init_sensor(void) {
    /**
     * @brief Initialize and configure sensor block (MAX86150)
     *
     */
    ns_io_i2c_init(&i2cConfig);
    max86150_powerup(&maxCtx);
    ns_delay_us(10000);
    max86150_reset(&maxCtx);
    ns_delay_us(10000);
    max86150_set_fifo_slots(&maxCtx, maxSlotsConfig);
    max86150_set_almost_full_rollover(&maxCtx, 1);      // FIFO rollover: should decide
    max86150_set_ppg_sample_average(&maxCtx, 2);        // Avg 4 samples
    max86150_set_ppg_adc_range(&maxCtx, 2);             // 16,384 nA Scale
    max86150_set_ppg_sample_rate(&maxCtx, 5);           // 200 Samples/sec
    max86150_set_ppg_pulse_width(&maxCtx, 1);           // 100 us
    // max86150_set_proximity_threshold(&i2c_dev, MAX86150_ADDR, 0x1F); // Disabled

    max86150_set_led_current_range(&maxCtx, 0, 0);      // IR LED 50 mA
    max86150_set_led_current_range(&maxCtx, 1, 0);      // RED LED 50 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 0, 0x32); // IR LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 1, 0x32); // RED LED 20 mA
    max86150_set_led_pulse_amplitude(&maxCtx, 2, 0x32); // AMB LED 20 mA

    max86150_set_ecg_sample_rate(&maxCtx, 3);           // Fs = 200 Hz
    max86150_set_ecg_ia_gain(&maxCtx, 0);               // 5 V/V
    max86150_set_ecg_pga_gain(&maxCtx, 2);              // 4 V/V
    max86150_set_fifo_enable(&maxCtx, 1);
    max86150_clear_fifo(&maxCtx);
    return 0;
}


void start_sensor(void) {
    /**
     * @brief Takes sensor out of low-power mode and enables FIFO
     *
     */
    max86150_powerup(&maxCtx);
    max86150_set_fifo_enable(&maxCtx, 1);
    max86150_clear_fifo(&maxCtx);
}

void stop_sensor(void) {
    /**
     * @brief Puts sensor in low-power mode
     *
     */
    // max86150_set_fifo_enable(&maxCtx, 0);
    // max86150_shutdown(&maxCtx);
}


uint32_t capture_sensor_data(float32_t* buffer) {
    uint32_t numSamples;
#ifdef EMULATION
    numSamples = 10;
    generate_synthetic_data(buffer, numSamples*NUM_SLOTS);
#else
    int32_t val;
    numSamples = max86150_read_fifo_samples(&maxCtx, maxFifoBuffer, maxSlotsConfig, NUM_SLOTS);
    for (size_t i = 0; i < numSamples; i++) {
        for (size_t j = 0; j < NUM_SLOTS; j++) {
            val = maxFifoBuffer[NUM_SLOTS*i+j];
            // ECG data is 18-bit 2's complement. If MSB=1 then make negative
            if (val & (1 << 17)) {
            // if ((maxSlotsConfig[j] == Max86150SlotEcg) && (val & (1 << 17))) {
                val -= (1 << 18);
            }
            buffer[i] = (float32_t)(val);
        }
    }
#endif
    return numSamples;
}
