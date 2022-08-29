#include <ctype.h>
#include "max86150.h"

// Status Registers
static const uint8_t MAX86150_INT_STAT1 = 0x00;
static const uint8_t MAX86150_INT_STAT2 = 0x01;
static const uint8_t MAX86150_INT_EN1 = 0x02;
static const uint8_t MAX86150_INT_EN2 = 0x03;
// FIFO Registers
static const uint8_t MAX86150_FIFO_WR_PTR = 0x04;
static const uint8_t MAX86150_FIFO_OVERFLOW = 0x05;
static const uint8_t MAX86150_FIFO_RD_PTR = 0x06;
static const uint8_t MAX86150_FIFO_DATA = 0x07;
static const uint8_t MAX86150_FIFO_CONFIG = 0x08;
// FIFO Data Control
static const uint8_t MAX86150_FIFO_CONTROL1 = 0x09;
static const uint8_t MAX86150_FIFO_CONTROL2 = 0x0A;
// System Control
static const uint8_t MAX86150_SYS_CONTROL = 0x0D;
// PPG Configuration
static const uint8_t MAX86150_PPG_CONFIG1 = 0x0E;
static const uint8_t MAX86150_PPG_CONFIG2 = 0x0F;
static const uint8_t MAX86150_PPG_PROX_INT_THRESH = 0x10;
// LED Pulse Amplitude
static const uint8_t MAX86150_LED1_PA = 0x11;
static const uint8_t MAX86150_LED2_PA = 0x12;
static const uint8_t MAX86150_LEDP_PA = 0x15;
static const uint8_t MAX86150_LED_RANGE = 0x14;
// ECG Configuration
static const uint8_t MAX86150_ECG_CONFIG1 = 0x3C;
static const uint8_t MAX86150_ECG_CONFIG3 = 0x3E;
// Part ID
static const uint8_t MAX86150_PART_ID = 0xFF;
static const uint8_t MAX86150_PART_ID_VAL = 0x1E;

uint16_t max86150_get_register(const max86150_context_t *ctx, uint8_t reg, uint8_t mask) {
    /**
     * @brief Read register field
     * @return Register value
     */
    uint8_t value = 0;
    ctx->i2c_write_read(ctx->addr, &reg, 1, &value, 1);
    if (mask != 0xFF) { value &= mask; }
    return value;
}

int max86150_set_register(const max86150_context_t *ctx, uint8_t reg, uint8_t value, uint8_t mask) {
    /**
     * @brief Set register field
     * @return 0 if successful
     */
    int err = 0;
    uint16_t i2c_buffer;
    if (mask != 0xFF) {
        value = max86150_get_register(ctx, reg, ~mask) | (value & mask);
    }
    i2c_buffer = (reg << 4) | (value && 0x00FF);
    err = ctx->i2c_write((uint8_t *)&i2c_buffer, 2, ctx->addr);
    return err;
}

#pragma region Status Registers

uint8_t max86150_get_int1(const max86150_context_t *ctx) {
    /**
     * @brief Get interrupt 1 register
     * A_FULL[7] PPG_RDY[6] ALC_OVF[5] PROX_INT[4] PWR_RDY[0]
     * @param  ctx Device context
     * @return register value
     */
    return max86150_get_register(ctx, MAX86150_INT_STAT1, 0xFF);
}

uint8_t max86150_get_int2(const max86150_context_t *ctx) {
    /**
     * @brief Get interrupt 2 register
     * VDD_OOR[7] ECG_RDY[2]
     * @param  ctx Device context
     * @return register value
     */
    return max86150_get_register(ctx, MAX86150_INT_STAT2, 0xFF);
}

void max86150_set_alm_full_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set FIFO almost full interrupt enable flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN1, enable << 7, 0x80);
}

void max86150_set_data_rdy_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set new PPG FIFO data ready interrupt enable flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN1, enable << 6, 0x40);
}

void max86150_set_alc_ovf_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set ambient light cancellation (ALC) overflow interrupt enable flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN1, enable << 5, 0x20);
}

void max86150_set_prox_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set proximity interrupt enable flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN1, enable << 4, 0x10);
}

void max86150_set_vdd_oor_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set VDD Out-of-Range indicator interrupt flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN2, enable << 7, 0x80);
}

void max86150_set_ecg_rdy_int_flag(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set new ECG FIFO data ready interrupt enable flag
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_INT_EN2, enable << 2, 0x04);
}

#pragma endregion

#pragma region FIFO Registers

uint8_t max86150_get_fifo_wr_pointer(const max86150_context_t *ctx) {
    /**
     * @brief Get FIFO write pointer
     * @param  ctx Device context
     * @return write pointer
     */
    return max86150_get_register(ctx, MAX86150_FIFO_WR_PTR, 0x1F);
}

void max86150_set_fifo_wr_pointer(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set FIFO write pointer
     * @param  ctx Device context
     * @param value Write pointer
     *
     */
    max86150_set_register(ctx, MAX86150_FIFO_WR_PTR, value, 0x1F);
}

void max86150_set_fifo_slot(const max86150_context_t *ctx, uint8_t slot, Max86150SlotType type) {
    uint8_t reg = slot & 0x02 ? MAX86150_FIFO_CONTROL2 : MAX86150_FIFO_CONTROL1;
    uint8_t value = slot & 0x01 ? type << 4 : type;
    uint8_t mask = slot & 0x01 ? 0xF0 : 0x0F;
    max86150_set_register(ctx, reg, value, mask);
}

void max86150_set_fifo_slots(const max86150_context_t *ctx, Max86150SlotType slot0, Max86150SlotType slot1, Max86150SlotType slot2, Max86150SlotType slot3) {
    max86150_set_fifo_slot(ctx, 0, slot0);
    max86150_set_fifo_slot(ctx, 1, slot1);
    max86150_set_fifo_slot(ctx, 2, slot2);
    max86150_set_fifo_slot(ctx, 3, slot3);
}

void max86150_disable_slots(const max86150_context_t *ctx) {
     max86150_set_register(ctx, MAX86150_FIFO_CONTROL1, 0x00, 0xFF);
     max86150_set_register(ctx, MAX86150_FIFO_CONTROL2, 0x00, 0xFF);
}

uint32_t max86150_read_fifo_samples(const max86150_context_t *ctx, uint8_t *buffer, uint8_t elementsPerSample) {
    /**
     * @brief Reads all data available in FIFO
     * @param  ctx Device context
     * @param buffer Buffer to store FIFO data. Should be at least 32*3*elementsPerSample (max 384 bytes)
     * @param elementsPerSample Number of elements per sample. Depends on values written to FD1-FD4
     * @return Number of samples read
     *
     */
    uint8_t rdPtr = max86150_get_fifo_rd_pointer(ctx);
    uint8_t wrPtr = max86150_get_fifo_wr_pointer(ctx);
    uint32_t numSamples = rdPtr < wrPtr ? wrPtr - rdPtr : MAX86150_FIFO_DEPTH - rdPtr + wrPtr;
    uint32_t bytesPerSample = 3*elementsPerSample;
    uint32_t numBytes = bytesPerSample*numSamples;
    for (size_t i = 0; i < numSamples; i++){
        ctx->i2c_write_read(ctx->addr, &MAX86150_FIFO_DATA, 1, &buffer[i*bytesPerSample], bytesPerSample);
    }
    return numSamples;
}

uint8_t max86150_get_fifo_overflow_counter(const max86150_context_t *ctx) {
    /**
     * @brief Get FIFO overflow counter
     * @param  ctx Device context
     * @return FIFO overflow counter
     */
    return max86150_get_register(ctx, MAX86150_FIFO_OVERFLOW, 0x1F);
}

uint8_t max86150_get_fifo_rd_pointer(const max86150_context_t *ctx) {
    /**
     * @brief Get FIFO write pointer
     * @param  ctx Device context
     * @return FIFO write pointer
     */
    return max86150_get_register(ctx, MAX86150_FIFO_RD_PTR, 0x1F);
}

void max86150_set_fifo_rd_pointer(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set FIFO read pointer
     * @param  ctx Device context
     */
    max86150_set_register(ctx, MAX86150_FIFO_RD_PTR, value, 0x1F);
}

void max86150_set_almost_full_int_options(const max86150_context_t *ctx, uint8_t options) {
    /**
     * @brief Set FIFO almost full interrupt options
     * @param  ctx Device context
     * @param options 1-bit:
     * 0: A_FULL interrupt does not get cleared by FIFO_DATA register read. It gets cleared by status register read.
     * 1: A_FULL interrupt gets cleared by FIFO_DATA register read or status register read.
     *
     */
    max86150_set_register(ctx, MAX86150_FIFO_CONFIG, options << 6, 0x40);
}

void max86150_set_almost_full_flag_options(const max86150_context_t *ctx, uint8_t options) {
    /**
     * @brief Set FIFO almost full flag options
     * @param  ctx Device context
     * @param options 1-bit:
     * 0: Assert on a_full condition, clear by status reg read, and re-assert on subsequent samples
     * 1: Assert on a_full condition, clear by status reg read, and not re-assert on subsequent samples
     *
     */
    max86150_set_register(ctx, MAX86150_FIFO_CONFIG, options << 5, 0x20);
}

void max86150_set_almost_full_rollover(const max86150_context_t *ctx, uint8_t enable) {
    /**
     * @brief Set whether FIFO rollsover when full
     * @param  ctx Device context
     * @param enable 1-bit:
     * 0: No rollover - FIFO stop on full
     * 1: Rollover - FIFO auto rolls over on full
     *
     */
    max86150_set_register(ctx, MAX86150_FIFO_CONFIG, enable << 4, 0x10);
}

void max86150_set_almost_full_threshold(const max86150_context_t *ctx, uint8_t space) {
    /**
     * @brief Set FIFO almost full value (i.e. how many samples till interrupt is triggered)
     * @param  ctx Device context
     * @param space Remaining FIFO space before intr trigger
     *
     */
    max86150_set_register(ctx, MAX86150_FIFO_CONFIG, space, 0x0F);
}

#pragma endregion

#pragma region System Registers

void max86150_set_fifo_enable(const max86150_context_t *ctx, bool enable) {
    /**
     * @brief Set FIFO enable state.
     * @param  ctx Device context
     * @param enable Enable/disable FIFO. Clears on enable
     */
    max86150_set_register(ctx, MAX86150_SYS_CONTROL, enable << 2, 0x4);
}

void max86150_shutdown(const max86150_context_t *ctx) {
    /**
     * @brief Put chip into power-save mode. Registers retain their values
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_SYS_CONTROL, 0x2, 0x2);
}

void max86150_powerup(const max86150_context_t *ctx) {
    /**
     * @brief Takes chip out ofpower-save mode
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_SYS_CONTROL, 0x0, 0x2);
}

void max86150_reset(const max86150_context_t *ctx) {
    /**
     * @brief This performs full power-on-reset. All registers are reset
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_SYS_CONTROL, 0x1, 0x1);
}

uint8_t max86150_get_part_id(const max86150_context_t *ctx) {
    /**
     * @brief Get part ID. Should be 0x1E for this part
     * @param  ctx Device context
     *
     * @return return
     */
    return max86150_get_register(ctx, MAX86150_PART_ID, 0xFF);
}

#pragma endregion

#pragma region PPG Configuration

void max86150_set_ppg_adc_range(const max86150_context_t *ctx, uint8_t range) {
    /**
     * @brief Set PPG ADC range
     * @param  ctx Device context
     * @param value 2-bit | Full scale = 2**(14+value) nA, LSB = 7.8125 * (2 ** value)
     *
     */
    max86150_set_register(ctx, MAX86150_PPG_CONFIG1, range << 6, 0xC0);
}

void max86150_set_ppg_sample_rate(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set PPG sample rate and pulse rate.
     * NOTE: If rate cant be acheived, then highest available will be selected
     * @param  ctx Device context
     * @param value 4-bit |
     * Samples/sec: [10, 20, 50, 84, 100, 200, 400, 800, 1000, 1600, 3200, 10, 20, 50, 84, 100]
     *  Pulses/sec: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
     *
     */
    max86150_set_register(ctx, MAX86150_PPG_CONFIG1, value << 2, 0x3C);
}

void max86150_set_ppg_pulse_width(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set PPG pulse width
     * @param  ctx Device context
     * @param value 2-bit | 0: 50 1: 100 2: 200 3: 400 (us)
     *
     */
    max86150_set_register(ctx, MAX86150_PPG_CONFIG1, value, 0x03);
}

void max86150_set_ppg_sample_average(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set PPG sample averaging. Will average and decimate adjacent samples
     * @param  ctx Device context
     * @param value avg = min(2**value, 32)
     *
     */
    max86150_set_register(ctx, MAX86150_PPG_CONFIG2, value, 0x07);
}

void max86150_set_proximity_threshold(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Sets the IR ADC count that triggers the beginning of the PPG mode
     * @param  ctx Device context
     *
     */
    max86150_set_register(ctx, MAX86150_PPG_PROX_INT_THRESH, value, 0xFF);
}

#pragma endregion

#pragma region LED Configuration

void max86150_set_led_pulse_amplitude(const max86150_context_t *ctx, uint8_t led, uint8_t value) {
    /**
     * @brief
     * @param  ctx Device context
     * @param led 0: LED1 (IR), 1: LED2 (RED), 2: LEDP (proximity)
     * @param value 7-bit | Pulse amplitude = 0.2*value*(LEDn_RANGE+1) (mA)
     *
     */
    uint8_t reg = MAX86150_LED1_PA;
    switch (led) {
        case 0:
            reg = MAX86150_LED1_PA;
            break;
        case 1:
            reg = MAX86150_LED2_PA;
            break;
        case 2:
            reg = MAX86150_LEDP_PA;
            break;
        default:
            break;
    }
    max86150_set_register(ctx, reg, value & 0x7F, 0xFF);
}

void max86150_set_led_current_range(const max86150_context_t *ctx, uint8_t led, uint8_t value) {
    /**
     * @brief Set LED current range
     * @param  ctx Device context
     * @param led 0: LED1 (IR), 1: LED2 (RED)
     * @param value 2-bit | 0: 50, 1: 100 (mA)
     *
     */
    uint8_t mask = led == 0 ? 0x3 : 0xC;
    uint8_t val = led == 0 ? value : value << 2;
    max86150_set_register(ctx, MAX86150_LED_RANGE, val, mask);
}

#pragma endregion

#pragma region ECG Configuration

void max86150_set_ecg_sample_rate(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief
     * @param  ctx Device context
     * @param value 3-bit
     * Sample Rate Table (Hz)
     * VALUE    FS      FBW_70  FBW_90
     * 0        1600    420     232
     * 1        800     210     116
     * 2        400     105     58
     * 3        200     52      29
     * 4        3200    840     464
     * 5        1600    420     232
     * 6        800     210     116
     * 7        400     105     58
     */
    max86150_set_register(ctx, MAX86150_ECG_CONFIG1, value, 0x07);
}

void max86150_set_ecg_pga_gain(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set ECG PGA gain. Gain = 2**value (V/V)
     * @param  ctx Device context
     * @param value 2-bit
     *
     */
    max86150_set_register(ctx, MAX86150_ECG_CONFIG3, value << 2, 0x0C);
}

void max86150_set_ecg_ia_gain(const max86150_context_t *ctx, uint8_t value) {
    /**
     * @brief Set ECG instrument amplifier gain.
     * @param  ctx Device context
     * @param value
     * Gain table: 0: 5, 1: 9.5, 2: 20, 3: 50 (V/V)
     *
     */
    max86150_set_register(ctx, MAX86150_ECG_CONFIG3, value, 0x03);
}

#pragma endregion
