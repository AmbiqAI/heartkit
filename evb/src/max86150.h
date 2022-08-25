#ifndef __MAX86150_H
#define __MAX86150_H

#include <cstdint>

enum Max86150SlotType {
    Max86150SlotOff = 0,
    Max86150SlotPpgLed1 = 1,
    Max86150SlotPpgLed2 = 2,
    Max86150SlotPilotLed1 = 5,
    Max86150SlotPilotLed2 = 6,
    Max86150SlotEcg = 9
}; typedef enum Max86150SlotType Max86150SlotType;

uint16_t max86150_get_register(const struct device *dev, uint16_t addr, uint8_t reg, uint8_t mask = NULL);
int max86150_set_register(const struct device *dev, uint16_t addr, uint8_t reg, uint8_t value, uint8_t mask = NULL);

// Interrupts
uint8_t max86150_get_int1(const struct device *dev, uint16_t addr);
uint8_t max86150_get_int2(const struct device *dev, uint16_t addr);

void max86150_set_alm_full_int_flag(const struct device *dev, uint16_t addr, bool enable);
void max86150_set_data_rdy_int_flag(const struct device *dev, uint16_t addr, bool enable);
void max86150_set_alc_ovf_int_flag(const struct device *dev, uint16_t addr, bool enable);
void max86150_set_prox_int_flag(const struct device *dev, uint16_t addr, bool enable);
void max86150_set_vdd_oor_int_flag(const struct device *dev, uint16_t addr, bool enable);
void max86150_set_ecg_rdy_int_flag(const struct device *dev, uint16_t addr, bool enable);

uint8_t max86150_get_fifo_wr_pointer(const struct device *dev, uint16_t addr);
void max86150_set_fifo_wr_pointer(const struct device *dev, uint16_t addr, uint8_t value);
uint8_t max86150_set_fifo_slot(const struct device *dev, uint16_t addr, uint8_t slot, Max86150SlotType type);
uint8_t max86150_set_fifo_slots(const struct device *dev, uint16_t addr, Max86150SlotType slot0, Max86150SlotType slot1, Max86150SlotType slot2, Max86150SlotType slot3);
uint8_t max86150_disable_slots(const struct device *dev, uint16_t addr);
uint32_t max86150_read_fifo_samples(const struct device *dev, uint16_t addr, uint8_t *buffer, uint8_t elementsPerSample = 3);
uint8_t max86150_get_fifo_overflow_counter(const struct device *dev, uint16_t addr);
uint8_t max86150_get_fifo_rd_pointer(const struct device *dev, uint16_t addr);
void max86150_set_fifo_rd_pointer(const struct device *dev, uint16_t addr, uint8_t value);

void max86150_set_almost_full_int_options(const struct device *dev, uint16_t addr, uint8_t options);
void max86150_set_almost_full_flag_options(const struct device *dev, uint16_t addr, uint8_t options);
void max86150_set_almost_full_rollover(const struct device *dev, uint16_t addr, uint8_t enable);
void max86150_set_almost_full_threshold(const struct device *dev, uint16_t addr, uint8_t space);
void max86150_set_fifo_enable(const struct device *dev, uint16_t addr, bool enable);

void max86150_powerup(const struct device *dev, uint16_t addr);
void max86150_shutdown(const struct device *dev, uint16_t addr);
void max86150_reset(const struct device *dev, uint16_t addr);

void max86150_set_ppg_adc_range(const struct device *dev, uint16_t addr, uint8_t range);
void max86150_set_ppg_sample_rate(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_ppg_pulse_width(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_ppg_sample_average(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_proximity_threshold(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_led_pulse_amplitude(const struct device *dev, uint16_t addr, uint8_t led, uint8_t value);
void max86150_set_led_current_range(const struct device *dev, uint16_t addr, uint8_t led, uint8_t value);
void max86150_set_ecg_sample_rate(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_ecg_pga_gain(const struct device *dev, uint16_t addr, uint8_t value);
void max86150_set_ecg_ia_gain(const struct device *dev, uint16_t addr, uint8_t value);
uint8_t max86150_get_part_id(const struct device *dev, uint16_t addr);

#endif
