//*****************************************************************************
//
//! @file ns_io_i2c.c
//!
//! @brief Utility for performing I2C transactions.
//!
//! Purpose: Reading EVB buttons
//!
//
//*****************************************************************************

//*****************************************************************************
//
// ${copyright}
//
// This is part of revision ${version} of the AmbiqSuite Development Package.
//
//*****************************************************************************

#include "am_bsp.h"
#include "am_mcu_apollo.h"
#include "am_util.h"
#include "ns_io_i2c.h"

#ifndef NUM_I2C_DEVICES
    #define NUM_I2C_DEVICES 2
#endif

am_hal_iom_config_t iomConfigs[NUM_I2C_DEVICES] = {{
    .eInterfaceMode = AM_HAL_IOM_I2C_MODE,
    .ui32ClockFreq = AM_HAL_IOM_100KHZ,
    .pNBTxnBuf = NULL,
    .ui32NBTxnBufLength = 0
}};

void *iomHandles[NUM_I2C_DEVICES];

void ns_io_i2c_init(ns_i2c_config_t *cfg) {
    am_hal_pwrctrl_periph_enable((am_hal_pwrctrl_periph_e)(AM_HAL_PWRCTRL_PERIPH_IOM0 + cfg->device));
    iomConfigs[cfg->device].ui32ClockFreq = cfg->speed;
    //
    if (am_hal_iom_initialize(cfg->device, &iomConfigs[cfg->device]) ||
        am_hal_iom_power_ctrl(iomHandles[cfg->device], AM_HAL_SYSCTRL_WAKE, false) ||
        am_hal_iom_configure(iomHandles[cfg->device], &iomConfigs[cfg->device]) ||
        am_hal_iom_enable(iomHandles[cfg->device])
    ) {
        return 1;
    }
    am_bsp_iom_pins_enable(cfg->device, AM_HAL_IOM_I2C_MODE);
    // Override pullup
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_IOM1_SCL,  g_AM_BSP_GPIO_IOM1_nopullup_SCL);
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_IOM1_SDA,  g_AM_BSP_GPIO_IOM1_nopullup_SDA);
    return 0;
}


int ns_io_i2c_write_read(ns_i2c_config_t *cfg, uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    am_hal_iom_transfer_t       Transaction;
    Transaction.uPeerInfo.ui32I2CDevAddr = addr;
    Transaction.ui32InstrLen    = num_write;
    Transaction.ui64Instr       = write_buf;
    Transaction.eDirection      = AM_HAL_IOM_RX;
    Transaction.ui32NumBytes    = num_read;
    Transaction.pui32RxBuffer   = read_buf;
    Transaction.bContinue       = false;
    Transaction.ui8RepeatCount  = 0;
    Transaction.ui32PauseCondition = 0;
    Transaction.ui32StatusSetClr = 0;
    return am_hal_iom_blocking_transfer(iomHandles[cfg->device], &Transaction);
}

int ns_io_i2c_read(ns_i2c_config_t *cfg, uint8_t *buf, uint32_t num_bytes, uint16_t addr) {
    am_hal_iom_transfer_t       Transaction;
    Transaction.uPeerInfo.ui32I2CDevAddr = addr;
    Transaction.ui32InstrLen    = 0;
    Transaction.eDirection      = AM_HAL_IOM_RX;
    Transaction.ui32NumBytes    = num_bytes;
    Transaction.pui32RxBuffer   = buf;
    Transaction.bContinue       = false;
    Transaction.ui8RepeatCount  = 0;
    Transaction.ui32PauseCondition = 0;
    Transaction.ui32StatusSetClr = 0;
    return am_hal_iom_blocking_transfer(iomHandles[cfg->device], &Transaction);
}

int ns_io_i2c_write(ns_i2c_config_t *cfg, const uint8_t *buf, uint32_t num_bytes, uint16_t addr) {
    am_hal_iom_transfer_t       Transaction;
    Transaction.uPeerInfo.ui32I2CDevAddr = addr;
    Transaction.ui32InstrLen    = 0;
    Transaction.eDirection      = AM_HAL_IOM_TX;
    Transaction.ui32NumBytes    = num_bytes;
    Transaction.pui32TxBuffer   = buf;
    Transaction.bContinue       = false;
    Transaction.ui8RepeatCount  = 0;
    Transaction.ui32PauseCondition = 0;
    Transaction.ui32StatusSetClr = 0;
    return am_hal_iom_blocking_transfer(iomHandles[cfg->device], &Transaction);
}
