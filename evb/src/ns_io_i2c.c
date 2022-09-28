//*****************************************************************************
//
//! @file ns_io_i2c.c
//!
//! @brief Utility for performing I2C transactions.
//!
//! Purpose: Perform I2C transactions
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

am_hal_iom_config_t iomConfigs[NUM_I2C_DEVICES];

void *iomHandles[NUM_I2C_DEVICES];

int ns_io_i2c_init(ns_i2c_config_t *cfg) {
    /**
     * @brief Initialize I2C interface
     * @param cfg I2C configuration pointer
     *
     */
    am_hal_pwrctrl_periph_enable((am_hal_pwrctrl_periph_e)(AM_HAL_PWRCTRL_PERIPH_IOM0 + cfg->device));
    iomConfigs[cfg->i2cBus].eInterfaceMode = AM_HAL_IOM_I2C_MODE;
    iomConfigs[cfg->i2cBus].pNBTxnBuf = NULL;
    iomConfigs[cfg->i2cBus].ui32NBTxnBufLength = 0;
    iomConfigs[cfg->i2cBus].ui32ClockFreq = cfg->speed;
    if (am_hal_iom_initialize(cfg->device, &(iomHandles[cfg->i2cBus])) ||
        am_hal_iom_power_ctrl(iomHandles[cfg->i2cBus], AM_HAL_SYSCTRL_WAKE, false) ||
        am_hal_iom_configure(iomHandles[cfg->i2cBus], &(iomConfigs[cfg->i2cBus])) ||
        am_hal_iom_enable(iomHandles[cfg->i2cBus])
    ) {
        return NS_I2C_STATUS_ERROR;
    }
    am_bsp_iom_pins_enable(cfg->device, AM_HAL_IOM_I2C_MODE);
    // Override pullup
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_IOM1_SCL,  g_AM_BSP_GPIO_IOM1_nopullup_SCL);
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_IOM1_SDA,  g_AM_BSP_GPIO_IOM1_nopullup_SDA);
    return NS_I2C_STATUS_SUCCESS;
}

int ns_io_i2c_read(ns_i2c_config_t *cfg, const void *buf, uint32_t num_bytes, uint16_t addr) {
    /**
     * @brief Perform low-level I2C read using IOM transfer
     * @param cfg I2C configuration
     * @param buf Buffer to store read bytes
     * @param num_bytes Number of bytes to read
     * @param addr I2C device address
     *
     */
    am_hal_iom_transfer_t       Transaction;
    Transaction.ui8Priority     = 1;
    Transaction.ui32InstrLen    = 0;
    Transaction.ui64Instr       = 0;
    Transaction.eDirection      = AM_HAL_IOM_RX;
    Transaction.ui32NumBytes    = num_bytes;
    Transaction.pui32RxBuffer   = (uint32_t *)buf;
    Transaction.bContinue       = false;
    Transaction.ui8RepeatCount  = 0;
    Transaction.ui32PauseCondition = 0;
    Transaction.ui32StatusSetClr = 0;
    Transaction.uPeerInfo.ui32I2CDevAddr = addr;
    if (am_hal_iom_blocking_transfer(iomHandles[cfg->i2cBus], &Transaction) != AM_HAL_STATUS_SUCCESS) {
        return NS_I2C_STATUS_ERROR;
    }
    return NS_I2C_STATUS_SUCCESS;
}

int ns_io_i2c_write(ns_i2c_config_t *cfg, const void *buf, uint32_t num_bytes, uint16_t addr) {
    /**
     * @brief Perform low-level I2C write using IOM transfer
     * @param cfg I2C configuration
     * @param buf Buffer of bytes to write
     * @param num_bytes Number of bytes to write
     * @param addr I2C device address
     *
     */
    int err;
    am_hal_iom_transfer_t       Transaction;
    Transaction.ui8Priority     = 1;
    Transaction.ui32InstrLen    = 0;
    Transaction.ui64Instr       = 0;
    Transaction.eDirection      = AM_HAL_IOM_TX;
    Transaction.ui32NumBytes    = num_bytes;
    Transaction.pui32TxBuffer   = (uint32_t *)buf;
    Transaction.bContinue       = false;
    Transaction.ui8RepeatCount  = 0;
    Transaction.ui32PauseCondition = 0;
    Transaction.ui32StatusSetClr = 0;
    Transaction.uPeerInfo.ui32I2CDevAddr = addr;
    err = am_hal_iom_blocking_transfer(iomHandles[cfg->i2cBus], &Transaction);
    if (err != AM_HAL_STATUS_SUCCESS) {
        return NS_I2C_STATUS_ERROR;
    }
    return NS_I2C_STATUS_SUCCESS;
}

uint32_t ns_io_i2c_xfer(ns_i2c_config_t *cfg, ns_i2c_msg *msgs, size_t num_msgs) {
    /**
     * @brief Perform sequence of low-level I2C transfers (similar to Linux)
     * @param cfg I2C configuration
     * @param msgs I2C messages to transfer
     * @param num_msgs Number of I2C messsages
     */

    ns_i2c_msg *msg;
    uint32_t msg_len = 0;
    for (size_t i = 0; i < num_msgs; i++){
        msg = &msgs[i];

        if (msg->flags == NS_I2C_XFER_RD) {
            ns_io_i2c_read(cfg, msg->buf, msg->len, msg->addr);
            msg_len += msg->len;
        }
        else if (msg->flags == NS_I2C_XFER_WR) {
            ns_io_i2c_write(cfg, msg->buf, msg->len, msg->addr);
            msg_len += (msg->len - 1);
        }
    }
    return msg_len;
}

int ns_io_i2c_write_read(ns_i2c_config_t *cfg, uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    /**
     * @brief Perform low-level I2C write followed by immediate read using IOM transfer
     * @param cfg I2C configuration
     * @param write_buf Write buffer
     * @param num_write Number of bytes to write
     * @param read_buf Read buffer
     * @param num_read Number of bytes to read
     * @param addr I2C device address
     */
    ns_io_i2c_write(cfg, write_buf, num_write, addr);
    return ns_io_i2c_read(cfg, read_buf, num_read, addr);
}
