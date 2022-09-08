//*****************************************************************************
//
//! @file ns_io_i2c.h
//!
//! @brief Utility for performing I2C transactions.
//!
//! Purpose: Perform I2C transactions
//
//*****************************************************************************

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
// This is part of revision release_sdk_4_0_1-bef824fa27 of the AmbiqSuite Development Package.
//
//*****************************************************************************
#ifndef NS_IO_I2C
#define NS_IO_I2C

#ifdef __cplusplus
extern "C"
{
#endif

#define NS_I2C_XFER_WR 0x0000
#define NS_I2C_XFER_RD (1u << 0)

typedef enum {
    NS_I2C_STATUS_SUCCESS,
    NS_I2C_STATUS_ERROR
} ns_i2c_status_t;

typedef struct {
    uint32_t i2cBus;
    uint32_t device;
    uint32_t speed;
} ns_i2c_config_t;

typedef struct {
    uint16_t addr;
    uint16_t flags;
    uint16_t len;
    uint8_t  *buf;
} ns_i2c_msg;

int ns_io_i2c_init(ns_i2c_config_t *cfg);
int ns_io_i2c_read(ns_i2c_config_t *cfg, const void *buf, uint32_t num_bytes, uint16_t addr);
int ns_io_i2c_write(ns_i2c_config_t *cfg, const void *buf, uint32_t num_bytes, uint16_t addr);
uint32_t ns_io_i2c_xfer(ns_i2c_config_t *cfg, ns_i2c_msg *msgs, size_t num_msgs);
int ns_io_i2c_write_read(ns_i2c_config_t *cfg, uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read);


#ifdef __cplusplus
}
#endif

#endif // NS_IO_I2C
