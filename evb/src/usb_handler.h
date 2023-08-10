/**
 * @file usb_handler.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Handle USB events
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __USB_HANDLER_H
#define __USB_HANDLER_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    bool available;
} usb_config_t;

uint32_t
init_usb_handler(usb_config_t *ctx);
void
usb_update_state();
#endif
