/**
 * @file constants.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Global app constants
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __HK_CONSTANTS_H
#define __HK_CONSTANTS_H

#define GPIO_TRIGGER 22

#define RPC_BUF_LEN (128)
#define USB_RX_BUFSIZE 4096
#define USB_TX_BUFSIZE 4096

#define MAX_ARENA_SIZE (120)
#define MAX_MODEL_SIZE (80)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif // __HK_CONSTANTS_H
