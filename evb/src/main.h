/**
 * @file main.h
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Main application
 * @version 1.0
 * @date 2023-11-09
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __MAIN_H
#define __MAIN_H

enum AppState { IDLE_STATE, INFERENCE_STATE, FAIL_STATE };
typedef enum AppState AppState;

void
setup(void);
void
loop(void);

#endif
