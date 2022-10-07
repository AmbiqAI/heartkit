#ifndef __MAIN_H
#define __MAIN_H

enum AppState {
    IDLE_STATE,
    START_COLLECT_STATE,
    COLLECT_STATE,
    STOP_COLLECT_STATE,
    PREPROCESS_STATE,
    INFERENCE_STATE,
    DISPLAY_STATE,
    FAIL_STATE
}; typedef enum AppState AppState;

enum DataCollectMode {
    SENSOR_DATA_COLLECT,
    CLIENT_DATA_COLLECT
}; typedef enum DataCollectMode DataCollectMode;


void setup(void);
void loop(void);

#endif
