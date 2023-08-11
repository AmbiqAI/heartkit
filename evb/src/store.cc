#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "arm_math.h"

// Local
#include "constants.h"
#include "physiokit.h"
#include "store.h"

///////////////////////////////////////////////////////////////////////////////
// EVB Configuration
///////////////////////////////////////////////////////////////////////////////

const ns_power_config_t nsPwrCfg = {.api = &ns_power_V1_0_0,
                                    .eAIPowerMode = NS_MINIMUM_PERF,
                                    .bNeedAudAdc = false,
                                    .bNeedSharedSRAM = false,
                                    .bNeedCrypto = true,
                                    .bNeedBluetooth = false,
                                    .bNeedUSB = true,
                                    .bNeedIOM = true,
                                    .bNeedAlternativeUART = false,
                                    .b128kTCM = false};

int volatile sensorCollectBtnPressed = false;
int volatile clientCollectBtnPressed = false;
ns_button_config_t nsBtnCfg = {.api = &ns_button_V1_0_0,
                               .button_0_enable = true,
                               .button_1_enable = true,
                               .button_0_flag = &sensorCollectBtnPressed,
                               .button_1_flag = &clientCollectBtnPressed};

ns_rpc_config_t nsRpcCfg = {.api = &ns_rpc_gdo_V1_0_0,
                            .mode = NS_RPC_GENERICDATA_CLIENT,
                            .sendBlockToEVB_cb = NULL,
                            .fetchBlockFromEVB_cb = NULL,
                            .computeOnEVB_cb = NULL};

ns_i2c_config_t nsI2cCfg = {.api = &ns_i2c_V1_0_0, .iom = I2C_IOM};

ns_core_config_t nsCoreCfg = {.api = &ns_core_V1_0_0};

usb_config_t usbCfg = {.available = false};

///////////////////////////////////////////////////////////////////////////////
// Sensor Configuration
///////////////////////////////////////////////////////////////////////////////

static inline int
max86150_write_read(uint16_t addr, const void *write_buf, size_t num_write, void *read_buf, size_t num_read) {
    return ns_i2c_write_read(&nsI2cCfg, addr, write_buf, num_write, read_buf, num_read);
}
static inline int
max86150_read(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_i2c_read(&nsI2cCfg, buf, num_bytes, addr);
}
static inline int
max86150_write(const void *buf, uint32_t num_bytes, uint16_t addr) {
    return ns_i2c_write(&nsI2cCfg, buf, num_bytes, addr);
}

max86150_context_t maxCtx = {
    .addr = MAX86150_ADDR,
    .i2c_write_read = max86150_write_read,
    .i2c_read = max86150_read,
    .i2c_write = max86150_write,
};

max86150_slot_type maxSlotsCfg[] = {Max86150SlotEcg, Max86150SlotOff, Max86150SlotOff, Max86150SlotOff};

max86150_config_t maxCfg = {
    .numSlots = 1,
    .fifoSlotConfigs = maxSlotsCfg,
    .fifoRolloverFlag = 1,
    .ppgSampleAvg = 2,          // Avg 4 samples
    .ppgAdcRange = 2,           // 16,384 nA Scale
    .ppgSampleRate = 5,         // 200 Hz
    .ppgPulseWidth = 1,         // 100 us
    .led0CurrentRange = 0,      // IR LED 50 mA range
    .led1CurrentRange = 0,      // RED LED 50 mA range
    .led2CurrentRange = 0,      // Pilot LED 50 mA range
    .led0PulseAmplitude = 0x32, // IR LED 20 mA 0x32
    .led1PulseAmplitude = 0x32, // RED LED 20 mA 0x32
    .led2PulseAmplitude = 0x32, // AMB LED 20 mA 0x32
    .ecgSampleRate = 3,         // Fs = 200 Hz
    .ecgIaGain = 2,             // 9.5 V/V
    .ecgPgaGain = 3             // 8 V/V
};

hk_sensor_t sensorCtx = {
    .maxCtx = &maxCtx,
    .maxCfg = &maxCfg,
};

///////////////////////////////////////////////////////////////////////////////
// Preprocess Configuration
///////////////////////////////////////////////////////////////////////////////

float32_t ecgSosState[4 * ECG_SOS_LEN] = {0};
// print(generate_arm_biquad_sos(0.5, 30, 200, order=3, var_name="ecgBandPass"))
float32_t ecgSos[5 * ECG_SOS_LEN] = {0.047553547828447645,
                                     0.09510709565689529,
                                     0.047553547828447645,
                                     0.850641691932028,
                                     -0.4354096596521514,
                                     1.0,
                                     0.0,
                                     -1.0,
                                     1.3229937537161818,
                                     -0.3336252395631515,
                                     1.0,
                                     -2.0,
                                     1.0,
                                     1.9844157739727124,
                                     -0.9846642712579249};
arm_biquad_casd_df1_inst_f32 ecgFilterCtx = {.numStages = ECG_SOS_LEN, .pState = ecgSosState, .pCoeffs = ecgSos};

// print(generate_arm_biquad_sos(10, 30, 200, order=3, var_name="qrsBandPass"))
static float32_t qrsSosState[4 * QRS_SOS_LEN] = {0};
static float32_t qrsSos[5 * QRS_SOS_LEN] = {0.018098933007514438,
                                            0.036197866015028876,
                                            0.018098933007514438,
                                            1.2840790438404122,
                                            -0.5095254494944287,
                                            1.0,
                                            0.0,
                                            -1.0,
                                            1.0263940971801464,
                                            -0.6507946692078062,
                                            1.0,
                                            -2.0,
                                            1.0,
                                            1.7386603322829222,
                                            -0.8385491481879397};
arm_biquad_casd_df1_inst_f32 qrsFilterCtx = {.numStages = QRS_SOS_LEN, .pState = qrsSosState, .pCoeffs = qrsSos};

///////////////////////////////////////////////////////////////////////////////
// HeartKit Configuration
///////////////////////////////////////////////////////////////////////////////

const char *HK_RHYTHM_LABELS[] = {"NSR", "AFIB/AFL", "AFIB/AFL"};
const char *HK_BEAT_LABELS[] = {"NORMAL", "PAC", "PVC"};
const char *HK_HEART_RATE_LABELS[] = {"NORMAL", "TACHYCARDIA", "BRADYCARDIA"};
const char *HK_SEGMENT_LABELS[] = {"NONE", "P-WAVE", "QRS", "T-WAVE"};

static float32_t pkArena[3 * HK_DATA_LEN];
ecg_peak_f32_t qrsFindPeakCtx = {.qrsWin = 0.1,
                                 .avgWin = 1.0,
                                 .qrsPromWeight = 1.5,
                                 .qrsMinLenWeight = 0.4,
                                 .qrsDelayWin = 0.3,
                                 .sampleRate = SAMPLE_RATE,
                                 .state = pkArena};
static uint32_t hkQrsPeaks[HK_PEAK_LEN];
static uint32_t hkRRIntervals[HK_PEAK_LEN];
static uint8_t hkRRMask[HK_PEAK_LEN];

static float32_t hkRawData[SENSOR_LEN + SENSOR_RATE];
static float32_t hkEcgData[HK_DATA_LEN];
static float32_t hkQrsData[HK_DATA_LEN];
static float32_t hkBufData[HK_DATA_LEN];

static uint8_t hkSegMask[HK_DATA_LEN];
static hrv_td_metrics_t hkHrvMetrics;
static hk_result_t hkResults;

///////////////////////////////////////////////////////////////////////////////
// App Configuration
///////////////////////////////////////////////////////////////////////////////

qrs_context_t qrsCtx = {
    .qrsFindPeakCtx = &qrsFindPeakCtx,
    .numQrsPeaks = 0,
    .qrsPeaks = hkQrsPeaks,
    .rrIntervals = hkRRIntervals,
    .rrMask = hkRRMask,
};

hk_app_store_t hkStore = {.numSamples = 0,
                          .state = IDLE_STATE,
                          .collectMode = NO_DATA_COLLECT,
                          .rawData = hkRawData,
                          .ecgData = hkEcgData,
                          .qrsData = hkQrsData,
                          .bufData = hkBufData,
                          .segMask = hkSegMask,
                          .hrvMetrics = &hkHrvMetrics,
                          .results = &hkResults,
                          .errorCode = 0};
