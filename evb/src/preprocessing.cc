/**
 * @file preprocessing.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Perform preprocessing of sensor data (standardize and bandpass filter)
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "preprocessing.h"
#include "arm_math.h"

#define NUM_STAGE_IIR 2
#define NUM_ORDER_IIR (NUM_STAGE_IIR * 2)
#define NUM_STD_COEFFS 5
#define NUM_COMP_COEFFS 8
static float32_t iirState[NUM_ORDER_IIR];
static float32_t iirCoeffs[NUM_STAGE_IIR * NUM_STD_COEFFS] = {
    // Bandpass 0.5 - 40 Hz [b0, b1, b2, a1, a2]
    0.1424442454075173, 0.2848884908150346, 0.1424442454075173, 0.6856000535320614, -0.26069603227349614, 1.0, -2.0, 1.0,
    1.9822347503854438, -0.9823948462782603};

arm_biquad_cascade_df2T_instance_f32 iirInst;

int
init_preprocess() {
    /**
     * @brief Initialize preprocessing block
     *
     */
    arm_biquad_cascade_df2T_init_f32(&iirInst, NUM_STAGE_IIR, iirCoeffs, iirState);
    return 0;
}

int
bandpass_filter(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Perform bandpass filter (0.5-40 Hz) on signal
     */
    arm_biquad_cascade_df2T_f32(&iirInst, pSrc, pResult, blockSize);
    return 0;
}

int
standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Standardize input y = (x - mu) / std. Provides safegaurd against small st devs
     *
     */
    float32_t mu, std;
    arm_mean_f32(pSrc, blockSize, &mu);
    arm_std_f32(pSrc, blockSize, &std);
    std = std < 1e-6 ? 1 : std;
    arm_offset_f32(pSrc, -mu, pResult, blockSize);
    arm_scale_f32(pResult, 1.0f / std, pResult, blockSize);
    return 0;
}
