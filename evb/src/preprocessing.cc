// #include <cstdio>
// #include <cstdlib>
#include "arm_math.h"
#include "preprocessing.h"

#define NUM_STAGE_IIR 2
#define NUM_ORDER_IIR (NUM_STAGE_IIR * 2)
#define NUM_STD_COEFFS 5 // b0, b1, b2, a1, a2
#define NUM_COMP_COEFFS 8
static float32_t iirState[NUM_ORDER_IIR];
static float32_t iirCoeffs[NUM_STAGE_IIR * NUM_STD_COEFFS] = { // Bandpass 0.5 - 30 Hz
    // [b0, b1, b2, a1, a2]
    0.044157196394093684, 0.08831439278818737, 0.044157196394093684, 1.3309555967298963, -0.5095097187836408,
    // [b0, b1, b2, a1, a2]
    1.0, -2.0, 1.0, 1.98226563125227, -0.9824297820610058
};

arm_biquad_cascade_df2T_instance_f32 iirInst;

void init_preprocess() {
    /**
     * @brief Initialize preprocessing block
     *
     */
    arm_biquad_cascade_df2T_init_f32(&iirInst, NUM_STAGE_IIR, iirCoeffs, iirState);
}

int bandpass_filter(float32_t* pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Perform bandpass filter (0.5-30 Hz) on signal
     */
    arm_biquad_cascade_df2T_f32(&iirInst, pSrc, pResult, blockSize);
    return 0;
}

int standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Standardize input y = (x - mu) / std. Provides safegaurd against small st devs
     *
     */
    float32_t mu, std;
    arm_mean_f32(pSrc, blockSize, &mu);
    arm_std_f32(pSrc, blockSize, &std);
    std = std < 1e-6 ? 1 : std;
    arm_offset_f32(pSrc, -mu, pResult, blockSize);
    arm_scale_f32(pResult, 1.0f/std, pResult, blockSize);
    return 0;
}
