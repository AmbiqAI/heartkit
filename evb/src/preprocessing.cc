#include <cstdio>
#include "arm_math.h"

#define ECG_FILTER_LEN 128
#define BLOCK_SIZE 32
#define NUM_BLOCKS (ECG_FILTER_LEN / BLOCK_SIZE)

#define NUM_STAGE_IIR 2
#define NUM_ORDER_IIR (NUM_STAGE_IIR * 2)
#define NUM_STD_COEFFS 5 // b0, b1, b2, a1, a2

static float32_t iirState[NUM_ORDER_IIR];
static float32_t iirCoeffs[NUM_STAGE_IIR * NUM_STD_COEFFS] = {
    // [b0, b1, b2, a1, a2]
    0.044157196394093684, 0.08831439278818737, 0.044157196394093684, 1.3309555967298963, -0.5095097187836408,
    // [b0, b1, b2, a1, a2]
    1.0, -2.0, 1.0, 1.98226563125227, -0.9824297820610058
};

float32_t ecgFilterInputF32[ECG_FILTER_LEN];
float32_t ecgFilterOutputF32[ECG_FILTER_LEN];

arm_biquad_cascade_df2T_instance_f32 iirInst;

int ecg_filter_init() {
    arm_biquad_cascade_df2T_init_f32(&iirInst, NUM_STAGE_IIR, &iirCoeffs[0], &iirState[0]);
    return 0;
}

int ecg_filter(q15_t* input, q15_t *output, size_t numSamples) {
    arm_q15_to_float(input, ecgFilterInputF32, numSamples);
    arm_biquad_cascade_df2T_f32(&iirInst, ecgFilterInputF32, ecgFilterOutputF32, numSamples);
    arm_float_to_q15(ecgFilterOutputF32 ,output, numSamples);
    return 0;
}
