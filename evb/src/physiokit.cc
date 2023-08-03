#include "arm_math.h"
#include "ns_ambiqsuite_harness.h"
#include <math.h>

#include "physiokit.h"

uint32_t
pk_mean(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Compute mean of signal
     *
     */
    arm_mean_f32(pSrc, blockSize, pResult);
    return 0;
}

uint32_t
pk_std(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Compute standard deviation of signal
     *
     */
    arm_std_f32(pSrc, blockSize, pResult);
    return 0;
}

uint32_t
pk_gradient(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Compute gradient of signal using forward, centered, and backward difference
     *
     */
    for (size_t i = 1; i < blockSize - 1; i++) {
        pResult[i] = (pSrc[i + 1] - pSrc[i - 1]) / 2.0;
    }
    // Edge cases: Use forward and backward difference
    pResult[0] = (-3 * pSrc[0] + 4 * pSrc[1] - pSrc[2]) / 2.0;
    pResult[blockSize - 1] = (3 * pSrc[blockSize - 1] - 4 * pSrc[blockSize - 2] + pSrc[blockSize - 3]) / 2.0;
    return 0;
}

uint32_t
pk_rms(float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Compute root mean square of signal
     *
     */
    arm_rms_f32(pSrc, blockSize, pResult);
    return 0;
}

uint32_t
pk_standardize(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t epsilon) {
    /**
     * @brief Standardize signal: y = (x - mu) / std. Provides safegaurd against small st devs
     *
     */
    float32_t mu, std;
    pk_mean(pSrc, &mu, blockSize);
    pk_std(pSrc, &std, blockSize);
    std = std + epsilon;
    arm_offset_f32(pSrc, -mu, pResult, blockSize);
    arm_scale_f32(pResult, 1.0f / std, pResult, blockSize);
    return 0;
}

uint32_t
pk_resample_signal(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, uint32_t upSample, uint32_t downSample) {
    /**
     * @brief Resample signal by upsampling followed by downsamping
     *
     */
    return 1;
}

uint32_t
pk_linear_downsample(float32_t *pSrc, uint32_t srcSize, uint32_t srcFs, float32_t *pRst, uint32_t rstSize, uint32_t rstFs) {
    /**
     * @brief Basic downsampling using linear interpolation
     *
     */
    float32_t xi, yl, yr;
    uint32_t xl, xr;
    float32_t ratio = ((float32_t)srcFs) / rstFs;
    for (size_t i = 0; i < rstSize; i++) {
        xi = i * ratio;
        xl = floorf(xi);
        xr = ceilf(xi);
        yl = pSrc[xl];
        yr = pSrc[xr];
        pRst[i] = xl == xr ? yl : yl + (xi - xl) * ((yr - yl) / (xr - xl));
    }
    return 0;
}

uint32_t
pk_blackman_coefs(float32_t *coefs, uint32_t len) {
    /**
     * @brief Generate Blackman window coefficients
     *
     */
    for (size_t i = 0; i < len; i++) {
        int32_t n = 2 * i - len + 1;
        coefs[i] = 0.42 + 0.5 * cosf(PI * n / (len - 1)) + 0.08 * cosf(2 * PI * n / (len - 1));
    }
    return 0;
}

uint32_t
pk_apply_biquad_filter(arm_biquad_casd_df1_inst_f32 *ctx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize) {
    /**
     * @brief Apply biquad filter to signal
     */
    arm_biquad_cascade_df1_f32(ctx, pSrc, pResult, blockSize);
    return 0;
}

uint32_t
pk_apply_biquad_filtfilt(arm_biquad_casd_df1_inst_f32 *ctx, float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *state) {
    /**
     * @brief Apply biquad filter forward-backward to signal
     */
    // Forward pass
    arm_fill_f32(0, ctx->pState, 4 * ctx->numStages);
    arm_biquad_cascade_df1_f32(ctx, pSrc, pResult, blockSize);
    for (size_t i = 0; i < blockSize; i++) {
        state[i] = pResult[blockSize - 1 - i];
    }
    // Backward pass
    arm_fill_f32(0, ctx->pState, 4 * ctx->numStages);
    arm_biquad_cascade_df1_f32(ctx, state, pResult, blockSize);
    for (size_t i = 0; i < blockSize; i++) {
        state[i] = pResult[blockSize - 1 - i];
    }
    for (size_t i = 0; i < blockSize; i++) {
        pResult[i] = state[i];
    }
    return 0;
}

uint32_t
pk_quotient_filter_mask(uint32_t *data, uint8_t *mask, uint32_t dataLen, uint32_t iterations, float32_t lowcut, float32_t highcut) {
    /**
     * @brief Apply quotient filter mask to signal
     */
    int32_t m = -1, n = -1;
    uint32_t numFound;
    float32_t q;
    for (size_t iter = 0; iter < iterations; iter++) {
        numFound = 0;
        for (size_t i = 0; i < dataLen; i++) {
            if (mask[i] == 0) {
                // Find first value
                if (m == -1) {
                    m = i;
                    n = -1;
                    // Find second value
                } else if (n == -1) {
                    n = i;
                }
                // Compute quotient and check if in range
                if (m != -1 && n != -1) {
                    q = (float32_t)data[m] / (float32_t)data[n];
                    if (q < lowcut || q > highcut) {
                        mask[m] = 1;
                        numFound++;
                    }
                    m = -1;
                    n = -1;
                }
            }
        }
        // Stop early if no new values found
        if (numFound == 0) {
            break;
        }
    }

    return 0;
}

uint32_t
smooth_signal(float32_t *pSrc, float32_t *pResult, uint32_t blockSize, float32_t *wBuffer, uint32_t windowSize) {
    /**
     * @brief Smooth signal using moving average filter
     */

    // Utilize dot product to compute moving average
    uint32_t halfWindowSize = windowSize / 2;
    arm_fill_f32(1.0f / windowSize, wBuffer, windowSize);
    for (size_t i = 0; i < blockSize - windowSize; i++) {
        arm_dot_prod_f32(pSrc + i, wBuffer, windowSize, pResult + i + halfWindowSize);
    }
    // Replicate first and last values at the edges
    arm_fill_f32(pResult[halfWindowSize], pResult, halfWindowSize);
    uint32_t dpEnd = blockSize - windowSize - 1 + halfWindowSize;
    arm_fill_f32(pResult[dpEnd], &pResult[dpEnd], blockSize - dpEnd);

    return 0;
}

uint32_t
pk_ppg_find_peaks(ppg_peak_f32_t *ctx, float32_t *ppg, uint32_t ppgLen, uint32_t *peaks) {
    /**
     * @brief Find systolic peaks in PPG signal
     */

    // Apply 1st moving average filter
    float32_t muSqrd;

    uint32_t maPeakLen = (uint32_t)(ctx->sampleRate * ctx->peakWin + 1);
    uint32_t maBeatLen = (uint32_t)(ctx->sampleRate * ctx->beatWin + 1);
    uint32_t minPeakDelay = (uint32_t)(ctx->sampleRate * ctx->peakDelayWin + 1);
    uint32_t minPeakWidth = maPeakLen;

    float32_t *maPeak = &ctx->state[0 * ppgLen];
    float32_t *maBeat = &ctx->state[1 * ppgLen];
    float32_t *sqrd = &ctx->state[2 * ppgLen];
    float32_t *wBuffer = &ctx->state[3 * ppgLen];

    // Compute squared signal
    for (size_t i = 0; i < ppgLen; i++) {
        sqrd[i] = ppg[i] > 0 ? ppg[i] * ppg[i] : 0;
    }

    pk_mean(sqrd, &muSqrd, ppgLen);
    muSqrd = muSqrd * ctx->beatOffset;

    // Apply peak moving average
    smooth_signal(sqrd, maPeak, ppgLen, wBuffer, maPeakLen);

    // Apply beat moving average
    smooth_signal(sqrd, maBeat, ppgLen, wBuffer, maBeatLen);
    arm_offset_f32(maBeat, muSqrd, maBeat, ppgLen);

    arm_sub_f32(maPeak, maBeat, maPeak, ppgLen);

    uint32_t riseEdge, fallEdge, peakDelay, peakLen, peak;
    float32_t peakVal;
    uint32_t numPeaks = 0;
    int32_t m = -1, n = -1;
    for (size_t i = 1; i < ppgLen; i++) {
        riseEdge = maPeak[i - 1] <= 0 && maPeak[i] > 0;
        fallEdge = maPeak[i - 1] > 0 && maPeak[i] <= 0;
        if (riseEdge) {
            m = i;
        } else if (fallEdge && m != -1) {
            n = i;
        }
        // Detected peak
        if (m != -1 && n != -1) {
            peakLen = n - m + 1;
            arm_max_f32(&ppg[m], peakLen, &peakVal, &peak);
            peak += m;
            peakDelay = numPeaks > 0 ? peak - peaks[numPeaks - 1] : minPeakDelay;
            if (peakLen >= minPeakWidth && peakDelay >= minPeakDelay) {
                peaks[numPeaks++] = peak;
            }
            m = -1;
            n = -1;
        }
    }
    return numPeaks;
}

uint32_t
pk_ecg_find_peaks(ecg_peak_f32_t *ctx, float32_t *ecg, uint32_t ecgLen, uint32_t *peaks) {
    /**
     * @brief Find R peaks in PPG signal
     */

    uint32_t qrsGradLen = (uint32_t)(ctx->sampleRate * ctx->qrsWin + 1);
    uint32_t avgGradLen = (uint32_t)(ctx->sampleRate * ctx->avgWin + 1);

    uint32_t minQrsDelay = (uint32_t)(ctx->sampleRate * ctx->qrsDelayWin + 1);
    uint32_t minQrsWidth = 0;

    float32_t *absGrad = &ctx->state[0 * ecgLen];
    float32_t *avgGrad = &ctx->state[0 * ecgLen];
    float32_t *qrsGrad = &ctx->state[1 * ecgLen];
    float32_t *wBuffer = &ctx->state[2 * ecgLen];

    // Compute absolute gradient
    pk_gradient(ecg, absGrad, ecgLen);
    arm_abs_f32(absGrad, absGrad, ecgLen);

    // Smooth gradients
    smooth_signal(absGrad, qrsGrad, ecgLen, wBuffer, qrsGradLen);
    smooth_signal(qrsGrad, avgGrad, ecgLen, wBuffer, avgGradLen);

    // Min QRS height
    arm_scale_f32(avgGrad, ctx->qrsPromWeight, avgGrad, ecgLen);

    // Subtract average gradient as threshold
    arm_sub_f32(qrsGrad, avgGrad, qrsGrad, ecgLen);

    uint32_t riseEdge, fallEdge, peakDelay, peakLen, peak;
    float32_t peakVal;
    uint32_t numPeaks = 0;
    int32_t m = -1, n = -1;
    for (size_t i = 1; i < ecgLen; i++) {
        riseEdge = qrsGrad[i - 1] <= 0 && qrsGrad[i] > 0;
        fallEdge = qrsGrad[i - 1] > 0 && qrsGrad[i] <= 0;
        if (riseEdge) {
            m = i;
        } else if (fallEdge && m != -1) {
            n = i;
        }
        // If detected
        if (m != -1 && n != -1) {
            peakLen = n - m + 1;
            arm_max_f32(&ecg[m], peakLen, &peakVal, &peak);
            peak += m;
            peakDelay = numPeaks > 0 ? peak - peaks[numPeaks - 1] : minQrsDelay;

            if (peakLen >= minQrsWidth && peakDelay >= minQrsDelay) {
                peaks[numPeaks++] = peak;
            }
            m = -1;
            n = -1;
        }
    }
    return numPeaks;
}

float32_t
pk_compute_spo2_from_perfusion(float32_t dc1, float32_t ac1, float32_t dc2, float32_t ac2, float32_t *coefs) {
    float32_t r = (ac1 / dc1) / (ac2 / dc2);
    float32_t spo2 = coefs[0] * r * r + coefs[1] * r + coefs[2];
    return spo2;
}

float32_t
pk_compute_spo2_in_time(float32_t *ppg1, float32_t *ppg2, float32_t ppg1Mean, float32_t ppg2Mean, uint32_t blockSize, float32_t *coefs,
                        float32_t sampleRate) {
    float32_t ppg1Dc, ppg2Dc, ppg1Ac, ppg2Ac, spo2;

    // Compute DC via mean
    ppg1Dc = ppg1Mean;
    ppg2Dc = ppg2Mean;

    // Assume signals are already filtered

    // Compute AC via RMS
    arm_rms_f32(ppg1, blockSize, &ppg1Ac);
    arm_rms_f32(ppg2, blockSize, &ppg2Ac);

    // Compute SpO2
    spo2 = pk_compute_spo2_from_perfusion(ppg1Dc, ppg1Ac, ppg2Dc, ppg2Ac, coefs);
    return spo2;
}

uint32_t
pk_compute_rr_intervals(uint32_t *peaks, uint32_t numPeaks, uint32_t *rrIntervals) {
    /**
     * @brief Compute RR intervals from peaks
     */
    for (size_t i = 1; i < numPeaks; i++) {
        rrIntervals[i - 1] = peaks[i] - peaks[i - 1];
    }
    rrIntervals[numPeaks - 1] = rrIntervals[numPeaks - 2];
    return 0;
}

uint32_t
pk_filter_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, uint32_t sampleRate) {
    /**
     * @brief Filter RR intervals
     */
    uint32_t lowcut = 0.3 * sampleRate;
    uint32_t highcut = 2.0 * sampleRate;

    // Filter out peaks with RR intervals outside of normal range
    uint32_t val;
    uint8_t maskVal;
    for (size_t i = 0; i < numPeaks; i++) {
        val = rrIntervals[i];
        maskVal = (val < lowcut) || (val > highcut) ? 1 : 0;
        mask[i] = maskVal;
    }

    pk_quotient_filter_mask(rrIntervals, mask, numPeaks, 2, 0.7, 1.3);
    return 0;
}

uint32_t
pk_compute_hrv_from_rr_intervals(uint32_t *rrIntervals, uint32_t numPeaks, uint8_t *mask, hrv_td_metrics_t *metrics) {
    // Deviation-based
    metrics->meanNN = 0;
    metrics->sdNN = 0;
    uint32_t numValid = 0;
    for (size_t i = 0; i < numPeaks; i++) {
        if (mask[i] == 0) {
            metrics->meanNN += rrIntervals[i];
            metrics->sdNN += rrIntervals[i] * rrIntervals[i];
            numValid++;
        }
    }
    metrics->meanNN /= numValid;
    metrics->sdNN = sqrt(metrics->sdNN / numValid - metrics->meanNN * metrics->meanNN);

    // Difference-based
    float32_t meanSD = 0;
    metrics->rmsSD = 0;
    metrics->sdSD = 0;
    metrics->nn20 = 0;
    metrics->nn50 = 0;
    metrics->minNN = -1;
    metrics->maxNN = -1;
    int32_t v1, v2, v3, v4;
    for (size_t i = 1; i < numPeaks; i++) {
        v1 = mask[i - 1] == 0 ? rrIntervals[i - 1] : metrics->meanNN;
        v2 = mask[i] == 0 ? rrIntervals[i] : metrics->meanNN;
        v3 = (v2 - v1);
        v4 = v3 * v3;
        meanSD += v3;
        metrics->rmsSD += v4;
        metrics->sdSD += v4;
        if (1000 * v4 > 20 * 20) {
            metrics->nn20++;
        }
        if (1000 * v4 > 50 * 50) {
            metrics->nn50++;
        }
        if (rrIntervals[i] < metrics->minNN || metrics->minNN == -1) {
            metrics->minNN = rrIntervals[i];
        }
        if (rrIntervals[i] > metrics->maxNN || metrics->maxNN == -1) {
            metrics->maxNN = rrIntervals[i];
        }
    }
    meanSD /= (numPeaks - 1);
    metrics->rmsSD = sqrt(metrics->rmsSD / (numPeaks - 1));
    metrics->sdSD = sqrt(metrics->sdSD / (numPeaks - 2) - meanSD * meanSD);
    metrics->pnn20 = 100.0f * metrics->nn20 / (numPeaks - 1);
    metrics->pnn50 = 100.0f * metrics->nn50 / (numPeaks - 1);
    // Normalized
    metrics->cvNN = metrics->sdNN / metrics->meanNN;
    metrics->cvSD = metrics->rmsSD / metrics->meanNN;

    // Robust
    metrics->medianNN = 0;
    metrics->madNN = 0;
    metrics->mcvNN = 0;
    // Use mean & std for IQR
    float32_t q1 = metrics->meanNN - 0.6745 * metrics->sdNN;
    float32_t q3 = metrics->meanNN + 0.6745 * metrics->sdNN;
    metrics->iqrNN = q3 - q1;
    metrics->prc20NN = 0;
    metrics->prc80NN = 0;
    return 0;
}
