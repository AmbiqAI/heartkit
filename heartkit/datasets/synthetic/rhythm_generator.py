import logging
import random

import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi

from . import presets
from . import wave_generator as wg
from .defines import SyntheticFiducials, SyntheticParameters, SyntheticSegments
from .helper_functions import evenly_spaced_y, smooth_and_noise

logger = logging.getLogger(__name__)


def generate_nsr(
    leads: int = 12,
    signal_frequency: float = 200,
    rate: int = 60,
    preset: str = "SR",
    noise_multiplier: float = 1.0,
    impedance: float = 1.0,
    p_multiplier: float = 1.0,
    t_multiplier: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, SyntheticParameters]:
    """Generate normal sinus rhythm (NSR) ECG signals

    Args:
        leads (int, optional): # ECG leads. Max is 12. Defaults to 12.
        signal_frequency (float, optional): Sampling frequency in Hz. Defaults to 200.
        rate (int, optional): Heart rate (BPM). Defaults to 60.
        preset (str, optional): ECG Preset. Defaults to "SR".
        noise_multiplier (float, optional): Noise multiplier. Defaults to 1.0.
        impedance (float, optional): Lead impedance to adjust y scale. Defaults to 1.0.
        p_multiplier (float, optional): P wave multiplier. Defaults to 1.0.
        t_multiplier (float, optional): T wave multiplier. Defaults to 1.0.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, SyntheticParameters]: x, y, segs, fids, params
    """
    frequency = 1000
    gap = int((60 / rate) * frequency)

    voltage_factor = 0.002
    leads = min(leads, 12)

    if signal_frequency > 1000:
        logger.warning("This program is designed for maximum frequency of 1000 Hz")
        logger.warning("Setting frequency to 1000 Hz")
        signal_frequency = 1000

    sizer = 11 * frequency
    X = np.zeros((leads, sizer))
    Y = np.zeros((leads, sizer))
    Y_segs = np.zeros((leads, sizer))
    Y_fids = np.zeros((leads, sizer))

    parameters = presets.generate_parameters(preset, rate)

    for h in range(leads):
        x = np.linspace(0, sizer, sizer)
        y = np.zeros(sizer)
        y_segs = np.zeros(sizer)
        y_fids = np.zeros(sizer)

        start = 0
        beat_counter = 0
        beat_start = 0
        beat_length = (
            parameters.p_length
            + parameters.pr_interval
            + parameters.qrs_duration
            + parameters.st_length
            + parameters.t_length
        )

        if (
            beat_length
            > gap + parameters.qrs_duration + parameters.st_length + parameters.t_length
        ):
            logger.exception("Error, heart rate too high for beat length")

        overlap = beat_length - gap

        while beat_start < sizer:
            start = beat_start

            ################################################
            ## P-Wave Segment
            ################################################

            x_p, y_p = wg.syn_p_wave(
                p_length=parameters.p_length,
                p_voltage=parameters.p_voltages[h] * 2,
                p_biphasic=parameters.p_biphasics[h],
                p_lean=parameters.p_leans[h],
                flipper=parameters.flippers[h],
            )
            x_p = x_p * p_multiplier
            y[start : min(start + y_p.size, sizer)] += evenly_spaced_y(
                x_p[: min(x_p.size, sizer - start)],
                y_p[: min(x_p.size, sizer - start)],
            )

            y_segs[start : min(start + y_p.size, sizer)] = SyntheticSegments.p_wave
            # if parameters.p_biphasics[h]:
            #   y_segs[start:min(start + y_p.size,sizer)] = SyntheticSegments.p_wave_biphasic

            if overlap > 0 and beat_counter > 0:
                y_segs[start : start + overlap] = SyntheticSegments.tp_overlap

            if start + y_p.size < sizer:
                y_segs[
                    start + y_p.size : start + parameters.pr_interval
                ] = SyntheticSegments.pr_interval

            y_fids[start] = SyntheticFiducials.p_wave

            start = start + parameters.pr_interval
            if start >= sizer:
                break

            ################################################
            ## QRS Complex
            ################################################

            x_qrs, y_qrs, wave_peak_list = wg.syn_qrs_complex(
                qrs_duration=parameters.qrs_duration,
                q_depth=parameters.q_depths[h],
                r_height=parameters.r_heights[h],
                r_prime_present=parameters.r_prime_presents[h],
                r_prime_height=parameters.r_prime_heights[h],
                r_to_r_prime_duration_ratio=parameters.r_to_r_prime_duration_ratio[h],
                s_prime_height=parameters.s_prime_heights[h],
                s_present=parameters.s_presents[h],
                s_depth=parameters.s_depths[h],
                s_to_qrs_duration_ratio=parameters.s_to_qrs_duration_ratio[h],
                flipper=parameters.flippers[h],
                j_point=parameters.j_points[h],
            )
            y[start : start + y_qrs.size] = evenly_spaced_y(
                x_qrs[: min(x_qrs.size, sizer - start)],
                y_qrs[: min(x_qrs.size, sizer - start)],
            )
            y_segs[start : start + y_qrs.size] = SyntheticSegments.qrs_complex

            # # check if QRS complex predominantly negative:
            # if parameters.s_presents[h] and parameters.s_depths[h] > parameters.r_heights[h] and parameters.s_depths[h] > parameters.r_prime_heights[h]:
            #     # check if broad QRS
            #     if parameters.qrs_duration > 120:
            #         # check if RSR pattern:
            #         if parameters.r_prime_presents[h] and parameters.r_prime_heights[h] > parameters.s_prime_heights[h] \
            #                 and parameters.r_heights[h] > parameters.s_prime_heights[h]:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_inv_rsr
            #         # if no RSR pattern:
            #         else:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_inv
            #     # if QRS narrow:
            #     else:
            #         y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_inv
            # # if QRS predominantly positive:
            # else:
            #     # check if broad QRS
            #     if parameters.qrs_duration > 120:
            #         # check if RSR pattern:
            #         if wave_peak_list[2] > 0:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_rsr
            #         # if no RSR pattern:
            #         else:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide

            for pk in range(len(wave_peak_list)):
                if wave_peak_list[pk] != -1 and start + wave_peak_list[pk] < sizer:
                    y_fids[start + wave_peak_list[pk]] = SyntheticFiducials(10 + pk)
            y_fids[start] = SyntheticFiducials.q_wave

            start = start + x_qrs.size
            if start >= sizer:
                break

            ################################################
            ## ST Segment
            ################################################

            if parameters.st_length > 0:
                x_st, y_st = wg.syn_st_segment(
                    j_point=parameters.j_points[h],
                    st_delta=parameters.st_deltas[h],
                    st_length=parameters.st_length,
                    flipper=parameters.flippers[h],
                )
                y[start : min(start + x_st.size, sizer)] = evenly_spaced_y(
                    x_st[: min(x_st.size, sizer - start)],
                    y_st[: min(x_st.size, sizer - start)],
                )
                y_segs[start : start + x_st.size] = SyntheticSegments.st_segment
                # # check if upsloping ST segment:
                # if parameters.st_delta > 0:
                #     y_segs[start:min(start + x_st.size,sizer)] = SyntheticSegments.st_segment_upsloping
                # # if downsloping ST segment:
                # elif parameters.st_delta < 0:
                #     y_segs[start:min(start + x_st.size,sizer)] =  SyntheticSegments.st_segment_downsloping

                y_fids[start] = SyntheticFiducials.j_point

                start = start + x_st.size
                if start >= sizer:
                    break

            ################################################
            ## T Wave
            ################################################

            st_end = (
                parameters.j_points[h] + parameters.st_deltas[h]
                if (parameters.st_length) > 0
                else parameters.j_points[h]
            )
            x_t, y_t = wg.syn_t_wave(
                st_end=st_end,
                t_height=parameters.t_heights[h] * 0.1,
                t_length=parameters.t_length,
                flipper=parameters.flippers[h],
                t_lean=parameters.t_leans[h],
            )
            x_t = x_t * t_multiplier
            y[start : min(start + x_t.size, sizer)] = evenly_spaced_y(
                x_t[: min(x_t.size, sizer - start)], y_t[: min(x_t.size, sizer - start)]
            )
            y_segs[start : start + x_t.size] = SyntheticSegments.t_wave

            if parameters.st_length > 5 and min(start + x_t.size, sizer) - start > 5:
                t_grad = (
                    np.amax(
                        np.abs(
                            np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5)
                        )
                    )
                    / 10
                )

                st_grad = np.mean(np.abs(np.gradient(y[start - 5 : start - 1] * 1e5)))

                grad = np.abs(
                    np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5)
                )

                for t_value in range(start, min(start + x_t.size, sizer), 1):
                    if abs(st_grad - grad[t_value - start]) < t_grad:
                        y_segs[t_value] = SyntheticSegments.st_segment
                    else:
                        y_segs[
                            t_value : min(start + x_t.size, sizer)
                        ] = SyntheticSegments.t_wave
                        break

            # # if T wave inverted:
            # y_segs[start:min(start + x_t.size,sizer)] = SyntheticSegments.t_wave_inv

            # mark J point if no ST segment:
            if parameters.st_length == 0:
                y_fids[start] = SyntheticFiducials.j_point

            beat_start += gap
            beat_counter += 1
            y_segs[start + x_t.size : beat_start] = SyntheticSegments.tp_segment

            # calculate end of QT interval using max slope method
            qt_end_set = False
            if x_t.size > 1 and start + x_t.size < sizer:
                grad = np.gradient(
                    y[
                        max(start, x_t.size - 100) : max(start, x_t.size - 100)
                        + x_t.size
                    ]
                )
                if parameters.flippers[h] > 0:
                    max_slope_x_coordinate = np.argmin(grad)
                else:
                    max_slope_x_coordinate = np.argmax(grad)

                if (
                    y[start + max_slope_x_coordinate] != 0
                    and grad[max_slope_x_coordinate] != 0
                ):
                    # x intercept of maximum slope = x value of maximum slope + (y value of maximum slope * -gradient)
                    qt_end = int(
                        max_slope_x_coordinate
                        + (
                            y[start + max_slope_x_coordinate]
                            / -grad[max_slope_x_coordinate]
                        )
                    )
                    if start + qt_end < sizer and qt_end > 0:
                        y_fids[start + qt_end] = SyntheticFiducials.qt_segment
                        qt_end_set = True
                y_fids[start + x_t.size] = SyntheticFiducials.t_wave
        # END WHILE

        y = smooth_and_noise(
            y,
            noise_multiplier=noise_multiplier,
            impedance=impedance,
        )
        X[h,] = x
        Y[h,] = y
        Y_segs[h,] = y_segs
        Y_fids[h,] = y_fids
    # END FOR

    delay_start = random.randint(0, frequency - 1)
    delay_end = -(frequency - delay_start)
    X = X[:, delay_start:delay_end]
    for lead in range(X.shape[0]):
        X[lead, :] = X[lead, :] - X[lead, 0]

    Y = Y[:, delay_start:delay_end]
    Y_segs = Y_segs[:, delay_start:delay_end]
    Y_fids = Y_fids[:, delay_start:delay_end]

    if signal_frequency < frequency:
        X_max = np.amax(X)
        Y_max = np.amax(Y)
        Y_segs_max = np.amax(Y_segs)
        Y_fids_max = np.amax(Y_fids)

        X /= X_max
        Y /= Y_max
        Y_segs /= Y_segs_max
        Y_fids /= Y_fids_max

        X = (
            ndi.zoom(
                X, (1, (10 * signal_frequency) / frequency), order=1, grid_mode=True
            )
            * X_max
        )
        Y = (
            ndi.zoom(
                Y, (1, (10 * signal_frequency) / frequency), order=1, grid_mode=True
            )
            * Y_max
        )
        Y_segs = (
            ndi.zoom(
                Y_segs,
                (1, (10 * signal_frequency) / frequency),
                order=1,
                grid_mode=True,
            )
            * Y_segs_max
        )
        Y_fids = (
            ndi.zoom(
                Y_fids,
                (1, (10 * signal_frequency) / frequency),
                order=1,
                grid_mode=True,
            )
            * Y_fids_max
        )
    # END IF

    return X, Y, Y_segs.astype(int), Y_fids.astype(int), parameters


def generate_af(
    leads: int = 12,
    signal_frequency: float = 200,
    rate: float = 60,
    preset: str = "SR",
    variability: float = 0.1,
    noise_multiplier: float = 1.0,
    impedance: float = 1.0,
    p_multiplier: float = 1.0,
    t_multiplier: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, SyntheticParameters]:
    """Generate AF rhythm ECG signals

    Args:
        leads (int, optional): # ECG leads. Defaults to 12.
        signal_frequency (float, optional): Sampling frequency in Hz. Defaults to 200.
        rate (int, optional): Heart rate (BPM). Defaults to 60.
        preset (str, optional): ECG Preset. Defaults to "SR".
        noise_multiplier (float, optional): Noise multiplier. Defaults to 1.0.
        impedance (float, optional): Lead impedance to adjust y scale. Defaults to 1.0.
        p_multiplier (float, optional): P wave multiplier. Defaults to 1.0.
        t_multiplier (float, optional): T wave multiplier. Defaults to 1.0.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, SyntheticParameters]: x, y, segs, fids, params
    """
    variability = random.uniform(0.05, 0.4)

    frequency = 1000
    gap = int((60 / rate) * frequency)
    voltage_factor = 0.002
    leads = min(leads, 12)

    if signal_frequency > 1000:
        print("This program is designed for maximum frequency of 1000Hz")
        print("Setting frequency to 1000Hz")
        signal_frequency = 1000

    sizer = 11 * frequency
    X = np.zeros((leads, sizer))
    Y = np.zeros((leads, sizer))
    Y_segs = np.zeros((leads, sizer))
    Y_fids = np.zeros((leads, sizer))

    parameters = presets.generate_parameters(preset, rate)

    trace_length = 0
    beat_length = parameters.qrs_duration + parameters.st_length + parameters.t_length
    gaps = []
    while trace_length < sizer:
        next_gap = gap + random.randint(int(-gap * variability), int(gap * variability))
        gaps.append(next_gap)
        trace_length += next_gap
    gaps.append(gap + random.randint(int(-gap * variability), int(gap * variability)))

    for h in range(leads):
        x = np.linspace(0, sizer, sizer)
        y = np.zeros(sizer)
        y_segs = np.zeros(sizer)
        y_fids = np.zeros(sizer)

        beat_counter = 0
        start = 0
        beat_start = 0

        if (
            beat_length
            > int(gap - (variability * gap))
            + parameters.qrs_duration
            + parameters.st_length
            + parameters.t_length
        ):
            logger.exception("Error, heart rate too high for beat length")

        while beat_start < sizer:
            start = beat_start

            x_qrs, y_qrs, wave_peak_list = wg.syn_qrs_complex(
                qrs_duration=parameters.qrs_duration,
                q_depth=parameters.q_depths[h],
                r_height=parameters.r_heights[h],
                r_prime_present=parameters.r_prime_presents[h],
                r_prime_height=parameters.r_prime_heights[h],
                r_to_r_prime_duration_ratio=parameters.r_to_r_prime_duration_ratio[h],
                s_prime_height=parameters.s_prime_heights[h],
                s_present=parameters.s_presents[h],
                s_depth=parameters.s_depths[h],
                s_to_qrs_duration_ratio=parameters.s_to_qrs_duration_ratio[h],
                flipper=parameters.flippers[h],
                j_point=parameters.j_points[h],
            )
            y[start : min(start + y_qrs.size, sizer)] = evenly_spaced_y(
                x_qrs[: min(x_qrs.size, sizer - start)],
                y_qrs[: min(x_qrs.size, sizer - start)],
            )
            y_segs[start : start + y_qrs.size] = SyntheticSegments.qrs_complex

            # # check if QRS complex predominantly negative:
            # if s_present and s_depth > r_height and s_depth > r_prime_height:
            # 	# check if broad QRS
            # 	if parameters.qrs_duration > 120:
            # 		# check if RSR pattern:
            # 		if parameters.r_prime_presents[h] and parameters.r_prime_height[h] > parameters.s_prime_heights[h] \
            # 				and parameters.r_height[h] > parameters.s_prime_heights[h]:
            # 			y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_inv_rsr
            # 		# if no RSR pattern:
            # 		else:
            # 			y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_inv
            # 	# if QRS narrow:
            # 	else:
            # 		y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_inv
            # # if QRS predominantly positive:
            # else:
            #     # check if broad QRS
            #     if parameters.qrs_duration > 120:
            #         # check if RSR pattern:
            #         if wave_peak_list[2] > 0:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide_rsr
            #         # if no RSR pattern:
            #         else:
            #             y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex_wide
            #     # if QRS narrow:
            #     else:
            #         y_segs[start : min(start + y_qrs.size, sizer)] = SyntheticSegments.qrs_complex

            for pk in range(len(wave_peak_list)):
                if wave_peak_list[pk] != -1 and start + wave_peak_list[pk] < sizer:
                    y_fids[start + wave_peak_list[pk]] = SyntheticFiducials(10.0 + pk)
                # mark onset of QRS complex:
                y_fids[start] = SyntheticFiducials.q_wave

            start = start + x_qrs.size
            if start >= sizer:
                break

            if parameters.st_length > 0:
                x_st, y_st = wg.syn_st_segment(
                    j_point=parameters.j_points[h],
                    st_delta=parameters.st_deltas[h],
                    st_length=parameters.st_length,
                    flipper=parameters.flippers[h],
                )
                y[start : min(start + x_st.size, sizer)] = evenly_spaced_y(
                    x_st[: min(x_st.size, sizer - start)],
                    y_st[: min(x_st.size, sizer - start)],
                )

                # check if upsloping ST segment:
                # if parameters.st_delta > 0:
                # 	y_segs[start:min(start + x_st.size,sizer)] = SyntheticSegments.st_segment_upsloping
                # # if downsloping ST segment:
                # elif parameters.st_delta < 0:
                # 	y_segs[start:min(start + x_st.size,sizer)] = SyntheticSegments.st_segment_downsloping

                y_segs[start : start + x_st.size] = SyntheticSegments.st_segment

                # mark J point
                y_fids[start] = SyntheticFiducials.j_point

                start = start + x_st.size
                if start > sizer:
                    break

            st_end = (
                parameters.j_points[h] + parameters.st_deltas[h]
                if parameters.st_length > 0
                else parameters.j_points[h]
            )
            x_t, y_t = wg.syn_t_wave(
                st_end=st_end,
                t_height=parameters.t_heights[h] * 0.1,
                t_length=parameters.t_length,
                flipper=parameters.flippers[h],
                t_lean=parameters.t_leans[h],
            )

            x_t = x_t * t_multiplier
            y[start : min(start + x_t.size, sizer)] = evenly_spaced_y(
                x_t[: min(x_t.size, sizer - start)], y_t[: min(x_t.size, sizer - start)]
            )

            if parameters.st_length > 5 and min(start + x_t.size, sizer) - start > 5:
                t_grad = (
                    np.amax(
                        np.abs(
                            np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5)
                        )
                    )
                    / 10
                )

                st_grad = np.mean(np.abs(np.gradient(y[start - 5 : start - 1] * 1e5)))

                grad = np.abs(
                    np.gradient(y[start : min(start + x_t.size, sizer)] * 1e5)
                )

                for t_value in range(start, min(start + x_t.size, sizer), 1):
                    if abs(st_grad - grad[t_value - start]) < t_grad:
                        y_segs[t_value] = SyntheticSegments.st_segment
                    else:
                        y_segs[
                            t_value : min(start + x_t.size, sizer)
                        ] = SyntheticSegments.t_wave
                        break
            else:
                y_segs[start : min(start + x_t.size, sizer)] = SyntheticSegments.t_wave

            # # if T wave inverted:
            # else:
            # 	y_segs[start:min(start + x_t.size,sizer)] = 24.

            # mark J point if no ST segment
            if parameters.st_length == 0:
                y_fids[start] = SyntheticFiducials.j_point

            gap = gaps[beat_counter]
            beat_start += gap
            beat_counter += 1
            y_segs[start + x_t.size : beat_start] = SyntheticSegments.tp_segment

            # calculate end of QT interval using max slope method
            qt_end_set = False
            if x_t.size > 1 and start + x_t.size < sizer:
                qt_end = -1
                grad = np.gradient(
                    y[
                        max(start, x_t.size - 100) : max(start, x_t.size - 100)
                        + x_t.size
                    ]
                )
                if flipper > 0:
                    max_slope_x_coordinate = np.argmin(grad)
                else:
                    max_slope_x_coordinate = np.argmax(grad)

                if (
                    y[start + max_slope_x_coordinate] != 0
                    and grad[max_slope_x_coordinate] != 0
                ):
                    # x intercept of maximum slope = x value of maximum slope + (y value of maximum slope * -gradient)
                    qt_end = int(
                        max_slope_x_coordinate
                        + (
                            y[start + max_slope_x_coordinate]
                            / -grad[max_slope_x_coordinate]
                        )
                    )
                    if start + qt_end < sizer and qt_end > 0:
                        y_label[start + qt_end] = SyntheticFiducials.qt_segment
                        qt_end_set = True
                y_fids[start + x_t.size] = SyntheticFiducials.t_wave
            # END IF
        # END WHILE

        y = smooth_and_noise(
            y,
            rhythm="af",
            noise_multiplier=noise_multiplier,
            impedance=impedance,
        )
        X[h,] = x
        Y[h,] = y
        Y_segs[h,] = y_segs
        Y_fids[h,] = y_fids

    delay_start = random.randint(0, frequency - 1)
    delay_end = -(frequency - delay_start)
    X = X[:, delay_start:delay_end]

    for lead in range(X.shape[0]):
        X[lead, :] = X[lead, :] - X[lead, 0]

    Y = Y[:, delay_start:delay_end]
    Y_segs = Y_segs[:, delay_start:delay_end]
    Y_fids = Y_fids[:, delay_start:delay_end]

    if signal_frequency < frequency:
        X_max = np.amax(X)
        Y_max = np.amax(Y)
        Y_segs_max = np.amax(Y_segs)
        Y_fids_max = np.amax(Y_fids)

        X /= X_max
        Y /= Y_max
        Y_segs /= Y_segs_max
        Y_fids /= Y_fids_max

        X = (
            ndi.zoom(
                X, (1, (10 * signal_frequency) / frequency), order=1, grid_mode=True
            )
            * X_max
        )
        Y = (
            ndi.zoom(
                Y, (1, (10 * signal_frequency) / frequency), order=1, grid_mode=True
            )
            * Y_max
        )
        Y_segs = (
            ndi.zoom(
                Y_segs,
                (1, (10 * signal_frequency) / frequency),
                order=1,
                grid_mode=True,
            )
            * Y_segs_max
        )
        Y_fids = (
            ndi.zoom(
                Y_fids,
                (1, (10 * signal_frequency) / frequency),
                order=1,
                grid_mode=True,
            )
            * Y_fids_max
        )
    # END IF

    return X, Y, Y_segs.astype(int), Y_fids.astype(int), parameters
