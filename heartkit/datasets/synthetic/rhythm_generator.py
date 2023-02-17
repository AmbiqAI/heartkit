import random

import numpy as np

from . import presets
from . import wave_generator as wg
from .helper_functions import evenly_spaced_y, smooth_and_noise

# from skimage.transform import resize

# y_label key:
# WAVES
# 0 = background
# 1 = P wave
# 16 = Biphasic P wave
# 2 = PR interval
# 3 = Narrow QRS complex
# 17 = Broad QRS with RSR pattern
# 18 = Broad QRS without RSR pattern
# 19 = Inverted narrow QRS complex #
# 20 = Inverted QRS with RSR pattern #
# 21 = Inverted QRS without RSR pattern #
# 4 = Flat ST segment
# 22 = Upsloping ST segment
# 23 = Downsloping ST segment
# 5 = Upright T wave
# 24 = Inverted T wave
# 6 = T-P segment
# 7 = T/P overlap
# POINTS
# 8 = start of P wave
# 9 = start of Q wave
# 10 = Q trough (-1 if absent)
# 11 = R peak
# 12 = R prime peak (-1 if absent)
# 13 = S trough (-1 if absent)
# 14 = J point
# 15 = end of QT segment (max slope method)
# 25 = end of T wave (absolute method)


def nsr(
    leads=12,
    signal_frequency=200,
    generate_labels=True,
    rate=60,
    preset="SR",
    universal_noise_multiplier=1.0,
    impedance=1.0,
    p_multiplier=1.0,
    t_multiplier=1.0,
    include_points=True,
):
    meta_data = {}
    meta_data["hr"] = rate
    frequency = 1000
    gap = int((60 / rate) * frequency)
    subplots = leads
    frequency = frequency
    voltage_factor = 0.002

    if signal_frequency > 1000:
        print("This program is designed for maximum frequency of 1000Hz")
        print("Setting frequency to 1000Hz")
        signal_frequency = 1000

    sizer = 11 * frequency
    X = np.zeros((leads, sizer))
    Y = np.zeros((leads, sizer))
    Y_label = np.zeros((leads, sizer))

    parameters, label_vector = presets.calculate_parameters(preset, rate)
    p_length = parameters["p_length"]
    pr_interval = parameters["pr_interval"]
    meta_data["pr"] = pr_interval
    qrs_duration = parameters["qrs_duration"]
    meta_data["qrs"] = qrs_duration
    noisiness = parameters["noisiness"]
    st_length = parameters["st_length"]
    t_length = parameters["t_length"]
    qt = parameters["qt"]
    qtc = parameters["qtc"]
    meta_data["qt"] = qt
    meta_data["qtc"] = int(qtc)
    flippers = parameters["flippers"]
    p_voltages = parameters["p_voltages"]
    p_biphasics = parameters["p_biphasics"]
    p_leans = parameters["p_leans"]
    q_depths = parameters["q_depths"]
    r_heights = parameters["r_heights"]
    r_prime_presents = parameters["r_prime_presents"]
    r_prime_heights = parameters["r_prime_heights"]
    r_to_r_prime_duration_ratios = parameters["r_to_r_prime_duration_ratio"]
    s_presents = parameters["s_presents"]
    s_depths = parameters["s_depths"]
    s_prime_heights = parameters["s_prime_heights"]
    s_to_qrs_duration_ratios = parameters["s_to_qrs_duration_ratio"]
    st_deltas = parameters["st_deltas"]
    j_points = parameters["j_points"]
    t_heights = parameters["t_heights"]
    t_leans = parameters["t_leans"]

    subplots = 12
    for h in range(subplots):
        x = np.linspace(0, sizer, sizer)
        y = np.zeros(sizer)
        y_label = np.zeros(sizer)
        flipper = flippers[h]
        q_depth = q_depths[h]
        p_voltage = p_voltages[h] * 2
        p_biphasic = p_biphasics[h]
        p_lean = p_leans[h]
        r_height = r_heights[h]
        r_prime_present = r_prime_presents[h]
        r_prime_height = r_prime_heights[h]
        r_to_r_prime_duration_ratio = r_to_r_prime_duration_ratios[h]
        s_present = s_presents[h] == 1
        s_depth = s_depths[h]
        s_prime_height = s_prime_heights[h]
        s_to_qrs_duration_ratio = s_to_qrs_duration_ratios[h]
        st_delta = st_deltas[h]
        j_point = j_points[h]
        st_end = j_point
        t_height = t_heights[h] * 0.1
        t_lean = t_leans[h]
        if (st_length) > 0:
            st_end = j_point + st_delta
        else:
            st_end = j_point
        beat_counter = 0
        start = 0
        beat_start = 0
        beat_length = p_length + pr_interval + qrs_duration + st_length + t_length

        if beat_length > gap + qrs_duration + st_length + t_length:
            print("Error, heart rate too high for beat length")
            exit()

        overlap = beat_length - gap

        while beat_start < sizer:
            start = beat_start

            x_p, y_p = wg.P_wave(
                p_length=p_length,
                p_voltage=p_voltage,
                p_biphasic=p_biphasic,
                p_lean=p_lean,
                flipper=flipper,
            )
            x_p = x_p * p_multiplier
            y[start : min(start + y_p.size, sizer)] = np.add(
                evenly_spaced_y(
                    x_p[: min(x_p.size, sizer - start)],
                    y_p[: min(x_p.size, sizer - start)],
                ),
                y[start : min(start + y_p.size, sizer)],
            )

            # if p_biphasic > 0:
            # 	y_label[start:min(start + y_p.size,sizer)] = 16.
            # else:
            # 	y_label[start:min(start + y_p.size,sizer)] = 1.

            y_label[start : min(start + y_p.size, sizer)] = 1.0

            if overlap > 0 and beat_counter > 0:
                y_label[start : start + overlap] = 7.0

            if start + y_p.size < sizer:
                y_label[start + y_p.size : min(start + pr_interval, sizer)] = 2.0

            if include_points:
                y_label[start] = 8.0

            start = start + pr_interval
            if start >= sizer:
                break

            x_qrs, y_qrs, wave_peak_list = wg.QRS_complex(
                qrs_duration=qrs_duration,
                q_depth=q_depth,
                r_height=r_height,
                r_prime_present=r_prime_present,
                r_prime_height=r_prime_height,
                r_to_r_prime_duration_ratio=r_to_r_prime_duration_ratio,
                s_prime_height=s_prime_height,
                s_present=s_present,
                s_depth=s_depth,
                s_to_qrs_duration_ratio=s_to_qrs_duration_ratio,
                flipper=flipper,
                j_point=j_point,
            )
            y[start : min(start + y_qrs.size, sizer)] = evenly_spaced_y(
                x_qrs[: min(x_qrs.size, sizer - start)],
                y_qrs[: min(x_qrs.size, sizer - start)],
            )

            # # check if QRS complex predominantly negative:
            # if s_present and s_depth > r_height and s_depth > r_prime_height:
            # 	# check if broad QRS
            # 	if qrs_duration > 120:
            # 		# check if RSR pattern:
            # 		if r_prime_present and r_prime_height > s_prime_height \
            # 				and r_height > s_prime_height:
            # 			y_label[start : min(start + y_qrs.size, sizer)] = 20.
            # 		# if no RSR pattern:
            # 		else:
            # 			y_label[start : min(start + y_qrs.size, sizer)] = 21.
            # 	# if QRS narrow:
            # 	else:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 19.
            # # if QRS predominantly positive:
            # else:

            # # check if broad QRS
            # if qrs_duration > 120:
            # 	# check if RSR pattern:
            # 	if wave_peak_list[2] > 0:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 17.
            # 	# if no RSR pattern:
            # 	else:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 18.
            # # if QRS narrow:
            # else:
            # 	y_label[start : min(start + y_qrs.size, sizer)] = 3.

            y_label[start : min(start + y_qrs.size, sizer)] = 3.0

            if include_points:
                for pk in range(len(wave_peak_list)):
                    if wave_peak_list[pk] != -1 and start + wave_peak_list[pk] < sizer:
                        y_label[start + wave_peak_list[pk]] = 10.0 + pk
                # mark onset of QRS complex:
                y_label[start] = 9.0

            start = start + x_qrs.size
            if start >= sizer:
                break

            if st_length > 0:
                x_st, y_st = wg.ST_segment(
                    j_point=j_point,
                    st_delta=st_delta,
                    st_length=st_length,
                    flipper=flipper,
                )
                y[start : min(start + x_st.size, sizer)] = evenly_spaced_y(
                    x_st[: min(x_st.size, sizer - start)],
                    y_st[: min(x_st.size, sizer - start)],
                )

                # # check if upsloping ST segment:
                # if st_delta > 0:
                # 	y_label[start:min(start + x_st.size,sizer)] = 22.
                # # if downsloping ST segment:
                # elif st_delta < 0:
                # 	y_label[start:min(start + x_st.size,sizer)] = 23.
                # # if flat ST segment:
                # else:

                y_label[start : min(start + x_st.size, sizer)] = 4.0

                if include_points:
                    # mark J point:
                    y_label[start] = 14.0

                start = start + x_st.size
                if start >= sizer:
                    break

            x_t, y_t = wg.T_wave(
                st_end=st_end,
                t_height=t_height,
                t_length=t_length,
                flipper=flipper,
                t_lean=t_lean,
            )
            x_t = x_t * t_multiplier
            y[start : min(start + x_t.size, sizer)] = evenly_spaced_y(
                x_t[: min(x_t.size, sizer - start)], y_t[: min(x_t.size, sizer - start)]
            )

            # # check if T wave upright:
            # if t_height > 0:

            if st_length > 5 and min(start + x_t.size, sizer) - start > 5:
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
                        y_label[t_value] = 4.0
                    else:
                        y_label[t_value : min(start + x_t.size, sizer)] = 5.0
                        break
            else:
                y_label[start : min(start + x_t.size, sizer)] = 5.0

            # # if T wave inverted:
            # else:
            # 	y_label[start:min(start + x_t.size,sizer)] = 24.

            if include_points:
                # mark J point if no ST segment:
                if st_length == 0:
                    y_label[start] = 14.0

            beat_start += gap
            beat_counter += 1
            y_label[min(start + x_t.size, sizer) : min(sizer, beat_start)] = 6.0

            # calculate end of QT interval using max slope method
            qt_end_set = False
            if x_t.size > 1 and start + x_t.size < sizer:
                if include_points:
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
                            y_label[start + qt_end] = 15.0
                            qt_end_set = True
                y_label[start + x_t.size] = 16.0

        y = smooth_and_noise(
            y,
            universal_noise_multiplier=universal_noise_multiplier,
            impedance=impedance,
        )
        X[h,] = x
        Y[h,] = y
        Y_label[h,] = y_label

    delay_start = random.randint(0, frequency - 1)
    X = X[:, delay_start : -(frequency - delay_start)]

    for lead in range(X.shape[0]):
        X[lead, :] = X[lead, :] - X[lead, 0]

    Y = Y[:, delay_start : -(frequency - delay_start)]
    Y_label = Y_label[:, delay_start : -(frequency - delay_start)]
    X_1000 = X
    Y_1000 = Y
    Y_label_1000 = Y_label

    # if signal_frequency < 1000:
    #     X_max = np.amax(X)
    #     Y_max = np.amax(Y)
    #     Y_label_max = np.amax(Y_label)
    #     X = X / X_max
    #     Y = Y / Y_max
    #     Y_label = Y_label / Y_label_max
    #     X = (resize(X, (X.shape[0], signal_frequency * 10)) * X_max)
    #     Y = (resize(Y, (Y.shape[0], signal_frequency * 10)) * Y_max)
    #     Y_label = (resize(Y_label, (Y_label.shape[0], signal_frequency * 10)) * Y_label_max).astype(int)

    return X, Y, Y_label, X_1000, Y_1000, Y_label_1000, meta_data, label_vector


def af(
    leads=12,
    signal_frequency=200,
    generate_labels=True,
    rate=60,
    preset="SR",
    variability=0.1,
    universal_noise_multiplier=1.0,
    impedance=1.0,
    p_multiplier=1.0,
    t_multiplier=1.0,
    include_points=True,
):
    variability = random.uniform(0.05, 0.4)
    meta_data = {}
    meta_data["hr"] = rate
    frequency = 1000
    subplots = leads
    frequency = frequency
    gap = int((60 / rate) * frequency)
    voltage_factor = 0.002

    if signal_frequency > 1000:
        print("This program is designed for maximum frequency of 1000Hz")
        print("Setting frequency to 1000Hz")
        signal_frequency = 1000

    sizer = 11 * frequency
    X = np.zeros((leads, sizer))
    Y = np.zeros((leads, sizer))
    Y_label = np.zeros((leads, sizer))

    parameters, label_vector = presets.calculate_parameters(preset, rate)
    if len(label_vector) > 0:
        label_vector[12] = 0
    qrs_duration = parameters["qrs_duration"]
    meta_data["qrs"] = qrs_duration
    noisiness = parameters["noisiness"]
    st_length = parameters["st_length"]
    t_length = parameters["t_length"]
    qt = parameters["qt"]
    qtc = parameters["qtc"]
    meta_data["qt"] = qt
    meta_data["qtc"] = int(qtc)
    flippers = parameters["flippers"]
    q_depths = parameters["q_depths"]
    r_heights = parameters["r_heights"]
    r_prime_presents = parameters["r_prime_presents"]
    r_prime_heights = parameters["r_prime_heights"]
    r_to_r_prime_duration_ratios = parameters["r_to_r_prime_duration_ratio"]
    s_presents = parameters["s_presents"]
    s_depths = parameters["s_depths"]
    s_prime_heights = parameters["s_prime_heights"]
    s_to_qrs_duration_ratios = parameters["s_to_qrs_duration_ratio"]
    st_deltas = parameters["st_deltas"]
    j_points = parameters["j_points"]
    t_heights = parameters["t_heights"]
    t_leans = parameters["t_leans"]

    subplots = 12
    trace_length = 0
    beat_length = qrs_duration + st_length + t_length
    gaps = []
    while trace_length < sizer:
        next_gap = gap + random.randint(int(-gap * variability), int(gap * variability))
        gaps.append(next_gap)
        trace_length += next_gap
    gaps.append(gap + random.randint(int(-gap * variability), int(gap * variability)))

    for h in range(subplots):
        x = np.linspace(0, sizer, sizer)
        y = np.zeros(sizer)
        y_label = np.zeros(sizer)
        flipper = flippers[h]
        q_depth = q_depths[h]
        r_height = r_heights[h]
        r_prime_present = r_prime_presents[h]
        r_prime_height = r_prime_heights[h]
        r_to_r_prime_duration_ratio = r_to_r_prime_duration_ratios[h]
        s_present = s_presents[h] == 1
        s_depth = s_depths[h]
        s_prime_height = s_prime_heights[h]
        s_to_qrs_duration_ratio = s_to_qrs_duration_ratios[h]
        st_delta = st_deltas[h]
        j_point = j_points[h]
        st_end = j_point
        t_height = t_heights[h] * 0.1
        t_lean = t_leans[h]
        if (st_length) > 0:
            st_end = j_point + st_delta
        else:
            st_end = j_point
        beat_counter = 0
        start = 0
        beat_start = 0

        if (
            beat_length
            > int(gap - (variability * gap)) + qrs_duration + st_length + t_length
        ):
            print("Error, heart rate too high for beat length")
            exit()

        while beat_start < sizer:
            start = beat_start

            x_qrs, y_qrs, wave_peak_list = wg.QRS_complex(
                qrs_duration=qrs_duration,
                q_depth=q_depth,
                r_height=r_height,
                r_prime_present=r_prime_present,
                r_prime_height=r_prime_height,
                r_to_r_prime_duration_ratio=r_to_r_prime_duration_ratio,
                s_prime_height=s_prime_height,
                s_present=s_present,
                s_depth=s_depth,
                s_to_qrs_duration_ratio=s_to_qrs_duration_ratio,
                flipper=flipper,
                j_point=j_point,
            )
            y[start : min(start + y_qrs.size, sizer)] = evenly_spaced_y(
                x_qrs[: min(x_qrs.size, sizer - start)],
                y_qrs[: min(x_qrs.size, sizer - start)],
            )

            # # check if QRS complex predominantly negative:
            # if s_present and s_depth > r_height and s_depth > r_prime_height:
            # 	# check if broad QRS
            # 	if qrs_duration > 120:
            # 		# check if RSR pattern:
            # 		if r_prime_present and r_prime_height > s_prime_height \
            # 				and r_height > s_prime_height:
            # 			y_label[start : min(start + y_qrs.size, sizer)] = 20.
            # 		# if no RSR pattern:
            # 		else:
            # 			y_label[start : min(start + y_qrs.size, sizer)] = 21.
            # 	# if QRS narrow:
            # 	else:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 19.
            # # if QRS predominantly positive:
            # else:

            # # check if broad QRS
            # if qrs_duration > 120:
            # 	# check if RSR pattern:
            # 	if wave_peak_list[2] > 0:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 17.
            # 	# if no RSR pattern:
            # 	else:
            # 		y_label[start : min(start + y_qrs.size, sizer)] = 18.
            # # if QRS narrow:
            # else:
            # 	y_label[start : min(start + y_qrs.size, sizer)] = 3.

            y_label[start : min(start + y_qrs.size, sizer)] = 3.0

            if include_points:
                for pk in range(len(wave_peak_list)):
                    if wave_peak_list[pk] != -1 and start + wave_peak_list[pk] < sizer:
                        y_label[start + wave_peak_list[pk]] = 10.0 + pk

                # mark onset of QRS complex:
                y_label[start] = 9.0

            start = start + x_qrs.size
            if start >= sizer:
                break

            if st_length > 0:
                x_st, y_st = wg.ST_segment(
                    j_point=j_point,
                    st_delta=st_delta,
                    st_length=st_length,
                    flipper=flipper,
                )
                y[start : min(start + x_st.size, sizer)] = evenly_spaced_y(
                    x_st[: min(x_st.size, sizer - start)],
                    y_st[: min(x_st.size, sizer - start)],
                )

                # check if upsloping ST segment:
                # if st_delta > 0:
                # 	y_label[start:min(start + x_st.size,sizer)] = 22.
                # # if downsloping ST segment:
                # elif st_delta < 0:
                # 	y_label[start:min(start + x_st.size,sizer)] = 23.
                # # if flat ST segment:
                # else:

                y_label[start : min(start + x_st.size, sizer)] = 4.0

                if include_points:
                    # mark J point:
                    y_label[start] = 14.0

                start = start + x_st.size
                if start > sizer:
                    break

            x_t, y_t = wg.T_wave(
                st_end=st_end,
                t_height=t_height,
                t_length=t_length,
                flipper=flipper,
                t_lean=t_lean,
            )

            x_t = x_t * t_multiplier
            y[start : min(start + x_t.size, sizer)] = evenly_spaced_y(
                x_t[: min(x_t.size, sizer - start)], y_t[: min(x_t.size, sizer - start)]
            )

            # # check if T wave upright:
            # if t_height > 0:

            if st_length > 5 and min(start + x_t.size, sizer) - start > 5:
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
                        y_label[t_value] = 4.0
                    else:
                        y_label[t_value : min(start + x_t.size, sizer)] = 5.0
                        break
            else:
                y_label[start : min(start + x_t.size, sizer)] = 5.0

            # # if T wave inverted:
            # else:
            # 	y_label[start:min(start + x_t.size,sizer)] = 24.

            if include_points:
                # mark J point if no ST segment:
                if st_length == 0:
                    y_label[start] = 14.0

            gap = gaps[beat_counter]
            beat_start += gap
            beat_counter += 1
            y_label[min(start + x_t.size, sizer) : min(sizer, beat_start)] = 6.0

            # calculate end of QT interval using max slope method
            qt_end_set = False
            if x_t.size > 1 and start + x_t.size < sizer:
                if include_points:
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
                            y_label[start + qt_end] = 15.0
                            qt_end_set = True
                y_label[start + x_t.size] = 16.0  # 25.

        y = smooth_and_noise(
            y,
            rhythm="af",
            universal_noise_multiplier=universal_noise_multiplier,
            impedance=impedance,
        )
        X[h,] = x
        Y[h,] = y
        Y_label[h,] = y_label

    delay_start = random.randint(0, frequency - 1)
    X = X[:, delay_start : -(frequency - delay_start)]

    for lead in range(X.shape[0]):
        X[lead, :] = X[lead, :] - X[lead, 0]

    Y = Y[:, delay_start : -(frequency - delay_start)]
    Y_label = Y_label[:, delay_start : -(frequency - delay_start)]
    X_1000 = X
    Y_1000 = Y
    Y_label_1000 = Y_label

    # if signal_frequency < 1000:
    #     X_max = np.amax(X)
    #     Y_max = np.amax(Y)
    #     Y_label_max = np.amax(Y_label)
    #     X = X / X_max
    #     Y = Y / Y_max
    #     Y_label = Y_label / Y_label_max
    #     X = (resize(X, (X.shape[0], signal_frequency * 10)) * X_max)
    #     Y = (resize(Y, (Y.shape[0], signal_frequency * 10)) * Y_max)
    #     Y_label = (resize(Y_label, (Y_label.shape[0], signal_frequency * 10)) * Y_label_max).astype(int)

    return X, Y, Y_label, X_1000, Y_1000, Y_label_1000, meta_data, label_vector
