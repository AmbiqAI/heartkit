import random


def calculate_parameters(preset, rate):
    parameters = {}
    label_vector = []
    parameters["p_length"] = random.randint(80, 110)
    parameters["pr_interval"] = max(
        random.randint(80, 90),
        int(120 + (random.randint(0, 80))) - int(6.9 * ((rate - 60) / 10)),
    )
    parameters["qrs_duration"] = random.randint(50, 120)
    parameters["noisiness"] = random.randint(0, 30) * 0.0001
    if preset == "ant_STEMI":
        parameters["st_length"] = random.randint(50, 150)
    else:
        parameters["st_length"] = random.randint(20, 150)
    parameters["t_length"] = int(
        max(
            random.randint(420, 460) * ((60 / rate) ** 0.5),
            parameters["qrs_duration"] + parameters["st_length"] + 100,
        )
        - parameters["qrs_duration"]
        - parameters["st_length"]
    )
    parameters["qt"] = (
        parameters["qrs_duration"] + parameters["st_length"] + parameters["t_length"]
    )
    parameters["qtc"] = parameters["qt"] / ((60 / rate) ** 0.5)
    parameters["flippers"] = [1] * 3 + [-1] + [1] * 8
    parameters["p_voltages"] = [random.randint(1, 15) * 0.01 for _ in range(12)]
    parameters["p_biphasics"] = [random.randint(0, 1) for _ in range(12)]
    parameters["p_leans"] = [random.randint(0, 15) * 0.1 for _ in range(12)]
    parameters["q_depths"] = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0]
    parameters["r_heights"] = [random.randint(100, 300) * 0.01 for _ in range(12)]
    parameters["r_prime_presents"] = [False] * 12
    parameters["r_prime_heights"] = [0] * 12
    parameters["r_to_r_prime_duration_ratio"] = [1] * 12
    parameters["s_presents"] = [1] * 12
    parameters["s_depths"] = [
        random.randint(0, 50) * 0.01,
        random.randint(0, 50) * 0.01,
        0,
        random.randint(0, 50) * 0.01,
        random.randint(0, 50) * 0.01,
        random.randint(0, 50) * 0.01,
        random.randint(125, 175) * 0.01,
        random.randint(100, 150) * 0.01,
        random.randint(75, 125) * 0.01,
        random.randint(50, 100) * 0.01,
        random.randint(25, 75) * 0.01,
        random.randint(0, 50) * 0.01,
    ]
    parameters["s_prime_heights"] = [0] * 12
    parameters["s_to_qrs_duration_ratio"] = [1] * 12
    parameters["st_deltas"] = [0] * 12
    parameters["j_points"] = [0] * 12
    parameters["t_heights"] = [random.randint(5, 30) * 0.1 for _ in range(12)]
    parameters["t_leans"] = [random.randint(5, 10) * -0.1] * 12

    if preset == "LAHB":
        parameters["r_heights"] = [
            random.randint(100, 300) * 0.01,
            random.randint(10, 30) * 0.01,
            random.randint(10, 30) * 0.01,
            random.randint(100, 150) * 0.01,
            random.randint(100, 300) * 0.01,
            random.randint(10, 30) * 0.01,
            0,
            0,
            0,
            random.randint(10, 25) * 0.01,
            random.randint(25, 50) * 0.01,
            random.randint(50, 150) * 0.01,
        ]
        parameters["s_presents"] = [0, 1, 1, 0, 0] + [1] * 7
        parameters["s_depths"] = [
            0,
            random.randint(100, 300) * 0.01,
            random.randint(300, 500) * 0.01,
            0,
            0,
            random.randint(200, 400) * 0.01,
            random.randint(200, 400) * 0.01,
            random.randint(200, 400) * 0.01,
            random.randint(200, 400) * 0.01,
            random.randint(150, 300) * 0.01,
            random.randint(100, 150) * 0.01,
            random.randint(50, 100) * 0.01,
        ]

    elif preset == "LPHB":
        parameters["r_heights"] = [random.randint(0, 50) * 0.01]
        parameters["r_heights"] += [random.randint(100, 300) * 0.01 for _ in range(3)]
        parameters["r_heights"] += [random.randint(0, 50) * 0.01]
        parameters["r_heights"] += [random.randint(100, 300) * 0.01 for _ in range(7)]
        parameters["s_depths"] = [
            random.randint(50, 200) * 0.01,
            random.randint(0, 50) * 0.01,
            0,
            random.randint(50, 200) * 0.01,
            random.randint(0, 50) * 0.01,
            random.randint(0, 50) * 0.01,
            random.randint(125, 175) * 0.01,
            random.randint(100, 150) * 0.01,
            random.randint(75, 125) * 0.01,
            random.randint(50, 100) * 0.01,
            random.randint(25, 75) * 0.01,
            random.randint(0, 50) * 0.01,
        ]

    elif preset == "high_take_off":
        parameters["st_length"] = 20
        parameters["j_points"] = [random.randint(0, 15) * 0.01 for _ in range(12)]
        parameters["t_heights"] = [
            random.randint(int(i * 100) + 20, int(i * 100) + 50) * 0.1
            for i in parameters["j_points"]
        ]
        parameters["t_leans"] = [random.randint(2, 4) * 0.1] * 12

    elif preset == "LBBB":
        parameters["qrs_duration"] = random.randint(160, 220)
        parameters["st_length"] = random.randint(20, 100)
        if parameters["st_length"] < 5:
            parameters["st_length"] = 0
        parameters["t_length"] = int(
            max(
                (random.randint(420, 460) + (parameters["qrs_duration"] - 120))
                * ((60 / rate) ** 0.5),
                parameters["qrs_duration"]
                + parameters["st_length"]
                + 100
                + parameters["qrs_duration"],
            )
            - parameters["qrs_duration"]
            - parameters["st_length"]
        )
        parameters["qt"] = (
            parameters["qrs_duration"]
            + parameters["st_length"]
            + parameters["t_length"]
        )
        parameters["qtc"] = parameters["qt"] / ((60 / rate) ** 0.5)
        parameters["q_depths"] = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0]
        parameters["r_heights"] = [
            random.randint(50, 200) * 0.01,
            random.randint(20, 60) * 0.01,
            random.randint(20, 60) * 0.01,
            random.randint(50, 200) * 0.01,
            random.randint(100, 200) * 0.01,
            random.randint(20, 100) * 0.01,
            random.randint(20, 80) * 0.01,
            random.randint(15, 25) * 0.01,
            random.randint(20, 30) * 0.01,
            random.randint(10, 25) * 0.01,
            random.randint(20, 100) * 0.01,
            random.randint(100, 200) * 0.01,
        ]
        parameters["r_prime_presents"] = [True] * 10 + [False] + [True]
        v1_depth = random.randint(100, 200) * 0.01
        v2_depth = v1_depth * random.uniform(1, 1.3)
        v3_depth = v1_depth * random.uniform(1.2, 1.5)
        v4_depth = v1_depth * random.uniform(0.7, 1.1)
        parameters["r_prime_heights"] = [
            parameters["r_heights"][0] - (random.randint(10, 50) * 0.01),
            parameters["r_heights"][1] + (parameters["r_heights"][1] * 0.8),
            random.randint(40, 80) * -0.01,
            parameters["r_heights"][3] - (random.randint(10, 50) * 0.01),
            parameters["r_heights"][4] - (random.randint(10, 50) * 0.01),
            random.randint(-100, 100) * 0.01,
            (v1_depth + random.uniform(0, 0.1)) * -1,
            (v2_depth + random.uniform(0, 0.1)) * -1,
            (v3_depth + random.uniform(0, 0.1)) * -1,
            (v4_depth + random.uniform(0, 0.1)) * -1,
            random.randint(-100, 100) * 0.01,
            parameters["r_heights"][11] - (random.randint(10, 50) * 0.01),
        ]
        parameters["r_to_r_prime_duration_ratio"] = [random.randint(15, 25) * 0.1] * 12
        parameters["s_presents"] = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        parameters["s_depths"] = [
            0,
            random.randint(20, 60) * 0.01,
            -(parameters["r_prime_heights"][2] - (random.randint(10, 30) * 0.01)),
            0,
            0,
            random.randint(0, 200) * 0.01,
            0,
            0,
            0,
            0,
            random.randint(0, 200) * 0.01,
            0,
        ]
        parameters["s_prime_heights"] = [
            parameters["r_prime_heights"][0]
            - ((parameters["r_heights"][0] - parameters["r_prime_heights"][0]) / 2)
        ]
        parameters["s_prime_heights"] += [
            (parameters["r_heights"][1] + parameters["r_prime_heights"][1]) / 2
        ]
        parameters["s_prime_heights"] += [
            -(parameters["r_prime_heights"][2] - (random.randint(10, 30) * 0.01))
        ]
        parameters["s_prime_heights"] += [
            parameters["r_prime_heights"][3]
            - ((parameters["r_heights"][3] - parameters["r_prime_heights"][3]) / 2)
        ]
        parameters["s_prime_heights"] += [
            parameters["r_prime_heights"][4]
            - (
                (parameters["r_heights"][4] - parameters["r_prime_heights"][4])
                / random.randint(2, 4)
            )
        ]
        parameters["s_prime_heights"] += [
            random.uniform(parameters["r_heights"][5], parameters["r_prime_heights"][5])
        ]
        parameters["s_prime_heights"] += [(v1_depth + random.uniform(0, 0.1)) * -1]
        parameters["s_prime_heights"] += [(v2_depth + random.uniform(0, 0.1)) * -1]
        parameters["s_prime_heights"] += [(v3_depth + random.uniform(0, 0.1)) * -1]
        parameters["s_prime_heights"] += [(v4_depth + random.uniform(0, 0.1)) * -1]
        parameters["s_prime_heights"] += [
            random.uniform(
                parameters["r_heights"][10], parameters["r_prime_heights"][5]
            )
        ]
        parameters["s_prime_heights"] += [
            parameters["r_prime_heights"][11]
            - (
                (parameters["r_heights"][11] - parameters["r_prime_heights"][11])
                / random.randint(2, 4)
            )
        ]
        parameters["s_to_qrs_duration_ratio"] = [1, 1, 2] + [1] * 9
        parameters["st_deltas"] = [0] * 12
        parameters["j_points"] = []
        parameters["t_heights"] = []
        for i in range(12):
            mx = max(
                [
                    parameters["r_heights"][i],
                    parameters["r_prime_heights"][i],
                    -parameters["s_depths"][i],
                    parameters["s_prime_heights"][i],
                ],
                key=abs,
            )
            t = random.uniform(0, mx * -0.2)
            parameters["j_points"].append(t)
            parameters["t_heights"].append(t * random.randint(20, 30))
        parameters["t_leans"] = [random.randint(5, 10) * -0.1] * 12

    elif preset == "ant_STEMI":
        parameters["s_presents"] = [0, 1, 1, 1, 0] + [1] * 7
        parameters["j_points"] = [
            random.randint(2, 6) * 0.1,
            random.randint(1, 20) * 0.01,
            random.randint(-4, -1) * 0.1,
            random.randint(1, 3) * 0.1,
            random.randint(2, 6) * 0.1,
            random.randint(1, 20) * 0.01,
            random.randint(2, 6) * 0.1,
            random.randint(2, 6) * 0.1,
            random.randint(2, 6) * 0.1,
            random.randint(2, 6) * 0.1,
            random.randint(2, 6) * 0.1,
            random.randint(2, 6) * 0.1,
        ]
        parameters["t_heights"] = [
            parameters["j_points"][i] * (random.randint(10, 30)) for i in range(12)
        ]

    elif preset == "random_morphology":
        parameters["q_depths"] = [random.uniform(0, 0.2) for _ in range(12)]
        parameters["pr_interval"] = random.randint(80, 110)
        parameters["qrs_duration"] = random.randint(50, 220)
        parameters["r_prime_presents"] = [
            bool(random.getrandbits(1)) for _ in range(12)
        ]
        parameters["r_prime_heights"] = [
            parameters["r_heights"][num] - (random.randint(-50, 50) * 0.01)
            for num in range(12)
        ]
        parameters["s_prime_heights"] = [
            parameters["r_heights"][num] - (random.randint(-50, 50) * 0.01)
            for num in range(12)
        ]
        parameters["s_presents"] = [0] * 12
        parameters["j_points"] = [random.randint(-6, 6) * 0.1 for _ in range(12)]
        parameters["t_heights"] = [random.randint(-30, 30) * 0.1 for _ in range(12)]

        for p in parameters["q_depths"]:
            if p > 0.01:
                label_vector.append(1)
            else:
                label_vector.append(0)
        if parameters["p_length"] + parameters["pr_interval"] > 200:
            label_vector.append(1)
        else:
            label_vector.append(0)
        if parameters["qrs_duration"] > 120:
            label_vector.append(1)
        else:
            label_vector.append(0)
        for p in parameters["r_prime_presents"]:
            label_vector.append(p * 1)
        for p in parameters["j_points"]:
            if p > 0.2:
                label_vector.append(1)
            else:
                label_vector.append(0)
        for p in parameters["j_points"]:
            if p < -0.2:
                label_vector.append(1)
            else:
                label_vector.append(0)
        for p in parameters["t_heights"]:
            if p < 0:
                label_vector.append(1)
            else:
                label_vector.append(0)

    return parameters, label_vector
