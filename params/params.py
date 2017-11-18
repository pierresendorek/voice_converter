import numpy as np


def get_params():
    params = {
        "project_base_path": "/Users/pierresendorek/projets/voice_converter",
        "temp_path":"/Users/pierresendorek/temp/",
        "sampling_frequency": 44100,
        "segment_len": 2048,
        "fq_elem_func_min": 50.0,
        "fq_elem_func_max": 22050.0,
        "fq_voice_min": 70.0,
        "fq_voice_max": 300.0,
        "n_triangle_function": 40,
        "verbose": True
    }

    params["n_gap"] = params["segment_len"] // 4 # 512 samples @ 44100Hz ~ 0.011 of a second

    params["apowin"] = np.sin(np.linspace(0, np.pi, num=params["segment_len"], endpoint=False))
    params["apowin2"] = params["apowin"] ** 2

    # Triangle linear interpolation function
    triangle = np.zeros(params["segment_len"])
    beginning, ending = params["n_gap"], params["segment_len"]//2
    triangle[beginning:ending] = np.linspace(0, 1, ending - beginning)
    beginning, ending = ending, params["segment_len"] // 2 + params["n_gap"]
    triangle[beginning:ending] = np.linspace(1, 0, ending - beginning)

    params["triangle_lin_interpol"] = triangle

    return params

