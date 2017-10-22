import numpy as np


def get_params():
    params = {
        "sampling_frequency": 44100,
        "segment_len": 2048,
        "n_gap": 2048 // 4,
        "fq_elem_func_min": 50.0,
        "fq_elem_func_max": 22050.0,
        "fq_voice_min": 70.0,
        "fq_voice_max": 400.0,
        "n_triangle_function": 20,
        "verbose": True
    }

    params["apowin"] = np.sin(np.linspace(0, np.pi, num=params["segment_len"], endpoint=False))
    params["apowin2"] = params["apowin"] ** 2

    return params

