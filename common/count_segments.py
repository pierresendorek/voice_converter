import numpy as np


def count_segments(sound=None, params=None):
    samples_per_segment = params["segment_len"]
    n_gap = params["n_gap"]
    return int(np.floor((len(sound) - samples_per_segment)/n_gap))