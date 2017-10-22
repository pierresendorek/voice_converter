import numpy as np

def set_segment(i_segment=None, source=None, dest=None, n_gap=None):
    i_start = n_gap * i_segment
    i_end = i_start + source.shape[0]

    if i_start >= 0 and i_end < dest.len:
        dest[i_start:i_end] = source


def add_to_segment(i_segment=None, source=None, dest=None, n_gap=None):
    i_start = n_gap * i_segment
    i_end = i_start + source.shape[0]

    if i_start >= 0 and i_end < dest.shape[0]:
        dest[i_start:i_end] += source
