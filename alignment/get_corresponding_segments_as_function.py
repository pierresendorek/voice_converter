import numpy as np


def a_to_average_b_function(a, a_to_average_b):
    if a in a_to_average_b.keys():
        return a_to_average_b[a]
    else:
        sorted_keys_list = sorted(a_to_average_b.keys())
        idx_min = np.argmin([np.abs(a - k) for k in sorted_keys_list])
        return a_to_average_b[sorted_keys_list[idx_min]]


def get_corresponding_segments_function(corresponding_segments):

    a_to_b_list = {}

    for ab in corresponding_segments:
        a, b = ab
        b_list = a_to_b_list.get(a, [])
        b_list.append(b)
        a_to_b_list[a] = b_list

    print(a_to_b_list)

    a_to_average_b = {}

    for a in a_to_b_list.keys():
        b_list = a_to_b_list[a]
        a_to_average_b[a] = np.mean(b_list)


    return lambda a : a_to_average_b_function(a, a_to_average_b)

