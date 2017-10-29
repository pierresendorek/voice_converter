import numpy as np


def get_dtw_matrix(s, t, distance_function):
    n = len(s)
    m = len(t)

    dtw_matrix = np.zeros([n, m])

    local_distance_matrix = np.zeros([n, m])

    for i in range(1, n):
        dtw_matrix[i, 0] = float("inf")

    for i in range(1, m):
        dtw_matrix[0, i] = float("inf")

    dtw_matrix[0, 0] = 0

    corresponding_segment_index_list = []

    for i in range(1, n):
        for j in range(1, m):
            cost = distance_function(s[i], t[j])
            local_distance_matrix[i, j] = cost

            possible_prev = [dtw_matrix[i-1, j],         # insertion
                             dtw_matrix[i, j-1],    # deletion
                             dtw_matrix[i-1, j-1]]

            dtw_matrix[i, j] = cost + min(possible_prev)


    # tracing back the optimal path
    [i, j] = [n-1, m-1]
    while(i>0 or j>0):
        corresponding_segment_index_list.append([i, j])
        possible_prev = [dtw_matrix[i - 1, j],  # insertion
                         dtw_matrix[i, j - 1],  # deletion
                         dtw_matrix[i - 1, j - 1]]
        idx = np.argmin(possible_prev)

        if idx == 0:
            i = i - 1
        elif idx == 1:
            j = j - 1
        else:
            i, j = i-1, j-1

    return dtw_matrix, corresponding_segment_index_list, local_distance_matrix






