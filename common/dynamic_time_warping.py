import numpy as np



def dtw_distance(s, t, distance_function):
    n = len(s)
    m=len(t)
    dtw_matrix = np.zeros([n, m])

    for i in range(n):
        dtw_matrix[i, 0] = float("inf")

    for i in range(m):
        dtw_matrix[0, i] = float("inf")

    dtw_matrix[0, 0] = 0

    for i in range(1, n):
        for j in range(1, m):
            cost = distance_function(s[i], t[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],         # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match

    return dtw_matrix[n-1, m-1]

