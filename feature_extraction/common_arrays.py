import numpy as np
from common.parameters import Parameters

class CommonArrays:
    def __init__(self, params: Parameters):
        self.n_gap = params.segment_len // 4 # 512 samples @ 44100Hz ~ 0.011 of a second
        self.apowin = np.sin(np.linspace(0, np.pi, num=params.segment_len, endpoint=False))
        self.apowin2 = self.apowin ** 2

        

    # def triangle(n_gap, segment_len, n_triangle_function):
#     triangle = np.zeros(segment_len)
#     beginning, ending = n_gap, segment_len // 2
#     triangle[beginning:ending] = np.linspace(0, 1, ending - beginning)
#     beginning, ending = ending, segment_len // 2 + n_gap
#     triangle[beginning:ending] = np.linspace(1, 0, ending - beginning)
#     return triangle
