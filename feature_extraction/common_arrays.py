import numpy as np
from params.params import Parameters

class CommonArrays:
    def __init__(self, params: Parameters):
        self.n_gap = params.segment_len // 4 # 512 samples @ 44100Hz ~ 0.011 of a second
        self.apowin = np.sin(np.linspace(0, np.pi, num=params.segment_len, endpoint=False))
        self.apowin2 = self.apowin ** 2