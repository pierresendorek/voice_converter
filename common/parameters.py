


class Parameters:
    def __init__(self):
        self.temp_path = "/tmp/"
        self.sampling_frequency = 44100
        self.segment_len = 2048
        self.fq_elem_func_min = 50.0
        self.fq_elem_func_max = 22050.0
        self.fq_voice_min = 70.0
        self.fq_voice_max = 300.0
        self.n_triangle_function = 40
        self.verbose = True
        self.use_gpu = False

        ######################
        # computed attributes
        ######################
        self.n_gap = self.segment_len // 4
        # corresponding range of periods (expressed in number of samples)
        self.period_min = round(self.sampling_frequency / self.fq_voice_max)
        self.period_max = round(self.sampling_frequency / self.fq_voice_min)

    
        

# def triangle(n_gap, segment_len, n_triangle_function):
#     triangle = np.zeros(segment_len)
#     beginning, ending = n_gap, segment_len // 2
#     triangle[beginning:ending] = np.linspace(0, 1, ending - beginning)
#     beginning, ending = ending, segment_len // 2 + n_gap
#     triangle[beginning:ending] = np.linspace(1, 0, ending - beginning)
#     return triangle
