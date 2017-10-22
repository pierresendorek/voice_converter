import numpy as np
from scipy.sparse import linalg

# TODO : try to compile this class

class PitchEstimator:

    def __init__(self, params=None):

        self.sampling_frequency = params["sampling_frequency"]

        # number of points used for the FFT or correlation
        self.segment_len = params["segment_len"]

        self.apowin = np.sin(np.linspace(0, np.pi, num=self.segment_len, endpoint=False))
        self.apowin2 = self.apowin ** 2

        # Gap between slices of the audio
        self.n_gap = params["n_gap"]

        # We're only interested in the pitches of the spoken voice
        # range of pitches (Hz)
        self.fq_voice_min = params["fq_voice_min"]
        self.fq_voice_max = params["fq_voice_max"]

        # corresponding range of periods (expressed in number of samples)
        self.period_min = round(self.sampling_frequency / self.fq_voice_max)
        self.period_max = round(self.sampling_frequency / self.fq_voice_min)

        # table of regularly spaced periods (expressed in number of samples)
        # Each time delay in this vector is a candidate period
        self.period_list = np.arange(self.period_min, self.period_max)

        # corresponding frequencies for the table of regularly spaced periods
        self.frequency_list = np.array(self.sampling_frequency / self.period_list)

        self.diffForPeriod = np.zeros(self.period_max - self.period_min)


    def estimate_period(self, x):
        return self.estimate_period_least_difference_FFT(x)


    def estimate_period_least_difference_FFT(self, x):
        """
        Returns the period as an amount of samples
        :param self:
        :param x: x is a segment of length self.nFFT
        :return:
        """

        xApo = x * self.apowin2
        cumsum_x2 = np.cumsum((xApo) ** 2)
        terme_croise = np.correlate(xApo, xApo, mode="full")

        N = self.segment_len
        for iPeriod in range(1, self.period_max - self.period_min):
            period = self.period_min + iPeriod
            self.diffForPeriod[iPeriod] = (cumsum_x2[int(N - 1 - period)] + cumsum_x2[int(N - 1)] - cumsum_x2[
                int(period - 1)] - 2 * terme_croise[int(N - 1 + period)]) / (N - period)
        return np.argmin(self.diffForPeriod[1:]) + self.period_min + 1




