import numpy as np
from scipy.sparse import linalg

from feature_extraction.common_arrays import CommonArrays
from feature_extraction.parameters import Parameters


class PitchEstimator:

    def __init__(self, params: Parameters):
        self.params = params

        common_arrays = CommonArrays(params)

        self.apowin = common_arrays.apowin
        self.apowin2 = common_arrays.apowin2

        # Gap between slices of the audio

        # We're only interested in the pitches of the spoken voice
        # range of pitches (Hz)
        # corresponding range of periods (expressed in number of samples)
        self.period_min = round(params.sampling_frequency / params.fq_voice_max)
        self.period_max = round(params.sampling_frequency / params.fq_voice_min)

        # table of regularly spaced periods (expressed in number of samples)
        # Each time delay in this vector is a candidate period
        self.period_list = np.arange(self.period_min, self.period_max)

        ## corresponding frequencies for the table of regularly spaced periods
        #self.frequency_list = np.array(params.sampling_frequency / self.period_list)

        self.diff_for_period = np.zeros(self.period_max - self.period_min)


    def estimate_period(self, x:np.ndarray) -> int:
        return self.estimate_period_least_difference_FFT(x)


    def estimate_period_least_difference_FFT(self, x) -> int:
        """
        Returns the period as an amount of samples
        :param self:
        :param x: x is a segment of length self.nFFT
        :return:
        """

        xApo = x * self.apowin2
        cumsum_x2 = np.cumsum((xApo) ** 2)
        terme_croise = np.correlate(xApo, xApo, mode="full")

        N = self.params.segment_len
        for i_period in range(1, self.period_max - self.period_min):
            period = self.period_min + i_period
            self.diff_for_period[i_period] = (cumsum_x2[int(N - 1 - period)] + cumsum_x2[int(N - 1)] - cumsum_x2[
                int(period - 1)] - 2 * terme_croise[int(N - 1 + period)]) / (N - period)
        return np.argmin(self.diff_for_period[1:]) + self.period_min + 1




