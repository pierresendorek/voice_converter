
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import lsqr
from scipy.sparse import linalg
import scipy.io.wavfile
import os


package_directory = os.path.dirname(os.path.abspath(__file__))

# This class contains all variables needed for sound analysis, as well as methods for extracting features.
class SpectralEnvelopeExtractor:

    def __init__(self, params):

        # number of elementary _triangle functions
        self.n_triangle_func = params['n_triangle_function']

        self.sampling_frequency = params["sampling_frequency"]

        # nFFT is the number samples taken to slice the audio into chunks for FFT analysis
        # When the sampling frequency is the default 44100, nFFT is 2048.
        # When the sampling rate is lower/higher, nFFT will be lower/higher.
        self.nFFT = params["segment_len"]

        # Gap between slices of the audio
        self.nGap = params["n_gap"]

        self.apowin = params["apowin"]
        self.apowin2 = params["apowin2"]


        self.fq_elem_func_min = params["fq_elem_func_min"]
        self.fq_elem_func_max = params["fq_elem_func_max"]

        self.fq_elem_func = np.exp(np.linspace(np.log(self.fq_elem_func_min),
                                               np.log(self.fq_elem_func_max),
                                               self.n_triangle_func + 2))

        self.triangle_window_matrix = self._create_triangle_window_matrix()

        #self.pseudo_inverse_transpose_triangle_windows_matrix = np.linalg.pinv(self.triangle_window_matrix.T)

        #self.projector_triangle_windows_matrix = np.linalg.pinv((self.triangle_window_matrix.T).dot(self.triangle_window_matrix).todense())


    def _triangle(self, x, xLeftZero, xTop, xRightZero):
        """
        Given three points on the abscissa (x, xLeftZero, xRightZero),
        constructs a _triangle-shape function that is non-zero inside [xLeftZero, xRightZero]
        This is used an interpolation function which returns a value given x inside [xLeftZero, xRightZero]
        :param x: float point where the function is evaluated
        :param xLeftZero: abscissa of the left point of the _triangle
        :param xTop: abscissa of the top point of the _triangle
        :param xRightZero: abscissa of the right point of the _triangle
        :return: float
        """
        if x < xLeftZero or x > xRightZero:
            return 0
        elif x < xTop:
            return (x - xLeftZero) / (xTop - xLeftZero)
        else:
            return (xRightZero - x) / (xRightZero - xTop)

    # Zero-centered interpolation
    def _interpolator(self, phi):
        return self._triangle(phi, -1, 0, 1)


    def _create_triangle_window_matrix(self):
        A = np.zeros((int(self.nFFT//2), self.n_triangle_func))
        for i in range(self.n_triangle_func):
            i_elem_func = i + 1
            fqC = self.fq_elem_func[i_elem_func]
            fqL = self.fq_elem_func[i_elem_func - 1]
            fqR = self.fq_elem_func[i_elem_func + 1]

            # only filling in non-zero values
            normalizedFqL = fqL * self.nFFT / self.sampling_frequency
            normalizedFqR = fqR * self.nFFT / self.sampling_frequency
            normalizedFqC = fqC * self.nFFT / self.sampling_frequency
            for iFq in range(int(np.floor(normalizedFqL)), int(min([np.ceil(normalizedFqR), self.nFFT//2]))):
                A[iFq, i] = self._triangle(iFq, normalizedFqL, normalizedFqC, normalizedFqR)

            # normalizing each row
            v = np.linalg.norm(A[:, i])
            if v > 0:
                A[:, i] = A[:, i] / v
        return bsr_matrix(A)


    def get_spectrum(self, x_apodized=None):
        abs_fft_x_apodized = np.abs(np.fft.fft(x_apodized))
        spectrum = abs_fft_x_apodized[0:int(self.nFFT // 2)]
        return spectrum

    def get_spectral_envelope_coeffs(self, x_apodized=None):
        """
        :param x_apodized: sound segment multiplied by apowin2
        :return: average energy per frequency band
        """
        spectrum = self.get_spectrum(x_apodized)
        spectral_envelope_coeffs = (self.triangle_window_matrix.T).dot(spectrum)
        return spectral_envelope_coeffs

    def get_spectral_enveloppe_from_coeffs(self, spectral_envelope_coeffs):

        return lsqr(self.triangle_window_matrix.T, spectral_envelope_coeffs, damp=1E-9)[0]


    def get_full_spectral_envelope_from_coeffs(self, spectal_envelope_coeffs):

        half_spectral_envelope = self.get_spectral_enveloppe_from_coeffs(spectal_envelope_coeffs)
        full_spectral_envelope = np.zeros(self.nFFT)
        full_spectral_envelope[0:self.nFFT//2] = half_spectral_envelope
        full_spectral_envelope[self.nFFT//2:self.nFFT] = half_spectral_envelope[::-1]

        return full_spectral_envelope

