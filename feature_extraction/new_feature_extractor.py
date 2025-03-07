import numpy as np

from feature_extraction.common_arrays import CommonArrays
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.spectral_envelope import SpectralEnvelopeExtractor
from typing import List


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
    
    def __post_init__(self):
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


def sound_segments_iterator(sound, segment_len, n_gap):
    i_segment = 0
    while i_segment * n_gap + segment_len <= len(sound):
        yield sound[i_segment * n_gap: i_segment * n_gap + segment_len]
        i_segment += 1


class Feature:
    def __init__(self, period, spectral_envelope_coeffs_harmonic, spectral_envelope_coeffs_noise):
        self.period = period
        self.spectral_envelope_coeffs_harmonic = spectral_envelope_coeffs_harmonic
        self.spectral_envelope_coeffs_noise = spectral_envelope_coeffs_noise

    def numpy(self):
        return np.concatenate([np.array([self.period]), self.spectral_envelope_coeffs_harmonic, self.spectral_envelope_coeffs_noise])
        

def extract_features(sound: np.ndarray, params: Parameters):
        
    common_arrays = CommonArrays(params)
        
    pitch_estimator = PitchEstimator(params)
    periodic_and_noise_separator = PeriodicAndNoiseSeparator(params)
    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)

    features: List[Feature] = []

    for i_x, x in enumerate(sound_segments_iterator(sound, params.segment_len, params.n_gap)):
        
        x_apodized = x * common_arrays.apowin2
        period = pitch_estimator.estimate_period(x_apodized)
        periodic, noise = periodic_and_noise_separator.separate_components(x_apodized=x_apodized, period=period)
        spectral_envelope_coeffs_periodic = spectral_envelope_extractor.get_coeffs(periodic)
        spectral_envelope_coeffs_noise = spectral_envelope_extractor.get_coeffs(noise)

        features.append(Feature(period, spectral_envelope_coeffs_periodic, spectral_envelope_coeffs_noise))

    return features


if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path
    from scipy.io import wavfile

    parser = ArgumentParser()
    parser.add_argument("sound_file", type=Path)
    args = parser.parse_args()

