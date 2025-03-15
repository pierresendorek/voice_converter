import numpy as np

from feature_extraction.common_arrays import CommonArrays
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.spectral_envelope import SpectralEnvelopeExtractor
from typing import List
from common.parameters import Parameters



def sound_segments_iterator(sound, segment_len, n_gap):
    i_segment = 0
    while i_segment * n_gap + segment_len <= len(sound):
        yield sound[i_segment * n_gap: i_segment * n_gap + segment_len]
        i_segment += 1


class Feature:
    def __init__(self, 
                 period, 
                 spectral_envelope_coeffs_harmonic, 
                 spectral_envelope_coeffs_noise):
        self.period = period
        self.spectral_envelope_coeffs_harmonic = spectral_envelope_coeffs_harmonic
        self.spectral_envelope_coeffs_noise = spectral_envelope_coeffs_noise

    @classmethod
    def from_numpy(cls, features:np.ndarray, params:Parameters):
        return cls(period=features[0], 
                   spectral_envelope_coeffs_harmonic=features[1:1+params.n_triangle_function], 
                   spectral_envelope_coeffs_noise=features[1+params.n_triangle_function:])

    def numpy(self) -> np.ndarray:
        return np.concatenate([np.array([self.period]), 
                               self.spectral_envelope_coeffs_harmonic, 
                               self.spectral_envelope_coeffs_noise])
        

def extract_features(sound: np.ndarray, params: Parameters):
    common_arrays = CommonArrays(params)
        
    pitch_estimator = PitchEstimator(params)
    periodic_and_noise_separator = PeriodicAndNoiseSeparator(params)
    spectral_envelope_extractor = SpectralEnvelopeExtractor(params)

    for x in sound_segments_iterator(sound, params.segment_len, params.n_gap):
        
        x_apodized = x * common_arrays.apowin2
        period = pitch_estimator.estimate_period(x_apodized)
        periodic, noise = periodic_and_noise_separator.separate_components(x_apodized=x_apodized, period=period)
        spectral_envelope_coeffs_periodic = spectral_envelope_extractor.get_coeffs(periodic)
        spectral_envelope_coeffs_noise = spectral_envelope_extractor.get_coeffs(noise)

        yield Feature(period, 
                      spectral_envelope_coeffs_periodic, 
                      spectral_envelope_coeffs_noise)
        

def extract_features_as_np_array(sound: np.ndarray, params: Parameters) -> np.ndarray:
    return np.array([feature.numpy() for feature in extract_features(sound, params)])



if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path
    from scipy.io import wavfile

    parser = ArgumentParser()
    parser.add_argument("sound_file", type=Path)
    args = parser.parse_args()

    fs, x = wavfile.read(args.sound_file)
    
    if x.shape[1] == 2:
        x = x[:, 0]

    features = extract_features(x, Parameters())

    for feature in features:
        print(feature.numpy())
