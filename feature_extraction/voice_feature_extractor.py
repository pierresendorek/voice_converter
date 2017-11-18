from feature_extraction.pitch_estimator import PitchEstimator
from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from feature_extraction.periodic_and_noise_separator import PeriodicAndNoiseSeparator
from common.count_segments import count_segments
from database_tools.sound_file_loader import get_segment
import numpy as np


class VoiceFeatureExtractor:

    def __init__(self, params):
        self.pitch_extractor = PitchEstimator(params=params)
        self.period_and_noise_separator = PeriodicAndNoiseSeparator(params=params)
        self.spectral_envelope_extractor = SpectralEnvelopeExtractor(params=params)
        self.params = params


    def extract(self, sound=None):

        apowin2 = self.params["apowin2"]

        n_segment = count_segments(sound=sound, params=self.params)

        period_list = []
        spectral_envelope_coeffs_noise_list = []
        spectral_envelope_coeffs_harmonic_list = []

        sound_out = np.zeros(sound.shape)

        for i_segment in range(n_segment):
            x = get_segment(sound=sound,  i_segment=i_segment, params=self.params)
            x_apodized = x * apowin2
            period = self.pitch_extractor.estimate_period(x_apodized)
            periodic, noise = self.period_and_noise_separator.separate_components(x_apodized=x_apodized, period=period)
            spectral_envelope_coeffs_periodic = self.spectral_envelope_extractor.get_spectral_envelope_coeffs(periodic)
            spectral_envelope_coeffs_noise = self.spectral_envelope_extractor.get_spectral_envelope_coeffs(noise)

            spectral_envelope_coeffs_noise_list.append(spectral_envelope_coeffs_noise)
            spectral_envelope_coeffs_harmonic_list.append(spectral_envelope_coeffs_periodic)
            period_list.append(period)

        return {"period_list": period_list,
                "spectral_envelope_coeffs_harmonic_list": spectral_envelope_coeffs_harmonic_list,
                "spectral_envelope_coeffs_noise_list": spectral_envelope_coeffs_noise_list}


    def extract_as_array(self, sound=None):

        feature_dict = self.extract(sound=sound)

        period_array = np.array(feature_dict["period_list"]).reshape([-1, 1]) # shape = [n_segment, 1]
        spectral_envelope_coeffs_harmonic_array = np.array(feature_dict["spectral_envelope_coeffs_harmonic_list"]) # shape = [n_segment, n_triangle_function]
        spectral_envelope_coeffs_noise_array = np.array(feature_dict["spectral_envelope_coeffs_noise_list"]) # shape = [n_segment, n_triangle_function]

        feature_array = np.concatenate([period_array,
                                        spectral_envelope_coeffs_harmonic_array,
                                        spectral_envelope_coeffs_noise_array], axis=1)

        return feature_array