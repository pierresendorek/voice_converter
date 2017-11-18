from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from copy import deepcopy
from common.get_element_from_list import get_element_from_list_constant_outside
import numpy as np

class RoughSpectralEnvelopeExtractor:

    def __init__(self, params=None):
        self.params = deepcopy(params)
        self.params["n_triangle_function"] = 15
        self.spectral_features_extractor = SpectralEnvelopeExtractor(params)

    def get_spectral_envelope_from_sound(self, sound=None):
        rough_spectral_envelope_list = self.spectral_features_extractor.get_spectral_envelope_from_sound(sound=sound)
        rough_spectral_envelope_and_delta_list = []
        for it in range(len(rough_spectral_envelope_list)):
            log_rough_spectral_envelope_prev = np.log(get_element_from_list_constant_outside(it, rough_spectral_envelope_list) + 1E-6)
            log_rough_spectral_envelope = np.log(get_element_from_list_constant_outside(it-1, rough_spectral_envelope_list) + 1E-6)
            derivative_log_rough_spectral_envelope = log_rough_spectral_envelope - log_rough_spectral_envelope_prev
            feature_vector = np.concatenate([log_rough_spectral_envelope,
                                             derivative_log_rough_spectral_envelope], axis=0)

            rough_spectral_envelope_and_delta_list.append(feature_vector)


        return rough_spectral_envelope_and_delta_list