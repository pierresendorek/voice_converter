from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency

from feature_extraction.spectral_enveloppe_extractor import SpectralEnvelopeExtractor
from params.params import get_params
import os

from feature_extraction.voice_feature_extractor import VoiceFeatureExtractor
from alignment.dynamic_time_warping import get_dtw_matrix_and_corresponding_segments
from alignment.get_corresponding_segments_as_function import get_corresponding_segments_function
import numpy as np
from feature_extraction.rough_spectral_features_extractor import RoughSpectralEnvelopeExtractor

from common.get_element_from_list import get_element_from_list_constant_outside, get_element_from_list_zero_outside
from pprint import pprint


class PairSoundFeature:
    def __init__(self, params=None):

        bw_range_source = params["bw_range_source"]
        fw_range_source = params["fw_range_source"]
        bw_range_target = params["bw_range_target"]

        assert bw_range_source > 0
        assert fw_range_source > 0
        assert bw_range_target > 0

        self.params = params
        self.voice_feature_extractor = VoiceFeatureExtractor(params=params)
        self.rough_spectral_envelope_extractor = RoughSpectralEnvelopeExtractor(params=params)
        self.bw_range_source = bw_range_source
        self.fw_range_source = fw_range_source
        self.bw_range_target = bw_range_target
        self.feature_length = (params["n_triangle_function"] * 2 + 1) * (bw_range_source + fw_range_source + bw_range_target)


    def get_feature_length(self):
        return self.feature_length

    def get_sound_pair_features(self, source_path=None, target_path=None):

        source_sound, _ = get_mono_left_channel_sound_and_sampling_frequency(filename=source_path)
        target_sound, _ = get_mono_left_channel_sound_and_sampling_frequency(filename=target_path)

        rough_features_source = self.rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(source_sound)
        rough_features_target = self.rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(target_sound)

        (dtw_matrix,
         corresponding_segment_index_list,
         local_distance_matrix) = \
            get_dtw_matrix_and_corresponding_segments(rough_features_source,
                                                      rough_features_target,
                                                      distance_function=lambda x, y: np.linalg.norm(x - y))

        corresponding_segment = get_corresponding_segments_function(corresponding_segment_index_list)

        source_sound_features = self.voice_feature_extractor.extract(sound=source_sound)
        target_sound_features = self.voice_feature_extractor.extract(sound=target_sound)

        n_triangle_function = self.params["n_triangle_function"]

        source_feature_array_list = []
        target_feature_array_list = []

        for i in range(len(source_sound_features["period_list"])):
            delta_t = corresponding_segment(i) - corresponding_segment(i - 1)
            i_target = int(round(corresponding_segment(i)))

            feature_source_array = np.zeros([n_triangle_function * 2 + 1, self.fw_range_source + self.bw_range_source + self.bw_range_target])
            feature_target_array = np.zeros([n_triangle_function * 2 + 1 + 1])

            feature_target_array[0] = delta_t
            feature_target_array[1] = get_element_from_list_constant_outside(i, target_sound_features["period_list"])
            feature_target_array[2: 2 + n_triangle_function] = \
                get_element_from_list_constant_outside(i, target_sound_features["spectral_envelope_coeffs_harmonic_list"])

            feature_target_array[2 + n_triangle_function:2 + 2 * n_triangle_function] = \
                get_element_from_list_constant_outside(i, target_sound_features["spectral_envelope_coeffs_noise_list"])

            for k in range(-self.bw_range_source, self.fw_range_source):
                feature_source_array[0, k + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["period_list"])

                feature_source_array[1: 1 + n_triangle_function, k + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["spectral_envelope_coeffs_harmonic_list"])

                feature_source_array[1 + n_triangle_function:1 + 2 * n_triangle_function, k + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["spectral_envelope_coeffs_noise_list"])

            for k in range(-self.bw_range_target, 0):
                # We add sound features from the target speaker from previous time steps
                feature_source_array[0, - k - 1 + self.fw_range_source + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i_target + k, target_sound_features["period_list"])

                feature_source_array[1: 1 + n_triangle_function, - k - 1 + self.fw_range_source + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i_target + k, target_sound_features["spectral_envelope_coeffs_harmonic_list"])

                feature_source_array[1 + n_triangle_function:1 + 2 * n_triangle_function, - k - 1 + self.fw_range_source + self.bw_range_source] = \
                    get_element_from_list_constant_outside(i_target + k,
                                                           target_sound_features["spectral_envelope_coeffs_noise_list"])

            # print(feature_source_array)
            source_feature_array_list.append(feature_source_array)
            target_feature_array_list.append(feature_target_array)

        pair_feature_dict = {}

        pair_feature_dict["source_file"] = source_path
        pair_feature_dict["target_file"] = target_path
        pair_feature_dict["source_feature_array_list"] = source_feature_array_list
        pair_feature_dict["target_feature_array_list"] = target_feature_array_list

        return pair_feature_dict


if __name__ == "__main__":

    params = get_params()
    target_path = os.path.join(params["project_base_path"], "data/bruce_willis/Studio/1.wav")
    source_path = os.path.join(params["project_base_path"], "data/pierre_sendorek/Studio/1.wav")

    psf = PairSoundFeature(bw_range_source=5, fw_range_source=5, bw_range_target=5, params=params)

    d = psf.get_sound_pair_features(source_path=source_path, target_path=target_path)

    features_list = d["source_feature_array_list"]
    features_array = features_list[0]
    features_vector = np.reshape(features_array, [-1])

    pprint(features_vector)

    pprint(features_vector.shape)