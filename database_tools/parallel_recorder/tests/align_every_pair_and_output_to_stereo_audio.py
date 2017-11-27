from copy import deepcopy

from alignment.dynamic_time_warping import get_dtw_matrix_and_corresponding_segments
from alignment.get_corresponding_segments_as_function import get_corresponding_segments_function
from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
from feature_extraction.rough_spectral_features_extractor import RoughSpectralEnvelopeExtractor
from params.params import get_params
import numpy as np
from common.math.piecewise_linear_function import PiecewiseLinearFunction
from pprint import pprint
from feature_extraction.voice_feature_extractor import VoiceFeatureExtractor
from synthesis.voice_synthesizer import synthesize_voice
from scipy.io import wavfile

params = get_params()


parallel_corpus_maker = ParallelCorpusMaker(params=params)
d_full_path_list = parallel_corpus_maker.get_full_path_lists()

rough_spectral_envelope_extractor = RoughSpectralEnvelopeExtractor(params=params)
voice_feature_extractor = VoiceFeatureExtractor(params=params)


for i_file in range(len(d_full_path_list["pierre_sendorek_full_path_list"])):

    if True or i_file == 0:

        source_filepath = d_full_path_list["pierre_sendorek_full_path_list"][i_file]
        target_filepath = d_full_path_list["bruce_willis_full_path_list"][i_file]

        source_sound, _ = get_mono_left_channel_sound_and_sampling_frequency(filename=source_filepath)
        target_sound, _ = get_mono_left_channel_sound_and_sampling_frequency(filename=target_filepath)

        rough_features_source = rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(source_sound)
        rough_features_target = rough_spectral_envelope_extractor.get_spectral_envelope_from_sound(target_sound)

        (_, corresponding_segment_index_list, _) = \
            get_dtw_matrix_and_corresponding_segments(rough_features_source,
                                                      rough_features_target)

        corresponding_segment = get_corresponding_segments_function(corresponding_segment_index_list)

        source_features = voice_feature_extractor.extract(sound=source_sound)
        target_features = voice_feature_extractor.extract(sound=target_sound)

        pprint(source_features.keys())
        target_as_piecewise_linear_function = PiecewiseLinearFunction(params=params)


        for it_target in range(len(target_features["period_list"])):
            v = np.zeros([params["n_triangle_function"] * 2 + 1])
            v[0] = target_features["period_list"][it_target]
            v[1: 1 + params["n_triangle_function"]] = target_features["spectral_envelope_coeffs_harmonic_list"][it_target]
            v[1 + params["n_triangle_function"]:] = target_features["spectral_envelope_coeffs_noise_list"][it_target]

            target_as_piecewise_linear_function.add_point(time=it_target, value=v)

        aligned_target = {"period_list": [],
                          "spectral_envelope_coeffs_harmonic_list": [],
                          "spectral_envelope_coeffs_noise_list": []}

        for it_source in range(len(source_features["period_list"])):

            it_target = corresponding_segment(it_source)

            v = target_as_piecewise_linear_function.get_value(time=it_target)


            period = v[0]
            harmonic = v[1:1+params["n_triangle_function"]]
            noise = v[1+params["n_triangle_function"]:]


            aligned_target["period_list"]+= [period]
            aligned_target["spectral_envelope_coeffs_harmonic_list"] += [harmonic]
            aligned_target["spectral_envelope_coeffs_noise_list"] += [noise]


        target_sound_aligned = synthesize_voice(feature_list_dict=aligned_target, params=params, normalize=True)


        out_sound = np.zeros([source_sound.shape[0], 2])
        out_sound[:, 0] = source_sound / max(abs(source_sound))
        out_sound[:, 1] = target_sound_aligned

        filepath_array = source_filepath.split(sep="/")

        wavfile.write("/Users/pierresendorek/temp/parallel/" + filepath_array[-2] + "_" + filepath_array[-1], rate=44100, data=out_sound)












