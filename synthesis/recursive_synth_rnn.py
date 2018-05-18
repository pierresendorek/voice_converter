from copy import deepcopy

from common.get_element_from_list import get_element_from_list_constant_outside
from common.math.piecewise_linear_function import PiecewiseLinearFunction
import tensorflow as tf
import os
import numpy as np

from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
from feature_extraction.feature_vector_array_dict_conversion_helpers import feature_vector_array_to_feature_dict
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature
from feature_extraction.voice_feature_extractor import VoiceFeatureExtractor
from params.params import get_params
from common.math.relu import relu, np_relu
from synthesis.voice_synthesizer import generate_filtered_noise, generate_periodic_sound, \
    generate_periodic_filtered_sound, synthesize_voice

from scipy.io.wavfile import write
from pprint import pprint
import pickle

class RecursiveSynthRnn:

    def __init__(self, params=None):
        assert params is not None
        self.params = params
        #self.predictor = predictor
        tf.reset_default_graph()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.vector_normalizer_source = pickle.load(open(os.path.join(params["project_base_path"],
                                                                      "models/vector_normalizer_source.pickle"), "rb"))
        self.vector_normalizer_target = pickle.load(open(os.path.join(params["project_base_path"],
                                                                      "models/vector_normalizer_target.pickle"), "rb"))


        path = os.path.join(params["project_base_path"], "models/bruce_willis/")
        file_to_open = path + "model.ckpt-23000.meta"

        pprint(file_to_open)

        saver = tf.train.import_meta_graph(file_to_open)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

        graph = tf.get_default_graph()

        self.input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
        self.output_layer = graph.get_tensor_by_name("predicted_output:0")



    def synthesis_from_prediction(self, source_sound_features):

        n_time_steps_source = len(source_sound_features["period_list"])
        n_triangle_function = self.params["n_triangle_function"]

        i_target = 0.0
        print("n_time_steps_source ", n_time_steps_source)
        feature_source_array = np.zeros(
            [1, n_time_steps_source, n_triangle_function * 2 + 1])

        for i in range(n_time_steps_source):

            # gathering features
            feature_source_array[0, i, 0] = source_sound_features["period_list"][i]
            feature_source_array[0, i, 1: 1 + n_triangle_function] = source_sound_features["spectral_envelope_coeffs_harmonic_list"][i]
            feature_source_array[0, i, 1 + n_triangle_function:1 + 2 * n_triangle_function] = source_sound_features["spectral_envelope_coeffs_noise_list"][i]


        mu_source, sigma2_source = self.vector_normalizer_source.get_mu_sigma2()
        mu_target, sigma2_target = self.vector_normalizer_target.get_mu_sigma2()

        feature_source_array = (feature_source_array - mu_source) / (2 * np.sqrt(sigma2_source))

        predicted_vectors = self.sess.run(self.output_layer, feed_dict={self.input_placeholder: feature_source_array})


        predicted_vectors = predicted_vectors * 2 * np.sqrt(sigma2_target) + mu_target

        feature_dict = feature_vector_array_to_feature_dict(predicted_vectors[0, :, :])


        ####

        reconstruction = synthesize_voice(feature_list_dict=feature_dict,
                                          params=params,
                                          normalize=True)

        r_source_sound_features = feature_vector_array_to_feature_dict(feature_source_array[0,:,:])

        #original = synthesize_voice(feature_list_dict=r_source_sound_features,
        #                 params=params,
        #                 normalize=True)

        #write("/Users/pierresendorek/temp/is_the_phoque.wav", 44100, original)

        return reconstruction




if __name__ == "__main__":


    base_path = "/Users/pierresendorek/"


    params = get_params()
    recursive_synth = RecursiveSynthRnn(params=params)

    f = "/Users/pierresendorek/projets/voice_converter/data/pierre_sendorek/Studio/3.wav"
    sound, _ = get_mono_left_channel_sound_and_sampling_frequency(f)
    voice_feature_extractor = VoiceFeatureExtractor(params=params)
    features = voice_feature_extractor.extract(sound=sound)
    reconstruction = recursive_synth.synthesis_from_prediction(features)
    write(base_path + "temp/rnn_willis_reconstruction.wav", 44100, reconstruction)

    f = "/Users/pierresendorek/temp/pierre_to_convert.wav"
    sound, _ = get_mono_left_channel_sound_and_sampling_frequency(f)
    features = voice_feature_extractor.extract(sound=sound)
    reconstruction = recursive_synth.synthesis_from_prediction(features)
    write(base_path + "temp/rnn_pierre_to_convert.wav", 44100, reconstruction)



