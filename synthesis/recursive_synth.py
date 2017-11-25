from common.get_element_from_list import get_element_from_list_constant_outside
from common.math.piecewise_linear_function import PiecewiseLinearFunction
import tensorflow as tf
import os
import numpy as np

from database_tools.sound_file_loader import get_mono_left_channel_sound_and_sampling_frequency
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature
from feature_extraction.voice_feature_extractor import VoiceFeatureExtractor
from params.params import get_params
from common.math.relu import relu, np_relu
from synthesis.voice_synthesizer import generate_filtered_noise, generate_periodic_sound, \
    generate_periodic_filtered_sound

from scipy.io.wavfile import write


class RecursiveSynth:

    def __init__(self, params=None, predictor=None):
        assert params is not None
        self.params = params
        self.predictor = predictor
        tf.reset_default_graph()

        psf = PairSoundFeature(params=params)

        #self.input_placeholder = tf.get_variable("input_placeholder", shape=[1, psf.get_feature_length()])
        #self.output_layer = tf.get_variable("output_layer", shape=[1, params["n_triangle_function"] * 2 + 2])

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

#        saver = tf.train.Saver({"input_placeholder": self.input_placeholder, "output_layer": self.output_layer})


        path = os.path.join(params["project_base_path"], "models/bruce_willis/")
        file_to_open = path + "model.ckpt-770000.meta"

        saver = tf.train.import_meta_graph(file_to_open)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

        graph = tf.get_default_graph()

        self.input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
        self.output_layer = graph.get_tensor_by_name("output_layer:0")





    def synthesis_from_prediction(self, source_sound_features):

        n_time_steps_source = len(source_sound_features["period_list"])
        n_triangle_function = self.params["n_triangle_function"]
        fw_range_source = self.params["fw_range_source"]
        bw_range_source = self.params["bw_range_source"]
        bw_range_target = self.params["bw_range_target"]

        feature_source_array = np.zeros(
            [n_triangle_function * 2 + 1, fw_range_source + bw_range_source + bw_range_target])

        i_target = 0.0

        piecewise_linear_function = PiecewiseLinearFunction(params=params)
        piecewise_linear_function.add_point(time=-1, value=np.zeros([self.params["n_triangle_function"] * 2 + 1]))


        for i in range(n_time_steps_source):

            # gathering features
            for k in range(-bw_range_source, fw_range_source):
                feature_source_array[0, k + bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["period_list"])

                feature_source_array[1: 1 + n_triangle_function, k + bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["spectral_envelope_coeffs_harmonic_list"])

                feature_source_array[1 + n_triangle_function:1 + 2 * n_triangle_function, k + bw_range_source] = \
                    get_element_from_list_constant_outside(i + k, source_sound_features["spectral_envelope_coeffs_noise_list"])

            for k in range(-bw_range_target, 0):
                # We add sound features from the target speaker from previous time steps
                feature_source_array[:, - k - 1 + fw_range_source + bw_range_source] = \
                    piecewise_linear_function.get_value(i_target + k)

            # prediction
            predicted_vector = self.sess.run(self.output_layer, feed_dict={self.input_placeholder: np.log(1 + np_relu(np.reshape(feature_source_array, [1, -1])))})

            delta_t = predicted_vector[0, 0]
            i_target += 1 #relu(delta_t) + 1E-6

            piecewise_linear_function.add_point(time=i_target, value=predicted_vector[0, 1:])

        # generating the whole sequence
        time_target = int(np.floor(i_target))

        target_period_list = []
        target_spectral_envelope_coeffs_harmonic_list = []
        target_spectral_envelope_coeffs_noise_list = []

        for i in range(time_target):
            feature_vector = piecewise_linear_function.get_value(time=i)
            target_period_list.append(feature_vector[0])
            target_spectral_envelope_coeffs_harmonic_list.append(feature_vector[1:n_triangle_function+1])
            target_spectral_envelope_coeffs_noise_list.append(feature_vector[n_triangle_function+1:])

        ####

        out_sound_noise = generate_filtered_noise(target_spectral_envelope_coeffs_noise_list, params)
        out_sound_periodic = generate_periodic_sound(segment_period_expressed_in_sample_list=target_period_list,
                                                         params=params)
        out_sound_periodic_filtered = generate_periodic_filtered_sound(
            segment_period_expressed_in_sample_list=target_period_list,
            spectral_envelope_coeffs_list=target_spectral_envelope_coeffs_harmonic_list,
            params=params)

        reconstruction = out_sound_noise + out_sound_periodic_filtered
        reconstruction = reconstruction / max(abs(reconstruction))

        base_path = "/Users/pierresendorek/"

        write(base_path + "temp/willis_noise.wav", 44100, out_sound_noise / np.max(np.abs(out_sound_noise)))
        write(base_path + "temp/willis_periodic.wav", 44100, out_sound_periodic / np.max(np.abs(out_sound_periodic)))
        write(base_path + "temp/willis_out_periodic_filt.wav", 44100,
              out_sound_periodic_filtered / np.max(np.abs(out_sound_periodic_filtered)))
        write(base_path + "temp/willis_reconstruction.wav", 44100, reconstruction)


if __name__ == "__main__":

    params = get_params()
    recursive_synth = RecursiveSynth(params=params)

    sound, _ = get_mono_left_channel_sound_and_sampling_frequency("/Users/pierresendorek/projets/voice_converter/data/pierre_sendorek/Studio/3.wav")


    voice_feature_extractor = VoiceFeatureExtractor(params=params)
    features = voice_feature_extractor.extract(sound=sound)

    recursive_synth.synthesis_from_prediction(features)

