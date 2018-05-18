from copy import deepcopy

from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature
from params.params import get_params
import numpy as np

from synthesis.voice_synthesizer import synthesize_voice
from scipy.io.wavfile import write

class BatchGenerator:


    def __init__(self, new_protobatch_every=None, n_files_protobatch=None):
        assert new_protobatch_every is not None
        assert n_files_protobatch is not None

        self.params = get_params()

        self.psf = PairSoundFeature(params=self.params)
        parallel_corpus_maker = ParallelCorpusMaker(self.params)
        self.feature_len = parallel_corpus_maker.get_feature_vector_len()
        self.parallel_file_path_dict = parallel_corpus_maker.get_full_path_lists()
        self.new_protobatch_every = new_protobatch_every
        self.n_batch_generated_since_last_protobatch_generation = 0
        self.n_files_in_protobatch = n_files_protobatch


    def draw_protobatch(self):

        num_files = len(self.parallel_file_path_dict["pierre_sendorek_full_path_list"])
        i_file_list_protobatch = np.random.choice(range(num_files), self.n_files_in_protobatch, replace=False)

        self.d_pair_protobatch = {"target_feature_array_list": [],
                             "source_feature_array_list": []}

        for i_file in i_file_list_protobatch:
            pierre_sendorek_wav_file = self.parallel_file_path_dict["pierre_sendorek_full_path_list"][i_file]
            bruce_willis_wav_file = self.parallel_file_path_dict["bruce_willis_full_path_list"][i_file]
            d_pair = self.psf.get_sound_pair_features(pierre_sendorek_wav_file, bruce_willis_wav_file)

            self.d_pair_protobatch["target_feature_array_list"] += d_pair["target_feature_array_list"]
            self.d_pair_protobatch["source_feature_array_list"] += d_pair["source_feature_array_list"]

        protobatch_size = len(self.d_pair_protobatch["source_feature_array_list"])

        self.target_feature_protobatch = np.zeros([protobatch_size, self.params["n_triangle_function"] * 2 + 2])
        self.source_feature_protobatch = np.zeros([protobatch_size, self.feature_len])

        for i_sample in range(protobatch_size):
            self.target_feature_protobatch[i_sample, :] = self.d_pair_protobatch["target_feature_array_list"][i_sample]
            self.source_feature_protobatch[i_sample, :] = np.reshape(self.d_pair_protobatch["source_feature_array_list"][i_sample], [-1])


    def draw_batch(self, batch_size=None):
        assert batch_size is not None
        if self.n_batch_generated_since_last_protobatch_generation == 0:
            print("Drawing new protobatch...")
            self.draw_protobatch()
            print("Done.")

        n_feature = self.target_feature_protobatch.shape[0]
        i_sample_list = np.random.choice(n_feature, batch_size, replace=False)


        source_batch = self.source_feature_protobatch[i_sample_list, :]
        target_batch = self.target_feature_protobatch[i_sample_list, :]

        self.n_batch_generated_since_last_protobatch_generation += 1
        self.n_batch_generated_since_last_protobatch_generation = self.n_batch_generated_since_last_protobatch_generation % self.new_protobatch_every

        return {"source_batch": source_batch,
                "target_batch": target_batch}




if __name__ == "__main__":

    params = get_params()
    batch_generator = BatchGenerator(new_protobatch_every=10, n_files_protobatch=17)
    batch_generator.draw_protobatch()

    protobatch_len = batch_generator.target_feature_protobatch.shape[0]

    target_period_list = []
    target_spectral_envelope_coeffs_harmonic_list = []
    target_spectral_envelope_coeffs_noise_list = []

    source_period_list = []
    source_spectral_envelope_coeffs_harmonic_list = []
    source_spectral_envelope_coeffs_noise_list = []

    for it in range(protobatch_len):
        a_source = batch_generator.d_pair_protobatch["source_feature_array_list"][it]
        v_source = a_source[:, params["bw_range_source"]]

        source_period_list.append(deepcopy(v_source[0]))
        source_spectral_envelope_coeffs_harmonic_list.append(deepcopy(v_source[1:1 + params["n_triangle_function"]]))
        source_spectral_envelope_coeffs_noise_list.append(deepcopy(v_source[1 + params["n_triangle_function"]:]))

        v_target = batch_generator.target_feature_protobatch[it]
        target_period_list.append(deepcopy(v_target[1]))
        target_spectral_envelope_coeffs_harmonic_list.append(deepcopy(v_target[2:2 + params["n_triangle_function"]]))
        target_spectral_envelope_coeffs_noise_list.append(deepcopy(v_target[2 + params["n_triangle_function"]:]))


    source_feature_dict = {"period_list": source_period_list,
                           "spectral_envelope_coeffs_harmonic_list": source_spectral_envelope_coeffs_harmonic_list,
                           "spectral_envelope_coeffs_noise_list": source_spectral_envelope_coeffs_noise_list}

    target_feature_dict = {"period_list": target_period_list,
                           "spectral_envelope_coeffs_harmonic_list": target_spectral_envelope_coeffs_harmonic_list,
                           "spectral_envelope_coeffs_noise_list": target_spectral_envelope_coeffs_noise_list}


    source_sound = synthesize_voice(feature_list_dict=source_feature_dict, params=params, normalize=True)
    target_sound = synthesize_voice(feature_list_dict=target_feature_dict, params=params, normalize=True)

    base_path = "/Users/pierresendorek/"
    write(base_path + "temp/source_sound.wav", 44100, source_sound)
    write(base_path + "temp/target_sound.wav", 44100, target_sound)













