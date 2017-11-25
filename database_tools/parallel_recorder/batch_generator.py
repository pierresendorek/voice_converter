from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature
from params.params import get_params
import numpy as np


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

        self.d_pair_protobatch = {"target_feature_array_list": [],
                                  "source_feature_array_list": []}


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


    def draw_batch(self, batch_size=None):
        assert batch_size is not None
        if self.n_batch_generated_since_last_protobatch_generation == 0:
            print("Drawing new protobatch...")
            self.draw_protobatch()
            print("Done.")


        n_feature = len(self.d_pair_protobatch["target_feature_array_list"])
        i_feature_list = np.random.choice(n_feature, batch_size, replace=False)

        source_batch = np.zeros([batch_size, self.feature_len])
        target_batch = np.zeros([batch_size, self.params["n_triangle_function"] * 2 + 2])

        # Creating batch
        for i_batch in range(batch_size):

            i_feature = i_feature_list[i_batch]

            target_vector = self.d_pair_protobatch["target_feature_array_list"][i_feature]
            feature_vector = np.reshape(self.d_pair_protobatch["source_feature_array_list"][i_feature], [-1])

            source_batch[i_batch, :] = np.log(feature_vector + 1)
            target_batch[i_batch, :] = target_vector

        self.n_batch_generated_since_last_protobatch_generation += 1
        self.n_batch_generated_since_last_protobatch_generation = self.n_batch_generated_since_last_protobatch_generation % self.new_protobatch_every

        return {"source_batch": source_batch, "target_batch": target_batch}


