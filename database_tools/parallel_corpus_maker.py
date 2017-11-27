from database_tools.get_all_files_in_tree import get_file_and_path_list
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature

import os
import numpy as np

class ParallelCorpusMaker:


    def _assemble_name(self, el):
        return os.path.join(el[0], el[1])


    def __init__(self, params):
        self.params = params

        # takes filenames
        bw_filme = get_file_and_path_list(
            os.path.join(self.params["project_base_path"], "data/bruce_willis/FILME"))

        bw_studio = get_file_and_path_list(
            os.path.join(self.params["project_base_path"], "data/bruce_willis/Studio"))


        self._bruce_willis_file_and_path_list = bw_studio #+

        # assembles the whole corresponding file name for Pierre
        ps_filme = [
            (os.path.join(self.params["project_base_path"], "data/pierre_sendorek/FILME"),
             filename[1]) for filename in bw_filme]

        ps_studio = [
            (os.path.join(self.params["project_base_path"], "data/pierre_sendorek/Studio"),
             filename[1]) for filename in bw_studio]

        self._pierre_sendorek_file_and_path_list = ps_studio #+


        self.pierre_sendorek_full_path_list = [self._assemble_name(path_and_file) for path_and_file in self._pierre_sendorek_file_and_path_list]
        self.bruce_willis_full_path_list = [self._assemble_name(path_and_file) for path_and_file in self._bruce_willis_file_and_path_list]


    def get_feature_vector_len(self):

        bruce_willis_wav_file = self._assemble_name(self._bruce_willis_file_and_path_list[0])
        pierre_sendorek_wav_file = self._assemble_name(self._pierre_sendorek_file_and_path_list[0])

        psf = PairSoundFeature(params=self.params)
        d = psf.get_sound_pair_features(source_path=pierre_sendorek_wav_file, target_path=bruce_willis_wav_file)

        features_list = d["source_feature_array_list"]
        features_array = features_list[0]
        features_vector = np.reshape(features_array, [-1])
        feature_len = features_vector.shape[0]

        return feature_len

    def get_full_path_lists(self):
        return {"pierre_sendorek_full_path_list": self.pierre_sendorek_full_path_list,
                "bruce_willis_full_path_list": self.bruce_willis_full_path_list}



