from common.math.piecewise_linear_function import PiecewiseLinearFunction
import tensorflow as tf
import os

from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature
from params.params import get_params

class RecursiveSynth:

    def __init__(self, params=None, predictor=None):
        assert params is not None
        self.piecewise_linear_function = PiecewiseLinearFunction(params=params)
        self.predictor = predictor
        tf.reset_default_graph()

        psf = PairSoundFeature(params=params)

        #self.input_placeholder = tf.get_variable("input_placeholder", shape=[1, psf.get_feature_length()])
        #self.output_layer = tf.get_variable("output_layer", shape=[1, params["n_triangle_function"] * 2 + 2])

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

#        saver = tf.train.Saver({"input_placeholder": self.input_placeholder, "output_layer": self.output_layer})



        path = os.path.join(params["project_base_path"], "models/bruce_willis/")
        file_to_open = path + "model.ckpt-20000.meta"

        saver = tf.train.import_meta_graph(file_to_open)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

        graph = tf.get_default_graph()

        self.input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
        self.output_layer = graph.get_tensor_by_name("output_layer:0")



    def synthesis_from_prediction(self, source_features_as_dict):

        n_time_steps_source = 100
        for i in range(n_time_steps_source):
            pass
            # get_features_from_source
            # get_past_features_from_target
            # concatene
            # feed into predictor
            # append to source features





if __name__ == "__main__":

    params = get_params()
    recursive_synth = RecursiveSynth(params=params)
