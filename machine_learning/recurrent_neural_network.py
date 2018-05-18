import tensorflow as tf
import numpy as np
from params.params import get_params
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.core import dense


class RecurrentNeuralNetwork:

    def __init__(self,
                 params=None,
                 num_units_list=None,  # 20
                 burn_in_time=None,
                 forget_bias=None):

        assert params is not None
        assert num_units_list is not None
        assert burn_in_time is not None
        assert forget_bias is not None

        batch_size = None # kept for readability
        seq_max_len = None # kept for readability

        self.params = params
        self.feature_vector_dim = params["n_triangle_function"] * 2 + 1

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_max_len, self.feature_vector_dim], name="input_placeholder")

        self.predicted_output = self.create_neural_network(num_units_list=num_units_list,
                                                           forget_bias=forget_bias)


        self.expected_output_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.feature_vector_dim], name="expected_output_placeholder")



        cost = lambda x: tf.reduce_sum(tf.square(x[:, burn_in_time:,:]))

        self.loss = tf.divide(cost(self.predicted_output - self.expected_output_placeholder),
                              cost(self.expected_output_placeholder), name="loss")


    def make_rnn_cell(self, num_units, forget_bias):
        if forget_bias is None:
            forget_bias = 1.0
            print("Info : forget_bias is not used in make_rnn_cell")

        return tf.contrib.rnn.LSTMCell(num_units=num_units,
                                       activation=tf.nn.tanh,
                                       forget_bias=forget_bias)
                                       #bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))


    def affine_and_transform(self, input_layer, input_dim, output_dim, activation=tf.nn.relu, name=None):
        return tf.layers.dense(inputs=input_layer,
                               units=output_dim,
                               activation=activation,
                               kernel_initializer=xavier_initializer(uniform=False),
                               name=name)

    def affine_and_transform_last_dim(self, input_layer, input_dim, output_dim, activation=tf.nn.tanh, name=None, mult_w=1.0):
        input_layer_concat = tf.concat(input_layer, axis=2)
        batch_size, seq_len, dim = tf.unstack(tf.shape(input_layer_concat))
        data_reshape = tf.reshape(input_layer_concat, shape=[batch_size * seq_len, input_dim])
        data_transform_reshape = self.affine_and_transform(data_reshape, input_dim=input_dim, output_dim=output_dim, activation=activation)
        data_transform = tf.reshape(data_transform_reshape, shape=[batch_size, seq_len, output_dim], name=name)
        return data_transform



    def create_neural_network(self,
                              num_units_list=None,
                              forget_bias=None):

        assert num_units_list is not None
        assert forget_bias is not None


        cell_list = [self.make_rnn_cell(num_units, forget_bias) for num_units in num_units_list]

        c_fwd = tf.contrib.rnn.MultiRNNCell(cell_list)

        cell_list_bkwd = [self.make_rnn_cell(num_units, forget_bias) for num_units in num_units_list]
        c_bkwd = tf.contrib.rnn.MultiRNNCell(cell_list_bkwd)

        rnn_output, _ = tf.nn.dynamic_rnn(cell=c_fwd,
                                          inputs=self.input_placeholder,
                                          initial_state=None,
                                          time_major=False,
                                          dtype=tf.float32)

        rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=c_bkwd,
                                                      cell_fw=c_fwd,
                                                      inputs=self.input_placeholder,
                                                      initial_state_bw=None,
                                                      initial_state_fw=None,
                                                      time_major=False,
                                                      dtype=tf.float32)

        #rnn_output_dropout= tf.nn.dropout(
        #                            rnn_output,
        #                            0.5,
        #                            noise_shape=None,
        #                           seed=None,
        #                            name=None)

        # Scaling the result
        predicted_output = self.affine_and_transform_last_dim(input_layer=rnn_output,
                                                              input_dim=num_units_list[-1] * 2,
                                                              output_dim=self.feature_vector_dim,
                                                              activation=None,
                                                              name="predicted_output")

        return predicted_output


    def get_loss(self):
        return self.loss

    def get_neural_network(self):
        return {"input_placeholder": self.input_placeholder,
                "predicted_output": self.predicted_output,
                "expected_output_placeholder": self.expected_output_placeholder,
                "loss": self.loss}




if __name__ == "__main__":
    params = get_params()

    recurrent_neural_network = RecurrentNeuralNetwork(params=params,
                                                      rnn_output_transformed_dim=160,
                                                      num_units_list=[11, 31, 81],
                                                      burn_in_time=40,
                                                      feature_extractor_dim=200,
                                                      intermediate_dim=100,
                                                      forget_bias=0.95)

    print("recurrent_neural_network ", recurrent_neural_network)