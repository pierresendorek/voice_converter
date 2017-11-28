import tensorflow as tf
import numpy as np
from params.params import get_params
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.core import dense


class RecurrentNeuralNetwork:

    def __init__(self,
                 params=None,
                 num_units=None,  # 20
                 rnn_output_transformed_dim=None,
                 burn_in_time=None,
                 feature_extractor_dim=None,
                 intermediate_dim=None,
                 forget_bias=None):

        assert params is not None
        assert num_units is not None
        assert rnn_output_transformed_dim is not None
        assert burn_in_time is not None
        assert feature_extractor_dim is not None
        assert intermediate_dim is not None
        assert forget_bias is not None

        batch_size = None # kept for readability
        seq_max_len = None # kept for readability

        self.params = params
        self.feature_vector_dim = params["n_triangle_function"] * 2 + 1

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_max_len, self.feature_vector_dim], name="input_placeholder")

        self.predicted_output = self.create_neural_network(rnn_output_transformed_dim=rnn_output_transformed_dim,
                                                           num_units=num_units,
                                                           input_transformed_dim=feature_extractor_dim,
                                                           intermediate_dim=intermediate_dim,
                                                           forget_bias=forget_bias)

        self.expected_output_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.feature_vector_dim], name="expected_output_placeholder")



        cost = lambda x: tf.reduce_sum(tf.square(x[:, burn_in_time:,:]))
        self.loss = tf.divide(cost(self.predicted_output - self.expected_output_placeholder),
                              cost(self.expected_output_placeholder), name="loss")


    def make_rnn_cell(self, num_units, forget_bias):
        return tf.contrib.rnn.LSTMCell(num_units=num_units,
                                       activation=tf.nn.tanh,
                                       forget_bias=forget_bias,
                                       )
                                       #kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
                                       #bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))

    def get_dense_layer(self, input_layer, output_dim, activation=tf.nn.tanh):
        return dense(inputs=input_layer,
                     kernel_initializer=xavier_initializer(uniform=False),
                     bias_initializer=xavier_initializer(uniform=False),
                     activation=activation,
                     units=output_dim)




    def create_neural_network_simple(self,
                              input_transformed_dim=None,
                              rnn_output_transformed_dim=None,
                              intermediate_dim=None,
                              num_units=None,
                              forget_bias=None):

        assert input_transformed_dim is not None
        assert rnn_output_transformed_dim is not None
        assert num_units is not None
        assert intermediate_dim is not None
        assert forget_bias is not None


        batch_size_var, max_time_var, feature_vector_dim_var = tf.unstack(tf.shape(self.input_placeholder))

        input_reshape = tf.reshape(self.input_placeholder, shape=[batch_size_var * max_time_var, self.feature_vector_dim])

        # transforming
        input_transformed_reshape = self.get_dense_layer(input_layer=input_reshape, output_dim=input_transformed_dim)

        # scaling the result out of the sigmoid
        output_reshape = self.get_dense_layer(input_layer=input_transformed_reshape, output_dim=self.feature_vector_dim, activation=None)

        predicted_output = tf.reshape(output_reshape, shape=[batch_size_var, max_time_var, self.feature_vector_dim], name="predicted_output")
        return predicted_output




    def create_neural_network(self,
                              input_transformed_dim=None,
                              rnn_output_transformed_dim=None,
                              intermediate_dim=None,
                              num_units=None,
                              forget_bias=None):

        assert input_transformed_dim is not None
        assert rnn_output_transformed_dim is not None
        assert num_units is not None
        assert intermediate_dim is not None
        assert forget_bias is not None

        c_0 = self.make_rnn_cell(num_units, forget_bias)
        #c_1 = self.make_rnn_cell(num_units)
        #c_2 = self.make_rnn_cell(num_units)
        #c = tf.contrib.rnn.MultiRNNCell([c_0, c_1])

        batch_size_var, max_time_var, feature_vector_dim_var = tf.unstack(tf.shape(self.input_placeholder))

        input_reshape = tf.reshape(self.input_placeholder, shape=[batch_size_var * max_time_var, self.feature_vector_dim])

        input_transformed_reshape = self.get_dense_layer(input_layer=input_reshape, output_dim=input_transformed_dim)

        # The right format for the dynamic_rnn
        input_transformed = tf.reshape(input_transformed_reshape, [batch_size_var, max_time_var, input_transformed_dim])

        rnn_output, _ = tf.nn.dynamic_rnn(cell=c_0,
                                          inputs=input_transformed,
                                          initial_state=None,
                                          time_major=False,
                                          dtype=tf.float32)

        rnn_output_reshape = tf.reshape(rnn_output, [batch_size_var * max_time_var, num_units])

        rnn_output_transformed_reshape = self.get_dense_layer(input_layer=rnn_output_reshape, output_dim=rnn_output_transformed_dim)
        rnn_output_transformed = tf.reshape(rnn_output_transformed_reshape, shape=[batch_size_var, max_time_var, rnn_output_transformed_dim])


        whole_state = tf.concat([rnn_output_transformed, input_transformed], axis=2)

        whole_state_dim = input_transformed_dim + rnn_output_transformed_dim
        whole_state_reshape = tf.reshape(whole_state, shape=[batch_size_var * max_time_var, whole_state_dim])


        # combination of the RNN and the input value

        combination = self.get_dense_layer(input_layer=whole_state, output_dim=intermediate_dim)

        # scaling the result
        output_reshape = self.get_dense_layer(input_layer=combination, output_dim=self.feature_vector_dim, activation=None)

        # reshaping the result
        predicted_output = tf.reshape(output_reshape, shape=[batch_size_var, max_time_var, self.feature_vector_dim], name="predicted_output")

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
                                                      num_units=40,
                                                      burn_in_time=40,
                                                      feature_extractor_dim=200,
                                                      intermediate_dim=100,
                                                      forget_bias=0.95)
