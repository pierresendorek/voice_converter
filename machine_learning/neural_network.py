import tensorflow as tf
from params.params import get_params
from pprint import pprint
import numpy as np

class NeuralNetwork:

    def __init__(self,
                 params=None,
                 intermediate_layers_num_output_list=None,
                 input_vector_len=None,
                 w_mult_factor=None,
                 b_mult_factor=None):

        assert intermediate_layers_num_output_list is not None
        assert input_vector_len is not None
        assert params is not None
        assert w_mult_factor is not None

        self.w_mult_factor = w_mult_factor
        self.b_mult_factor = b_mult_factor

        self.params = params
        # one for the pitch and one for delta_t
        self.final_layer_num_outputs = 2 * self.params["n_triangle_function"] + 1 + 1
        self.input_vector_len = input_vector_len

        self.input_placeholder = tf.placeholder(tf.float32, [None, input_vector_len], name="input_placeholder")
        self.output_layer = self.create_neural_network(intermediate_layers_num_output_list=intermediate_layers_num_output_list)

        self.expected_output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.final_layer_num_outputs], name="expected_output_placeholder")

        # Creating loss
        self.loss = tf.reduce_sum(tf.abs(self.output_layer - self.expected_output_placeholder))/tf.reduce_sum(tf.abs(self.expected_output_placeholder))


    def create_neural_network(self, intermediate_layers_num_output_list=None):

        # initialization
        layer = self.input_placeholder

        # loop
        num_inputs = self.input_vector_len

        for i_layer, num_outputs in enumerate(intermediate_layers_num_output_list):
            if i_layer == 0:
                w_mult_factor = self.w_mult_factor
                b_mult_factor = self.b_mult_factor
            else:
                w_mult_factor = 1
                b_mult_factor = 1

            W = tf.Variable(initial_value=np.random.randn(num_inputs, num_outputs) * w_mult_factor, dtype=tf.float32)
            b = tf.Variable(initial_value=np.random.randn(num_outputs) * b_mult_factor, dtype=tf.float32)
            layer = tf.add(tf.matmul(layer, W), b)
            layer = tf.nn.elu(layer)
            num_inputs = num_outputs

            #layer = tf.contrib.layers.fully_connected(inputs=layer,
            #                                          num_outputs=num_outputs,
            #                                          activation_fn=tf.nn.relu,
            #                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
            #                                          biases_initializer=tf.contrib.layers.xavier_initializer())

        num_inputs = num_outputs
        num_outputs = self.final_layer_num_outputs
        W = tf.Variable(initial_value=np.random.randn(num_inputs, num_outputs), dtype=tf.float32)
        b = tf.Variable(initial_value=np.random.randn(num_outputs), dtype=tf.float32)
        layer = tf.add(tf.matmul(layer, W), b, name="output_layer")

        return layer


    def get_neural_network(self):
        return {"input_placeholder": self.input_placeholder,
                "output_layer": self.output_layer,
                "expected_output_placeholder": self.expected_output_placeholder,
                "loss": self.loss}


    def get_loss(self):
        return self.loss


if __name__ == "__main__":

    params = get_params()
    intermediate_layers_num_output_list = [160, 30]
    nn = NeuralNetwork(params=params,
                       intermediate_layers_num_output_list=intermediate_layers_num_output_list,
                       input_vector_len=500,
                       w_mult_factor=10,
                       b_mult_factor=10)
    nn_as_dict = nn.get_neural_network()
    pprint(nn_as_dict)



