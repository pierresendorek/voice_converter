import tensorflow as tf
from params.params import get_params
from pprint import pprint

class NeuralNetwork:

    def __init__(self,
                 params=None,
                 intermediate_layers_num_output_list=None, input_vector_len=None, target_len=None):

        assert intermediate_layers_num_output_list is not None
        assert input_vector_len is not None
        assert params is not None

        self.params = params
        # one for the pitch and one for delta_t
        self.final_layer_num_outputs = 2 * self.params["n_triangle_function"] + 1 + 1

        self.input_placeholder = tf.placeholder(tf.float32, [None, input_vector_len])
        self.output_layer = self.create_neural_network(intermediate_layers_num_output_list=intermediate_layers_num_output_list)

        self.expected_output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.final_layer_num_outputs])

        # Creating loss
        self.loss = tf.reduce_mean(tf.abs(self.output_layer - self.expected_output_placeholder))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def create_neural_network(self, intermediate_layers_num_output_list=None):


        # initialization
        layer = self.input_placeholder
        # loop
        for i_layer, num_outputs in enumerate(intermediate_layers_num_output_list):
            print(i_layer)
            print(num_outputs)
            layer = tf.contrib.layers.fully_connected(inputs=layer, num_outputs=num_outputs)
            layer = tf.nn.relu(layer)

        layer = tf.contrib.layers.fully_connected(inputs=layer, num_outputs=self.final_layer_num_outputs)
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
    intermediate_layers_num_output_list = [160, 160]
    nn = NeuralNetwork(params=params,
                       intermediate_layers_num_output_list=intermediate_layers_num_output_list,
                       input_vector_len=500)
    nn_as_dict = nn.get_neural_network()

    pprint(nn_as_dict)



