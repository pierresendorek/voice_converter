from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from database_tools.parallel_recorder.batch_generator_rnn import BatchGeneratorRnn
from feature_extraction.feature_vector_array_to_feature_dict import feature_vector_array_to_feature_dict
from machine_learning.recurrent_neural_network import RecurrentNeuralNetwork

from params.params import get_params
from machine_learning.neural_network import NeuralNetwork
from pprint import pprint
import tensorflow as tf
import numpy as np
import os
from database_tools.get_all_files_in_tree import get_file_and_path_list
from synthesis.voice_synthesizer import synthesize_voice
from scipy.io.wavfile import write

if __name__ == "__main__":


    print_loss_smoothing_factor = 1E-2
    n_save_every = 1000
    batch_generator = BatchGeneratorRnn(new_protobatch_every=10000000, n_files_protobatch=17)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1E-2
    decay_rate = 0.1
    decay_steps = 10000

    learning_rate = tf.train.inverse_time_decay(starter_learning_rate,
                                                global_step=global_step,
                                                decay_steps=decay_steps,
                                                decay_rate=decay_rate,
                                                staircase=False)

    #learning_rate = tf.train.exponential_decay(starter_learning_rate,
    #                                           global_step=global_step,
    #                                           decay_steps=decay_steps,
    #                                           decay_rate=decay_rate,
    #                                           staircase=False)


    # printing params
    print_every = 10

    # rnn params
    rnn_output_transformed_dim = 300
    feature_extractor_dim = 300
    num_units_lstm = 40
    intermediate_dim = 200
    forget_bias = 0.95

    # batch params
    batch_size = 20
    seq_len = 100
    steps_ahead = 0
    listen_to_batch = False


    # loss param
    burn_in_time = 0

    params = get_params()
    parallel_corpus_maker = ParallelCorpusMaker(params)
    feature_len = params["n_triangle_function"] * 2 + 1


    # Initializing neural network and train_op accordingly
    nn = RecurrentNeuralNetwork(params=params,
                                rnn_output_transformed_dim=rnn_output_transformed_dim,
                                num_units=num_units_lstm,
                                burn_in_time=burn_in_time,
                                feature_extractor_dim=feature_extractor_dim,
                                intermediate_dim=intermediate_dim,
                                forget_bias=forget_bias)

    nn_dict = nn.get_neural_network()
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1E-4).minimize(nn.get_loss(),
                                                                                          global_step=global_step)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    smoothed_loss = 0
    smoothed_loss_variance = 0
    smoothing_compensator = 0

    gradient_descent_steps_count = 0

    saver = tf.train.Saver(max_to_keep=30, save_relative_paths=True)


    for iteration in range(1000000000):

        batch = batch_generator.draw_batch(batch_size=batch_size, seq_len=seq_len, steps_ahead=steps_ahead)
        source_batch = batch["source_batch"]
        target_batch = batch["target_batch"]

        # testing batch correctness
        if listen_to_batch:
            for i_batch in range(batch_size):
                feature_dict = feature_vector_array_to_feature_dict(target_batch[i_batch, :, :])
                sound = synthesize_voice(feature_list_dict=feature_dict, params=params, normalize=True)
                write("/Users/pierresendorek/temp/test/test_" + str(i_batch) + "_target.wav", 44100, sound)

                feature_dict = feature_vector_array_to_feature_dict(source_batch[i_batch, :, :])
                sound = synthesize_voice(feature_list_dict=feature_dict, params=params, normalize=True)
                write("/Users/pierresendorek/temp/test/test_" + str(i_batch) + "_source.wav", 44100, sound)

        _, loss = sess.run([train_op, nn.loss], feed_dict={
            nn_dict["input_placeholder"]: source_batch,
            nn_dict["expected_output_placeholder"]: target_batch})

        smoothed_loss = (1 - print_loss_smoothing_factor) * smoothed_loss + print_loss_smoothing_factor * loss
        smoothed_loss_variance = (1 - print_loss_smoothing_factor) * smoothed_loss_variance + print_loss_smoothing_factor * (loss - smoothed_loss)**2
        smoothing_compensator = (1 - print_loss_smoothing_factor) * smoothing_compensator + print_loss_smoothing_factor * 1

        if gradient_descent_steps_count % print_every == 0:
            print("smoothed loss ", smoothed_loss / smoothing_compensator,
                  "\t std dev loss ",  np.sqrt(smoothed_loss_variance) / smoothing_compensator,
                  "\t loss : ", loss,
                  "\t gradient_descent steps_count ", gradient_descent_steps_count)

        if gradient_descent_steps_count % n_save_every == 0:
            saver.save(sess, os.path.join(params["project_base_path"], "models/bruce_willis/model.ckpt"), global_step=gradient_descent_steps_count)

        gradient_descent_steps_count += 1








