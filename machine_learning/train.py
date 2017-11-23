from database_tools.parallel_corpus_maker import ParallelCorpusMaker

from params.params import get_params
from machine_learning.neural_network import NeuralNetwork
from pprint import pprint
import tensorflow as tf
import numpy as np
import os
from database_tools.get_all_files_in_tree import get_file_and_path_list
from feature_extraction.pair_sequence_feature_target_getter import PairSoundFeature


if __name__ == "__main__":

    batch_size = 11
    n_repeat_one_file = 100
    print_loss_smoothing_factor = 1E-3
    n_save_every = 10000
    learning_rate = 1E-4

    params = get_params()

    parallel_corpus_maker = ParallelCorpusMaker(params)
    feature_len = parallel_corpus_maker.get_feature_vector_len()
    parallel_file_path_dict = parallel_corpus_maker.get_full_path_lists()


    # Initializing neural network and train_op accordingly
    nn = NeuralNetwork(params=params,
                       intermediate_layers_num_output_list=[1000, 500, 500],
                       input_vector_len=feature_len,
                       w_mult_factor=1.0,
                       b_mult_factor=1.0)

    nn_dict = nn.get_neural_network()
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1E-4).minimize(nn.get_loss())
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # batch : creation and filling values
    source_batch = np.zeros([batch_size, feature_len])
    target_batch = np.zeros([batch_size, params["n_triangle_function"] * 2 + 2])

    smoothed_loss = 0
    smoothed_loss_compensator = 0

    psf = PairSoundFeature(params=params)

    gradient_descent_steps_count = 0

    saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)

    for iteration in range(10000000):

        i_file = np.random.choice(range(len(parallel_file_path_dict["pierre_sendorek_full_path_list"])), 1)[0]
        pierre_sendorek_wav_file = parallel_file_path_dict["pierre_sendorek_full_path_list"][i_file]
        bruce_willis_wav_file = parallel_file_path_dict["bruce_willis_full_path_list"][i_file]
        d_pair = psf.get_sound_pair_features(pierre_sendorek_wav_file, bruce_willis_wav_file)

        for _ in range(n_repeat_one_file):

            i_feature_list = np.random.choice(len(d_pair["target_feature_array_list"]), batch_size, replace=False)

            # Creating batch
            for i_batch in range(batch_size):
                i_feature = i_feature_list[i_batch]

                target_vector = d_pair["target_feature_array_list"][i_feature]
                feature_vector = np.reshape(d_pair["source_feature_array_list"][i_feature], [-1])

                source_batch[i_batch, :] = np.log(feature_vector + 1)
                target_batch[i_batch, :] = target_vector

            _, loss = sess.run([train_op, nn.loss],
                            feed_dict={nn_dict["input_placeholder"]: source_batch,
                                       nn_dict["expected_output_placeholder"]: target_batch})

            smoothed_loss = (1 - print_loss_smoothing_factor) * smoothed_loss + print_loss_smoothing_factor * loss
            smoothed_loss_compensator = (1 - print_loss_smoothing_factor) * smoothed_loss_compensator + print_loss_smoothing_factor * 1

            if gradient_descent_steps_count % 100 == 0:
                print("smoothed loss ", smoothed_loss/smoothed_loss_compensator, "\t loss : ", loss, "\t gradient_descent_steps_count ", gradient_descent_steps_count)

            if gradient_descent_steps_count % n_save_every == 0:
                saver.save(sess, os.path.join(params["project_base_path"], "models/bruce_willis/model.ckpt"), global_step=gradient_descent_steps_count)

            gradient_descent_steps_count += 1








