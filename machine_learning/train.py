from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from database_tools.parallel_recorder.batch_generator import BatchGenerator

from params.params import get_params
from machine_learning.neural_network import NeuralNetwork
from pprint import pprint
import tensorflow as tf
import numpy as np
import os
from database_tools.get_all_files_in_tree import get_file_and_path_list


if __name__ == "__main__":

    batch_size = 30
    print_loss_smoothing_factor = 1E-3
    n_save_every = 10000
    batch_generator = BatchGenerator(new_protobatch_every=10000000, n_files_protobatch=19)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1E-2
    decay_rate = 0.1
    decay_steps = 1000000
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=decay_rate,
                                               staircase=False)

    params = get_params()

    parallel_corpus_maker = ParallelCorpusMaker(params)
    feature_len = parallel_corpus_maker.get_feature_vector_len()
    #parallel_file_path_dict = parallel_corpus_maker.get_full_path_lists()


    # Initializing neural network and train_op accordingly
    nn = NeuralNetwork(params=params,
                       intermediate_layers_num_output_list=[1000, 500, 500],
                       input_vector_len=feature_len,
                       w_mult_factor=1.0,
                       b_mult_factor=1.0)

    nn_dict = nn.get_neural_network()
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1E-4).minimize(nn.get_loss(),
                                                                                          global_step=global_step)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # batch : creation and filling values

    smoothed_loss = 0
    smoothing_compensator = 0

    gradient_descent_steps_count = 0

    saver = tf.train.Saver(max_to_keep=30, save_relative_paths=True)

    for iteration in range(100000000):

        batch = batch_generator.draw_batch(batch_size)
        source_batch = np.log(batch["source_batch"] + 1)
        target_batch = batch["target_batch"]

        _, loss = sess.run([train_op, nn.loss], feed_dict={
            nn_dict["input_placeholder"]: source_batch,
            nn_dict["expected_output_placeholder"]: target_batch})

        smoothed_loss = (1 - print_loss_smoothing_factor) * smoothed_loss + print_loss_smoothing_factor * loss
        smoothing_compensator = (1 - print_loss_smoothing_factor) * smoothing_compensator + print_loss_smoothing_factor * 1

        if gradient_descent_steps_count % 100 == 0:
            print("smoothed loss ", smoothed_loss / smoothing_compensator, "\t loss : ", loss, "\t gradient_descent_steps_count ", gradient_descent_steps_count)

        if gradient_descent_steps_count % n_save_every == 0:
            saver.save(sess, os.path.join(params["project_base_path"], "models/bruce_willis/model.ckpt"), global_step=gradient_descent_steps_count)

        gradient_descent_steps_count += 1








