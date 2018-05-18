from database_tools.parallel_corpus_maker import ParallelCorpusMaker
from database_tools.parallel_recorder.batch_generator_rnn import BatchGeneratorRnn
from database_tools.vector_set_normalization_params import VectorSetNormalizationParams
from feature_extraction.feature_vector_array_dict_conversion_helpers import feature_vector_array_to_feature_dict
from machine_learning.recurrent_neural_network import RecurrentNeuralNetwork
import pickle
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

    config = tf.ConfigProto() # for GPUs
    config.allow_soft_placement = True


    sess = tf.InteractiveSession(config=config)

    n_save_every = 1000
    batch_generator = BatchGeneratorRnn(new_protobatch_every=10000000, n_files_protobatch=17)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1E-2
    decay_steps = 10000

    learning_rate = tf.train.inverse_time_decay(starter_learning_rate,
                                                global_step=global_step,
                                                decay_steps=decay_steps,
                                                decay_rate=1.0,
                                                staircase=False)

    #learning_rate = tf.train.exponential_decay(starter_learning_rate,
    #                                           global_step=global_step,
    #                                           decay_steps=decay_steps,
    #                                           decay_rate=decay_rate,
    #                                           staircase=False)


    # printing params
    print_every = 10
    print_loss_smoothing_factor = 1E-3 # parameter for moving average of the loss

    # rnn params
    num_units_lstm_list = [120] * 3
    forget_bias = tf.Variable(initial_value=0.5, trainable=True)

    # loss param
    burn_in_time = 0

    # batch params
    batch_size = 5
    seq_len = 100
    listen_to_batch = False
    steps_ahead = 0 # negative delay of source expressed in frames

    params = get_params()
    parallel_corpus_maker = ParallelCorpusMaker(params)
    feature_len = params["n_triangle_function"] * 2 + 1

    # Initializing neural network and train_op accordingly
    nn = RecurrentNeuralNetwork(params=params,
                                num_units_list=num_units_lstm_list,
                                burn_in_time=burn_in_time,
                                forget_bias=forget_bias)

    nn_dict = nn.get_neural_network()

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1E-4).minimize(nn.get_loss(),
                                                                                          global_step=global_step)


    sess.run(tf.global_variables_initializer())

    smoothed_loss = 0
    smoothed_loss_variance = 0
    smoothing_compensator = 0

    gradient_descent_steps_count = 0

    saver = tf.train.Saver(max_to_keep=1000, save_relative_paths=True)

    # initialize normalizers

    print("Normalization parameters computation...")

    vn_source = None
    vn_target = None

    for it in range(100):
        batch = batch_generator.draw_batch(batch_size=batch_size, seq_len=seq_len, steps_ahead=steps_ahead)
        if it == 0:
            dim = len(batch["source_batch"][0, 0, :])
            print(dim, "is dim source")
            vn_source = VectorSetNormalizationParams(dim)
            dim = len(batch["target_batch"][0, 0, :])
            print(dim, "is dim target")
            vn_target = VectorSetNormalizationParams(dim)

        for i_batch in range(batch_size):
            for i_time in range(seq_len):
                vn_source.use_also_this_vector_for_estimation(batch["source_batch"][i_batch, i_time, :])
                vn_target.use_also_this_vector_for_estimation(batch["target_batch"][i_batch, i_time, :])


    print("Normalization parameters computation : Done.")
    mu_source, sigma2_source = vn_source.get_mu_sigma2()
    mu_target, sigma2_target = vn_target.get_mu_sigma2()

    pickle.dump(vn_source,
                open(os.path.join(params["project_base_path"], "models/vector_normalizer_source.pickle"), "wb"))
    pickle.dump(vn_target,
                open(os.path.join(params["project_base_path"], "models/vector_normalizer_target.pickle"), "wb"))

    for iteration in range(1000000000):

        batch = batch_generator.draw_batch(batch_size=batch_size, seq_len=seq_len, steps_ahead=steps_ahead)
        source_batch = batch["source_batch"] + np.random.randn(*(batch["source_batch"].shape)) / 20
        target_batch = batch["target_batch"]

        source_batch = (source_batch - mu_source) / (2 * np.sqrt(sigma2_source))
        target_batch = (target_batch - mu_target) / (2 * np.sqrt(sigma2_target))

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
                  "\t std dev loss ",  np.sqrt(smoothed_loss_variance / smoothing_compensator) ,
                  "\t loss : ", loss,
                  "\t gradient_descent steps_count ", gradient_descent_steps_count)

        if gradient_descent_steps_count % n_save_every == 0:
            saver.save(sess, os.path.join(params["project_base_path"], "models/bruce_willis/model.ckpt"), global_step=gradient_descent_steps_count)

        gradient_descent_steps_count += 1








