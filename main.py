# -*- coding: utf-8 -*-
"""
    Optimizing VNF Placement Using DRL and RCPO

    Author: Ramy Mohamed, PhD candidate at Carleton University
    Date: July 2024
"""
import csv
import logging
import os

from agent import *
from config import *
from environment import *
from service_batch_generator import *

""" Globals """
DEBUG = True


def print_trainable_parameters():
    """ Calculate the number of weights """

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print('shape: ', shape, 'variable_parameters: ', variable_parameters)
        total_parameters += variable_parameters
    print('Total_parameters: ', total_parameters)


def calculate_reward(env, networkServices, placement, num_samples):
    """ Evaluate the batch of states into the environmnet """

    lagrangian = np.zeros(config.batch_size)
    penalty = np.zeros(config.batch_size)
    reward = np.zeros(config.batch_size)
    constraint_occupancy = np.zeros(config.batch_size)
    constraint_bandwidth = np.zeros(config.batch_size)
    constraint_latency = np.zeros(config.batch_size)

    reward_sampling = np.zeros(num_samples)
    constraint_occupancy_sampling = np.zeros(num_samples)
    constraint_bandwidth_sampling = np.zeros(num_samples)
    constraint_latency_sampling = np.zeros(num_samples)

    indices = np.zeros(config.batch_size)

    # Compute environment
    for batch in range(config.batch_size):
        for sample in range(num_samples):
            env.clear()
            env.step(networkServices.service_length[batch], networkServices.state[batch], placement[sample][batch])
            reward_sampling[sample] = env.reward
            constraint_occupancy_sampling[sample] = env.constraint_occupancy
            constraint_bandwidth_sampling[sample] = env.constraint_bandwidth
            constraint_latency_sampling[sample] = env.constraint_latency

        # Variable Lambdas
        penalty_sampling = agent.lambda_occupancy * constraint_occupancy_sampling + \
                           agent.lambda_bandwidth * constraint_bandwidth_sampling + \
                           agent.lambda_latency * constraint_latency_sampling

        lagrangian_sampling = reward_sampling + penalty_sampling

        index = np.argmin(lagrangian_sampling)

        lagrangian[batch] = lagrangian_sampling[index]
        penalty[batch] = penalty_sampling[index]
        reward[batch] = reward_sampling[index]

        constraint_occupancy[batch] = constraint_occupancy_sampling[index]
        constraint_bandwidth[batch] = constraint_bandwidth_sampling[index]
        constraint_latency[batch] = constraint_latency_sampling[index]

        indices[batch] = index

    return lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices


if __name__ == "__main__":

    """ Log """
    logging.basicConfig(level=logging.DEBUG)  # filename='example.log'
    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd, config.env_profile)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    networkServices = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, vocab_size)

    """ Agent """
    agent = Agent(config)

    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    print("Starting session ...")

    with tf.Session() as sess:

        # Activate Tensorflow CLI debugger
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Activate Tensorflow debugger in Tensorboard
        # sess = tf_debug.TensorBoardDebugWrapperSession(
        #    sess=sess,
        #    grpc_debug_server_addresses=['localhost:6064'],
        #    send_traceback_and_source_code=True)

        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Print total number of parameters
        print_trainable_parameters()

        # Learn model
        if config.learn_mode:
            """
                Learning
            """

            # Restore model
            if config.load_model:
                saver.restore(sess, "{}/tf_placement.ckpt".format(config.load_from))
                print("\nModel restored.")

            # Summary writer
            writer = tf.summary.FileWriter("summary/repo/drl-rcpo", sess.graph)

            if config.save_model:
                filePath = "{}/learning_history.csv".format(config.save_to)

                if not os.path.exists(os.path.dirname(filePath)):
                    os.makedirs(os.path.dirname(filePath))

                if os.path.exists(filePath) and not config.load_model:
                    os.remove(filePath)

            print("\nStart learning...")

            try:
                episode = 0
                for episode in range(config.num_epoch):

                    # New batch of states (Get New Batch of Service Requests)
                    networkServices.getNewState()

                    # Mask
                    # What is the function of mask?
                    # This technique is often used in sequence-to-sequence models where input sequences have varying
                    # lengths, and the mask is used to ensure that the model does not attend to or generate output
                    # for positions beyond the actual sequence length.
                    mask = np.zeros((config.batch_size, config.max_length))
                    for i in range(config.batch_size):
                        for j in range(networkServices.service_length[i], config.max_length):
                            mask[i, j] = 1

                    # RL Learning
                    # defines a feed dictionary that is used to provide the input data, sequence lengths,
                    # and binary mask to a TensorFlow graph representing a reinforcement learning agent during training.
                    feed = {agent.input_: networkServices.state,
                            agent.input_len_: [item for item in networkServices.service_length],
                            agent.mask: mask}

                    # Run placement
                    # run a forward pass of a neural network that contains an actor-critic model,
                    # which is used to predict a placement for a sequence of items
                    # The four output tensors are used in subsequent steps of the placement algorithm to evaluate
                    # the quality of the predicted placement and update the weights of the neural network.
                    placement, decoder_softmax, _, baseline = sess.run(
                        [agent.actor.decoder_exploration, agent.actor.decoder_softmax, agent.actor.attention_plot,
                         agent.valueEstimator.value_estimate], feed_dict=feed)
                    # positions, attention_plot = sess.run([agent.actor.positions, agent.actor.attention_plot], feed_dict=feed)

                    # Interact with the environment to return reward
                    num_samples = 1

                    lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices = calculate_reward(
                        env, networkServices, placement, num_samples)

                    # placement_ contains the predicted placement for each item in the current batch
                    placement_ = np.zeros((config.batch_size, config.max_length))
                    for batch in range(config.batch_size):
                        placement_[batch] = placement[int(indices[batch])][batch]

                    # uses the predicted placement, input data, and related information to feed data to the neural
                    # network during training, and update the weights of the neural network to improve the quality
                    # of the predicted placements.
                    feed = {agent.placement_holder: placement_,
                            agent.input_: networkServices.state,
                            agent.input_len_: [item for item in networkServices.service_length],
                            agent.mask: mask,
                            agent.baseline_holder: baseline,
                            agent.lagrangian_holder: [item for item in lagrangian],
                            agent.constraint_occupancy_holder: [item for item in constraint_occupancy],
                            agent.constraint_bandwidth_holder: [item for item in constraint_bandwidth],
                            agent.constraint_latency_holder: [item for item in constraint_latency],
                            agent.lambda_occupancy_pythonVar_holder: agent.lambda_occupancy,
                            agent.lambda_bandwidth_pythonVar_holder: agent.lambda_bandwidth,
                            agent.lambda_latency_pythonVar_holder: agent.lambda_latency}

                    # Update our value estimator (critic)
                    feed_dict_ve = {agent.input_: networkServices.state,
                                    agent.valueEstimator.target: lagrangian}

                    _, loss = sess.run([agent.valueEstimator.train_op, agent.valueEstimator.loss], feed_dict_ve)

                    # Update actor
                    summary, _, loss_rl, expected_occupancy_violation, expected_bandwidth_violation, \
                        expected_latency_violation, _, _, _ = sess.run([agent.merged, agent.train_step, agent.loss_rl,
                                                                        agent.expected_occupancy_violation,
                                                                        agent.expected_bandwidth_violation,
                                                                        agent.expected_latency_violation,
                                                                        agent.update_lambda_occupancy_tensorboardVar,
                                                                        agent.update_lambda_bandwidth_tensorboardVar,
                                                                        agent.update_lambda_latency_tensorboardVar],
                                                                       feed_dict=feed)

                    # Update lambdas (Lagrange Multipliers)
                    if episode >= config.num_epoch / 2:
                        decay_factor = 0
                    else:
                        decay_factor = 1 - episode / (config.num_epoch / 2)
                    agent.lambda_occupancy += agent.lambda_occupancy_learning_rate * decay_factor * expected_occupancy_violation
                    agent.lambda_bandwidth += agent.lambda_bandwidth_learning_rate * decay_factor * expected_bandwidth_violation
                    agent.lambda_latency += agent.lambda_latency_learning_rate * decay_factor * expected_latency_violation

                    # Print learning
                    if episode == 0 or episode % 100 == 0:
                        print("------------")
                        print("Episode: ", episode)
                        # The mini-batch loss for the actor is computed as the mean of the element-wise product
                        # of the negative log-likelihood and the advantages.
                        print("Minibatch loss: ", loss_rl)
                        print("Network service[batch0]: ", networkServices.state[0])
                        print("Len[batch0]", networkServices.service_length[0])
                        print("Placement[batch0]: ", placement_[0])

                        # agent.actor.plot_attention(attention_plot[0])
                        # print("prob:", decoder_softmax[0][0])
                        # print("prob:", decoder_softmax[0][1])
                        # print("prob:", decoder_softmax[0][2])

                        print("Baseline[batch0]: ", baseline[0])
                        print("Reward[batch0]: ", reward[0])
                        print("Penalty[batch0]: ", penalty[0])
                        print("Lagrangian[batch0]: ", lagrangian[0])
                        print("Value Estimator loss: ", np.mean(loss))
                        print("Mean penalty: ", np.mean(penalty))
                        print("Count_nonzero: ", np.count_nonzero(penalty))

                        print(f"Lambda Occupancy: {agent.lambda_occupancy}")
                        print(f"Lambda Bandwidth: {agent.lambda_bandwidth}")
                        print(f"Lambda Latency: {agent.lambda_latency}")

                    if episode % 10 == 0:
                        # Save in summary
                        writer.add_summary(summary, episode)

                    if config.save_model and (episode == 0 or episode % 100 == 0):
                        # Save in csv
                        csvData = ['batch: {}'.format(episode),
                                   ' network_service[batch 0]: {}'.format(networkServices.state[0]),
                                   ' placement[batch 0]: {}'.format(placement_[0]),
                                   ' reward: {}'.format(np.mean(reward)),
                                   ' lagrangian: {}'.format(np.mean(lagrangian)),
                                   ' baseline: {}'.format(np.mean(baseline)),
                                   ' advantage: {}'.format(np.mean(lagrangian) - np.mean(baseline)),
                                   ' penalty: {}'.format(np.mean(penalty)),
                                   ' minibatch_loss: {}'.format(loss_rl),
                                   ' Lambda_Occupancy: {}'.format(agent.lambda_occupancy),
                                   ' Lambda_Bandwidth: {}'.format(agent.lambda_bandwidth),
                                   ' Lambda_Latency: {}'.format(agent.lambda_latency)]

                        filePath = "{}/learning_history.csv".format(config.save_to)
                        with open(filePath, 'a') as csvFile:
                            writer2 = csv.writer(csvFile)
                            writer2.writerow(csvData)

                        csvFile.close()

                    # Save intermediary model variables
                    if config.save_model and episode % max(1, int(config.num_epoch / 5)) == 0 and episode != 0:
                        save_path = saver.save(sess, "{}/tmp.ckpt".format(config.save_to), global_step=episode)
                        print("\nModel saved in file: %s" % save_path)

                    episode += 1

                print("\nLearning COMPLETED!")

            except KeyboardInterrupt:
                print("\nLearning interrupted by user.")

            # Save model
            if config.save_model:
                save_path = saver.save(sess, "{}/tf_placement.ckpt".format(config.save_to))
                print("\nModel saved in file: %s" % save_path)
