import numpy as np

from environment import init_env, mark_path, check_game_over
from qtable import init_q_table, update_q_table
from actions import (
    epsilon_greedy_action,
    move_agent,
    get_state,
    get_max_qvalue,
    get_reward,
)

# For deep learning
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

import random


def qlearning(
    num_episodes: int, gamma: float, alpha: float, epsilon: float
) -> (np.array, list):
    """
    Q-learning algorithm
    """
    q_table = init_q_table()
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        # Set to target policy at final episode
        if episode == len(range(num_episodes)) - 1:
            epsilon = 0

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()
        num_steps = 0

        while not game_over:
            # Get state corresponding to agent position
            state = get_state(agent_pos)

            # Select action using ε-greedy policy
            action = epsilon_greedy_action(state, q_table, epsilon)

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward

            # Check whether game is over
            game_over = check_game_over(next_state, cliff_pos, goal_pos, num_steps)

            # Determine maximum Q-value next state (off-policy)
            max_qvalue_next_state = get_max_qvalue(next_state, q_table)

            # Update Q-table
            q_table = update_q_table(
                q_table, state, action, reward, max_qvalue_next_state, gamma, alpha
            )

            num_steps += 1

        steps_cache[episode] = num_steps

    return q_table, env, steps_cache, rewards_cache


def sarsa(num_episodes, gamma: float, alpha: float, epsilon: float) -> (np.array, list):
    """
    SARSA: on-policy RL algorithm to train agent
    """

    q_table = init_q_table()
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        # Set to target policy at final episode
        if episode == len(range(num_episodes)) - 1:
            epsilon = 0

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:

            if steps_cache[episode] == 0:
                # Get state corresponding to agent position
                state = get_state(agent_pos)

                # Select action using ε-greedy policy
                action = epsilon_greedy_action(state, q_table, epsilon)

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward

            # Check whether game is over
            game_over = check_game_over(
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            # Select next action using ε-greedy policy
            next_action = epsilon_greedy_action(next_state, q_table, epsilon)

            # Determine Q-value next state (on-policy)
            next_state_value = q_table[next_action][next_state]

            # Update Q-table
            q_table = update_q_table(
                q_table, state, action, reward, next_state_value, gamma, alpha
            )

            # Update state and action
            state = next_state
            action = next_action

            steps_cache[episode] += 1

    return q_table, env, steps_cache, rewards_cache


def deepqlearning(
        num_episodes: int, gamma: float, alpha: float, epsilon: float
) -> (np.array, list):
    def mean_squared_error_loss(q_value, reward):
        """Compute mean squared error loss"""
        loss_critic = 0.5 * (q_value - reward) ** 2

        return loss_critic

    def construct_q_network(state_dim: int, action_dim: int):
        """Construct the critic network with q-values per action as output"""
        inputs = layers.Input(shape=(state_dim,))  # input dimension
        hidden1 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_normal()
        )(inputs)
        hidden2 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_normal()
        )(hidden1)
        hidden3 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_normal()
        )(hidden2)
        q_values = layers.Dense(
            action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
        )(hidden3)

        q_network = keras.Model(inputs=inputs, outputs=[q_values])

        return q_network

    # Initialize environment and agent position
    agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

    opt = tf.keras.optimizers.Adam(learning_rate=alpha)


    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    state_dim = 48
    action_dim = 4
    q_network = construct_q_network(state_dim, action_dim)
    target_network = tf.keras.models.clone_model(q_network)  # Copy network architecture
    target_network.set_weights(q_network.get_weights())  # Copy network weights

    replay_buffer = []
    min_buffer_size = 10
    batch_size = 5  # Number of observations per update
    training = True
    step_counter = 0
    learning_frequency = batch_size # Set equal to batch size for fair comparisons
    update_frequency_target_network = 19

    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Set to target policy at final episodes
        if episode == len(range(num_episodes)) - 100:
            training = False

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:
            with tf.GradientTape() as tape:

                # Get state corresponding to agent position
                state = get_state(agent_pos)

                # Select action using ε-greedy policy
                # Obtain q-values from network
                state_input = np.zeros((1, state_dim))
                state_input[0, state] = 1
                q_values = tf.stop_gradient(q_network(state_input))

                sample_epsilon = np.random.rand()
                if sample_epsilon <= epsilon and training:
                    # Select random action
                    action = np.random.choice(action_dim)
                    agent_pos = move_agent(agent_pos, action)
                else:
                    # Select action with highest q-value
                    action = np.argmax(q_values[0])
                    agent_pos = move_agent(agent_pos, action)

                # Mark visited path
                env = mark_path(agent_pos, env)

                # Determine next state
                next_state = get_state(agent_pos)

                next_state_input = np.zeros((1, state_dim))
                next_state_input[0, next_state] = 1

                # Compute and store reward
                reward = get_reward(next_state, cliff_pos, goal_pos)
                rewards_cache[episode] += reward

                # Store observation in replay buffer
                observation = [state, action, reward, next_state]

                # replay_buffer = []
                replay_buffer.append(observation)

                # Check whether game is over
                game_over = check_game_over(
                    next_state, cliff_pos, goal_pos, steps_cache[episode]
                )

                step_counter += 1

                # Update network if (i) buffer sufficiently large and (ii) learning frequency matched and
                # (iii) in training
                if len(replay_buffer) >= min_buffer_size and step_counter % learning_frequency == 0 and training:

                    observations = random.choices(replay_buffer, k=batch_size)
                    loss_value = 0

                    # Compute mean loss
                    for observation in observations:
                        state = observation[0]
                        action = observation[1]
                        reward = observation[2]
                        next_state = observation[3]

                        # Select next action with highest q-value
                        # Check whether game is over (ignoring # steps)
                        game_over_update = check_game_over(next_state, cliff_pos, goal_pos, 0)

                        if game_over_update:
                            next_q_value = 0
                        else:
                            next_state_input = np.zeros((1, state_dim))
                            next_state_input[0, next_state] = 1
                            next_q_values = tf.stop_gradient(
                                target_network(next_state_input)
                            )
                            next_action = np.argmax(next_q_values[0])
                            next_q_value = next_q_values[0, next_action]

                        observed_q_value = reward + (gamma * next_q_value)

                        state_input = np.zeros((1, state_dim))
                        state_input[0, state] = 1

                        q_values = q_network(state_input)
                        current_q_value = q_values[0, action]

                        loss_value += mean_squared_error_loss(
                            observed_q_value, current_q_value
                        )

                    # Compute mean loss value
                    loss_value /= batch_size

                    # Compute gradients
                    grads = tape.gradient(
                        loss_value, q_network.trainable_variables
                    )

                    # Apply gradients to update q-network weights
                    opt.apply_gradients(zip(grads, q_network.trainable_variables))

                    # Periodically update target network
                    if episode % update_frequency_target_network == 0:
                        target_network.set_weights(q_network.get_weights())

                steps_cache[episode] += 1

    return q_network, env, steps_cache, rewards_cache






