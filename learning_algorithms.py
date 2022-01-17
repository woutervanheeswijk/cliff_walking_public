import numpy as np

from environment import (
    init_env,
    mark_path,
    check_game_over,
    encode_vector,
    get_state,
    get_position,
)
from qtable import init_q_table, update_q_table
from actions import (
    epsilon_greedy_action,
    move_agent,
    get_max_qvalue,
    get_reward,
    compute_cum_rewards
)

# For deep learning
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

import random

STATE_DIM = 48
ACTION_DIM = 4

# TODO: separate files for each learning algorithm
# TODO: class structure for agents


def qlearning(sim_input, sim_output) -> (np.array, list):
    """
    Q-learning algorithm
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

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
            game_over = check_game_over(episode, next_state, cliff_pos, goal_pos, num_steps)

            # Determine maximum Q-value next state (off-policy)
            max_qvalue_next_state = get_max_qvalue(next_state, q_table)

            # Update Q-table
            q_table = update_q_table(
                q_table, state, action, reward, max_qvalue_next_state, gamma, alpha
            )

            num_steps += 1

        steps_cache[episode] = num_steps

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Q-learning")

    return q_table, sim_output


def sarsa(sim_input, sim_output) -> (np.array, list):
    """
    SARSA: on-policy RL algorithm to train agent
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

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
            game_over = check_game_over(episode,
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

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)  # array of np arrays
    sim_output.name_cache.append("SARSA")

    return q_table, sim_output


def monte_carlo(sim_input, sim_output) -> (np.array, list):
    """
    Monte Carlo: full-trajectory RL algorithm to train agent
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

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

        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []

        while not game_over:

            # Initialize state at start of new episode
            if steps_cache[episode] == 0:
                # Get state corresponding to agent position
                state = get_state(agent_pos)

                # Select action using ε-greedy policy
                action = epsilon_greedy_action(state, q_table, epsilon)

            # Retrieve state
            state = get_state(agent_pos)

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)

            # Check whether game is over
            game_over = check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            # Select next action using ε-greedy policy
            next_action = epsilon_greedy_action(next_state, q_table, epsilon)

            # Update state and action
            action = next_action

            steps_cache[episode] += 1

        # At end of episode, update Q-table for full trajectory
        for t in range(len(reward_trajectory)-1, 0, -1):

            reward = reward_trajectory[t]
            action = action_trajectory[t]
            state = state_trajectory[t]

            cum_reward = compute_cum_rewards(gamma, t, reward_trajectory) + reward
            q_table[action, state] += alpha * (cum_reward - q_table[action, state])

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)  # array of np arrays
    sim_output.name_cache.append("Monte Carlo")

    return q_table, sim_output


def discrete_policy_gradient(sim_input, sim_output) -> (np.array, list):
    """
    REINFORCE with discrete policy gradient (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)


    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)

    def update_action_probabilities(
        alpha: float,
        gamma: float,
        theta: np.array,
        state_trajectory: list,
        action_trajectory: list,
        reward_trajectory: list,
        probs_trajectory: list,
    ) -> np.array:

        for t in range(len(reward_trajectory)):
            state = state_trajectory[t]
            action = action_trajectory[t]
            cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)

            # Determine action probabilities with policy
            #  action_probs = pi(state)
            action_probs = probs_trajectory[t]

            # Encode action
            phi = encode_vector(action, ACTION_DIM)

            # Construct weighted state-action vector (average phi over all actions)
            weighted_phi = np.zeros((1, ACTION_DIM))

            # For demonstration only, simply copies probability vector
            for action in range(ACTION_DIM):
                action_input = encode_vector(action, ACTION_DIM)
                weighted_phi[0] += action_probs[action] * action_input[0]

            # Return score function (phi - weighted phi)
            score_function = phi - weighted_phi

            # Update theta (only update for current state, no changes for other states)
            theta[0, state] += alpha * cum_reward * score_function[0]
        return theta

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:

            # Get state corresponding to agent position
            state = get_state(agent_pos)

            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward + entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

            # Check whether game is over
            game_over = check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            steps_cache[episode] += 1

        # Update action probabilities at end of each episode
        theta = update_action_probabilities(
            alpha,
            gamma,
            theta,
            state_trajectory,
            action_trajectory,
            reward_trajectory,
            probs_trajectory,
        )

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Discrete policy gradient")

    return all_probs, sim_output


def deep_policy_gradient(sim_input, sim_output) -> (np.array, list):
    """
    Deep discrete policy gradient (Tensorflow 2.0)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha

    def cross_entropy_loss(prob_action: float, reward: float) -> float:
        """Compute cross entropy loss"""
        log_prob = tf.math.log(prob_action + 1e-5)
        loss_actor = -reward * log_prob

        return loss_actor

    def construct_actor_network(STATE_DIM: int, ACTION_DIM: int):
        """Construct the actor network with action probabilities as output"""
        inputs = layers.Input(shape=(STATE_DIM,))  # input dimension
        hidden1 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_uniform()
        )(inputs)
        hidden2 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_uniform()
        )(hidden1)
        hidden3 = layers.Dense(
            25, activation="relu", kernel_initializer=initializers.he_uniform()
        )(hidden2)
        probabilities = layers.Dense(
            ACTION_DIM, kernel_initializer=initializers.Ones(), activation="softmax"
        )(hidden3)

        actor_network = keras.Model(inputs=inputs, outputs=[probabilities])

        return actor_network

    def compute_cum_rewards(gamma: float, t: int, rewards: np.array) -> float:
        """Cumulative reward function"""
        cum_reward = 0
        for tau in range(t, len(rewards)):
            cum_reward += gamma ** (tau - t) * rewards[tau]
        return cum_reward

    def get_entropy_bonus(action_probs: np.array) -> float:
        entropy_bonus = 0
        action_probs = action_probs.numpy()
        action_probs = np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * tf.math.log(prob_action + 1e-5)

        return float(entropy_bonus)

    def update_actor_network(
        gamma: float,
        actor_network,
        state_trajectory: list,
        action_trajectory: list,
        reward_trajectory: list,
    ) -> np.array:

        my_losses = []

        # Activate gradient tape
        with tf.GradientTape() as tape:
            my_cum_rewards = []
            probs = []

            # Loop over reward trajectory
            for t in range(len(reward_trajectory)):
                # Retrieve state, action and reward
                state = state_trajectory[t]
                action = action_trajectory[t]
                cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
                my_cum_rewards.append(cum_reward)

                # Encode state
                phi = encode_vector(state, STATE_DIM)

                # Determine action probabilities with policy
                action_probs = actor_network(phi)

                # Compute cross-entropy loss
                loss_value = cross_entropy_loss(action_probs[0, action], cum_reward)

                # Append probabilities and losses
                probs.append(action_probs[0, action])
                my_losses.append(loss_value)

            # Compute gradients
            grads = tape.gradient(my_losses, actor_network.trainable_variables)

            # Apply gradients to update actor network weights
            opt.apply_gradients(zip(grads, actor_network.trainable_variables))

        return actor_network

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    stored_state_trajectories = []
    stored_action_trajectories = []
    stored_reward_trajectories = []

    # Construct actor network
    actor_network = construct_actor_network(STATE_DIM, ACTION_DIM)

    # Define optimizer
    opt = keras.optimizers.Adam(learning_rate=alpha)

    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []

        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        while not game_over:

            # Get state corresponding to agent position
            state = get_state(agent_pos)

            # Encode state
            phi = encode_vector(state, STATE_DIM)

            # Get probabilities per action from current policy
            action_probs = actor_network(phi)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward + entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)

            # Check whether game is over
            game_over = check_game_over(episode,
                next_state, cliff_pos, goal_pos, steps_cache[episode]
            )

            steps_cache[episode] += 1

        # Update action probabilities at end of each episode
        batch_size = 1
        stored_state_trajectories.append(state_trajectory)
        stored_action_trajectories.append(action_trajectory)
        stored_reward_trajectories.append(reward_trajectory)
        if episode % batch_size == 0 and episode > 0:
            for i in range(episode - batch_size, episode):
                state_trajectory = stored_state_trajectories[i]
                action_trajectory = stored_action_trajectories[i]
                reward_trajectory = stored_reward_trajectories[i]
                actor_network = update_actor_network(
                    gamma,
                    actor_network,
                    state_trajectory,
                    action_trajectory,
                    reward_trajectory,
                )

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        phi = encode_vector(state, STATE_DIM)
        action_probs = actor_network(phi)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Deep policy gradient")

    return all_probs, sim_output


def deepqlearning(sim_input, sim_output) -> (np.array, list):

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon

    def mean_squared_error_loss(q_value, reward):
        """Compute mean squared error loss"""
        loss_critic = 0.5 * (q_value - reward) ** 2

        return loss_critic

    def construct_q_network(STATE_DIM: int, ACTION_DIM: int):
        """Construct the critic network with q-values per action as output"""
        inputs = layers.Input(shape=(STATE_DIM,))  # input dimension
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
            ACTION_DIM, kernel_initializer=initializers.Zeros(), activation="linear"
        )(hidden3)

        q_network = keras.Model(inputs=inputs, outputs=[q_values])

        return q_network

    # Initialize environment and agent position
    agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

    opt = tf.keras.optimizers.Adam(learning_rate=alpha)

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    q_network = construct_q_network(STATE_DIM, ACTION_DIM)
    target_network = tf.keras.models.clone_model(q_network)  # Copy network architecture
    target_network.set_weights(q_network.get_weights())  # Copy network weights

    replay_buffer = []
    min_buffer_size = 25  # 10
    batch_size = 25  # Number of observations per update 5
    training = True
    step_counter = 0
    learning_frequency = batch_size  # Set equal to batch size for fair comparisons
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
                state_encoded = encode_vector(state, STATE_DIM)
                q_values = tf.stop_gradient(q_network(state_encoded))

                sample_epsilon = np.random.rand()
                if sample_epsilon <= epsilon and training:
                    # Select random action
                    action = np.random.choice(ACTION_DIM)
                    agent_pos = move_agent(agent_pos, action)
                else:
                    # Select action with highest q-value
                    action = int(np.argmax(q_values[0]))
                    agent_pos = move_agent(agent_pos, action)

                # Mark visited path
                env = mark_path(agent_pos, env)

                # Determine next state
                next_state = get_state(agent_pos)

                next_state_encoded = np.zeros((1, STATE_DIM))
                next_state_encoded[0, next_state] = 1

                # Compute and store reward
                reward = get_reward(next_state, cliff_pos, goal_pos)
                rewards_cache[episode] += reward

                # Store observation in replay buffer
                observation = [state, action, reward, next_state]

                # replay_buffer = []
                replay_buffer.append(observation)

                # Check whether game is over
                game_over = check_game_over(episode,
                    next_state, cliff_pos, goal_pos, steps_cache[episode]
                )

                step_counter += 1

                # Update network if (i) buffer sufficiently large and (ii) learning frequency matched and
                # (iii) in training
                if (
                    len(replay_buffer) >= min_buffer_size
                    and step_counter % learning_frequency == 0
                    and training
                ):

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
                        game_over_update = check_game_over(episode,
                            next_state, cliff_pos, goal_pos, 0
                        )

                        if game_over_update:
                            next_q_value = 0
                        else:
                            next_state_input = np.zeros((1, STATE_DIM))
                            next_state_input[0, next_state] = 1
                            next_q_values = tf.stop_gradient(
                                target_network(next_state_input)
                            )
                            next_action = np.argmax(next_q_values[0])
                            next_q_value = next_q_values[0, next_action]

                        observed_q_value = reward + (gamma * next_q_value)

                        state_input = np.zeros((1, STATE_DIM))
                        state_input[0, state] = 1

                        q_values = q_network(state_input)
                        current_q_value = q_values[0, action]

                        loss_value += mean_squared_error_loss(
                            observed_q_value, current_q_value
                        )

                    # Compute mean loss value
                    loss_value /= batch_size

                    # Compute gradients
                    grads = tape.gradient(loss_value, q_network.trainable_variables)

                    # Apply gradients to update q-network weights
                    opt.apply_gradients(zip(grads, q_network.trainable_variables))

                    # Periodically update target network
                    if episode % update_frequency_target_network == 0:
                        target_network.set_weights(q_network.get_weights())

                steps_cache[episode] += 1

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Deep Q-learning")

    return q_network, sim_output
