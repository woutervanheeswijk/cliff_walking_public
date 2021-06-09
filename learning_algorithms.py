import numpy as np

from environment import init_env, mark_path, check_game_over, env_to_text
from qtable import init_q_table, update_q_table
from actions import (
    epsilon_greedy_action,
    move_agent,
    get_state,
    get_max_qvalue,
    get_reward,
)


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
