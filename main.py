"""
Implementation of the cliff walking problem presented in Sutton & Barto's Reinforcement Learning (2018) book.
The problem is solved with both Q-learning (off-policy learning) and SARSA (on-policy learning).
Carefully following the algorithmic steps should give a feeling for the basis of Reinforcement Learning.
The output consists of the learned path, the average rewards over time, and average # steps over time.
Author: W.J.A. van Heeswijk
9-6-2021
"""

# import modules
from plot import plot_steps, plot_rewards, console_output, plot_path
from learning_algorithms import qlearning, sarsa, deepqlearning

if __name__ == "__main__":
    """Learn cliff walking policies with SARSA and Q-learning"""
    # Set input parameters
    num_episodes = 10000  # Number of training episodes
    gamma = 0.9  # Discount rate γ 0.9
    alpha = 0.001  # Learning rate α 0.001
    epsilon = 0.05  # Exploration rate ε

    # Run Deep Q-learning
    q_network, env_deepqlearning, steps_cache_deepqlearning, rewards_cache_deepqlearning = deepqlearning(
        num_episodes, gamma, alpha, epsilon
    )

    # Run SARSA
    q_table_sarsa, env_sarsa, steps_cache_sarsa, rewards_cache_sarsa = sarsa(
        num_episodes, gamma, 0.1, epsilon
    )

    # Run Q-learning
    (
        qtable_qlearning,
        env_qlearning,
        steps_cache_qlearning,
        rewards_cache_qlearning,
    ) = qlearning(num_episodes, gamma, 0.1, epsilon)

    # Print console output
    console_output(
        env_sarsa,
        env_qlearning,
        env_deepqlearning,
        steps_cache_sarsa,
        steps_cache_qlearning,
        steps_cache_deepqlearning,
        rewards_cache_sarsa,
        rewards_cache_qlearning,
        rewards_cache_deepqlearning,
        num_episodes,
    )

    # Plot output
    plot_steps(steps_cache_qlearning, steps_cache_sarsa, steps_cache_deepqlearning)
    plot_rewards(rewards_cache_qlearning, rewards_cache_sarsa, rewards_cache_deepqlearning)
    plot_path(env_sarsa, env_qlearning, env_deepqlearning)
