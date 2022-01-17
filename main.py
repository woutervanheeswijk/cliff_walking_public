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
from learning_algorithms import (
    qlearning,
    sarsa,
    monte_carlo,
    deepqlearning,
    discrete_policy_gradient,
    deep_policy_gradient
)

import numpy as np

if __name__ == "__main__":
    """Learn cliff walking policies with SARSA and Q-learning"""
    # Set input parameters
    class sim_init:
        def __init__(self, num_episodes, gamma, alpha, epsilon):
            self.num_episodes = num_episodes  # Number of training episodes
            self.gamma = gamma  # Discount rate γ 0.9
            self.alpha = alpha  # Learning rate α 0.001
            self.epsilon = epsilon  # Exploration rate ε

        def __str__(self):
            return "# episodes: " + str(self.num_episodes) + "gamma: " + str(self.gamma) \
                   + "alpha: " + str(self.alpha) + "epsilon: " + str(self.epsilon)

    run_algorithms = {
 #       "Q-Learning",
        "SARSA",
        "Monte Carlo",
  #      "Discrete policy gradient",
  #      "Deep policy gradient",
  #      "Deep Q-Learning",
    }



    class sim_output:
        def __init__(self, rewards_cache, step_cache, env_cache, name_cache):
            self.reward_cache = rewards_cache  # list of rewards
            self.step_cache = step_cache  # list of steps
            self.env_cache = env_cache  # list of final paths
            self.name_cache = name_cache  # list of algorithm names

    sim_output = sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[]
    )

    # Run SARSA
    if "SARSA" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=1.0, alpha=0.01, epsilon=0.1)
        q_table_sarsa, sim_output = sarsa(sim_input, sim_output)

    # Run Monte Carlo
    if "Monte Carlo" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=0.8, alpha=0.01, epsilon=0.1)
        q_table_mc, sim_output = monte_carlo(sim_input, sim_output)

    # Run Deep Q-learning
    if "Deep Q-Learning" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=0.9, alpha=0.01, epsilon=0.05)
        q_network, sim_output = deepqlearning(sim_input, sim_output)

    # Run discrete policy gradient
    if "Discrete policy gradient" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=0.8, alpha=0.001, epsilon=0)
        all_probs, sim_output = discrete_policy_gradient(sim_input, sim_output)

    # Run deep policy gradient
    if "Deep policy gradient" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=1.0, alpha=0.0005, epsilon=0)
        all_probs, sim_output = deep_policy_gradient(sim_input, sim_output)

    # Run Q-learning
    if "Q-Learning" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=1.0, alpha=0.01, epsilon=0.05)
        (qtable_qlearning, sim_output) = qlearning(sim_input, sim_output)

    # Print console output
    console_output(
        sim_output,
        sim_input.num_episodes,
    )

    # Plot output
    plot_steps(sim_output)
    plot_rewards(sim_output)
    plot_path(sim_output)
