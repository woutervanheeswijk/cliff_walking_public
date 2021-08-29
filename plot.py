import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environment import env_to_text


def plot_rewards(
    reward_cache_qlearning: np.array,
    reward_cache_sarsa: np.array,
    reward_cache_deepqlearning: np.array,
) -> None:
    """
    Visualizes rewards
    """
    mod = len(reward_cache_qlearning) % 10
    mean_reward_qlearning = np.mean(
        reward_cache_qlearning[mod:].reshape(-1, 10), axis=1
    )

    mod = len(reward_cache_sarsa) % 10
    mean_reward_sarsa = np.mean(reward_cache_sarsa[mod:].reshape(-1, 10), axis=1)

    mod = len(reward_cache_deepqlearning) % 10
    mean_reward_deepqlearning = np.mean(
        reward_cache_deepqlearning[mod:].reshape(-1, 10), axis=1
    )

    # Set x-axis label
    positions = np.arange(0, len(reward_cache_sarsa) / 10, 100)
    labels = np.arange(0, len(reward_cache_sarsa), 1000)

    sns.set_theme(style="darkgrid")

    sns.lineplot(data=mean_reward_sarsa, label="SARSA")
    sns.lineplot(data=mean_reward_qlearning, label="Q-learning")
    sns.lineplot(data=mean_reward_deepqlearning, label="Deep Q-learning")

    # Plot graph
    plt.xticks(positions, labels)
    plt.ylabel("rewards")
    plt.xlabel("# episodes")
    plt.legend(loc="best")

    plt.show()

    return


def plot_steps(
    steps_cache_qlearning: np.array,
    steps_cache_sarsa: np.array,
    steps_cache_deepqlearning: np.array,
) -> None:
    """
    Visualize number of steps taken
    """
    mod = len(steps_cache_qlearning) % 10
    mean_step_qlearning = np.mean(steps_cache_qlearning[mod:].reshape(-1, 10), axis=1)

    mod = len(steps_cache_sarsa) % 10
    mean_step_sarsa = np.mean(steps_cache_sarsa[mod:].reshape(-1, 10), axis=1)

    mod = len(steps_cache_deepqlearning) % 10
    mean_step_deepqlearning = np.mean(
        steps_cache_deepqlearning[mod:].reshape(-1, 10), axis=1
    )


    positions = np.arange(0, len(steps_cache_sarsa)/10, 100)
    labels = np.arange(0, len(steps_cache_sarsa), 1000)
  #  positions = np.arange(0, len(steps_cache_sarsa) / 10, 10)
  #  labels = np.arange(0, len(steps_cache_sarsa), 10)

    sns.set_theme(style="darkgrid")

    sns.lineplot(data=mean_step_sarsa, label="SARSA")
    sns.lineplot(data=mean_step_qlearning, label="Q-learning")
    sns.lineplot(data=mean_step_deepqlearning, label="Deep Q-learning")

    # Plot graph
    plt.xticks(positions, labels)
    plt.ylabel("# steps")
    plt.xlabel("# episodes")
    plt.legend(loc="best")
    plt.show()

    return


def console_output(
    env_sarsa: np.array,
    env_qlearning: np.array,
    env_deepqlearning: np.array,
    steps_cache_sarsa: np.array,
    steps_cache_qlearning: np.array,
    steps_cache_deepqlearning: np.array,
    rewards_cache_sarsa: np.array,
    rewards_cache_qlearning: np.array,
    rewards_cache_deepqlearning: np.array,
    num_episodes: int,
) -> None:
    """Print path and key metrics in console"""
    env_sarsa_str = env_to_text(env_sarsa)

    print("SARSA action after {} iterations:".format(num_episodes), "\n")
    print(env_sarsa_str, "\n")
    print("Number of steps:", int(steps_cache_sarsa[-1]), "(min. = 13)", "\n")
    print("Reward:", int(rewards_cache_sarsa[-1]), "(max. = -2)", "\n")

    env_qlearning_str = env_to_text(env_qlearning)

    print("Q-learning action after {} iterations:".format(num_episodes), "\n")
    print(env_qlearning_str, "\n")
    print("Number of steps:", int(steps_cache_qlearning[-1]), "(min. = 13)", "\n")
    print("Cumulative reward:", int(rewards_cache_qlearning[-1]), "(max. = -2)", "\n")

    env_deepqlearning_str = env_to_text(env_deepqlearning)

    print("Deep Q-learning action after {} iterations:".format(num_episodes), "\n")
    print(env_deepqlearning_str, "\n")
    print("Number of steps:", int(steps_cache_deepqlearning[-1]), "(min. = 13)", "\n")
    print("Cumulative reward:", int(rewards_cache_deepqlearning[-1]), "(max. = -2)", "\n")

    return


def plot_path(
    env_sarsa: np.array, env_qlearning: np.array, env_deepqlearning: np.array
) -> None:
    """Plot latest paths for SARSA and Q-learning as heatmap"""

    # Plot path SARSA

    # Set values for cliff
    for i in range(1, 11):
        env_sarsa[3, i] = -1

    ax = sns.heatmap(
        env_sarsa, square=True, cbar=True, xticklabels=False, yticklabels=False
    )
    ax.set_title("SARSA")
    plt.show()

    # Plot path Q-learning

    # Set values for cliff
    for i in range(1, 11):
        env_qlearning[3, i] = -1

    ax = sns.heatmap(
        env_qlearning, square=True, cbar=True, xticklabels=False, yticklabels=False
    )
    ax.set_title("Q-learning")
    plt.show()

    # Plot path Deep Q-learning

    # Set values for cliff
    for i in range(1, 11):
        env_deepqlearning[3, i] = -1

    ax = sns.heatmap(
        env_deepqlearning, square=True, cbar=True, xticklabels=False, yticklabels=False
    )
    ax.set_title("Deep Q-learning")
    plt.show()

    return None
