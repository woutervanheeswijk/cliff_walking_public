import numpy as np


def init_q_table(x_dim: int = 12, y_dim: int = 4) -> np.array:
    """
    Initialize Q-table to store values state-action pairs
    Set Q(s, a) = 0, for all s ∈ S, a ∈ A(s)
    """
    # Initialize Q-table (4 actions per state) with zeros
    q_table = np.zeros((4, x_dim * y_dim))

    return q_table


def update_q_table(
    q_table: np.array,
    state: int,
    action: int,
    reward: int,
    next_state_value: float,
    gamma: float,
    alpha: float,
) -> np.array:
    """
    Update Q-table based on observed rewards and next state value
    For SARSA (on-policy):
    Q(S, A) <- Q(S, A) + [α * (r + (γ * Q(S', A'))) -  Q(S, A)]

    For Q-learning (off-policy):
    Q(S, A) <- Q(S, A) + [α * (r + (γ * max(Q(S', A*)))) -  Q(S, A)
    """
    # Compute new q-value
    new_q_value = q_table[action, state] + alpha * (
        reward + (gamma * next_state_value) - q_table[action, state]
    )

    # Replace old Q-value
    q_table[action, state] = new_q_value

    return q_table
