import numpy as np


def check_game_over(
    state: int, cliff_pos: np.array, goal_pos: int, number_of_steps: int
) -> bool:
    """
    Function returns reward in the given state
    """
    # Game over when reached goal, fell down cliff, or exceeded 1000 steps
    game_over = (
        True
        if (state == goal_pos  or state in cliff_pos or number_of_steps > 100)
        else False
    )
    if state == goal_pos and number_of_steps > 1:
        print("goal reached")

    return game_over


def init_env() -> (tuple, np.array, np.array, int, bool):
    """Initialize environment and agent position"""
    agent_pos = (3, 0)  # Left-bottom corner (start)
    env = np.zeros((4, 12), dtype=int)
    env = mark_path(agent_pos, env)
    cliff_states = np.arange(37, 47)  # States for cliff tiles
    goal_state = 47  # State for right-bottom corner (destination)
    game_over = False

    return agent_pos, env, cliff_states, goal_state, game_over


def mark_path(agent: tuple, env: np.array) -> np.array:
    """
    Store path taken by agent
    Only needed for visualization
    """
    (posY, posX) = agent
    env[posY][posX] += 1

    return env


def env_to_text(env: np.array) -> str:
    """
    Convert environment to text format
    Needed for visualization in console
    """
    env = np.where(env >= 1, 1, env)

    env = np.array2string(env, precision=0, separator=" ", suppress_small=False)
    env = env.replace("[[", " |")
    env = env.replace("]]", "|")
    env = env.replace("[", "|")
    env = env.replace("]", "|")
    env = env.replace("1", "x")
    env = env.replace("0", " ")

    return env


def encode_vector(index: int, dim: int) -> list:
    """Encode vector as one-hot vector"""
    vector_encoded = np.zeros((1, dim))
    vector_encoded[0, index] = 1

    return vector_encoded



def get_state(agent_pos: tuple) -> int:
    """
    Obtain state corresponding to agent position
    """
    x_dim = 12
    (pos_x, pos_y) = agent_pos
    state = x_dim * pos_x + pos_y
  #  state = pos_y * x_dim + pos_x

    return state


def get_position(state: int) -> tuple:
    """
    Obtain agent position corresponding to state
    """
    x_dim = 12

    pos_x = int(np.floor(state / x_dim))
    pos_y = state % x_dim

    agent_pos = (pos_x, pos_y)

    return agent_pos
