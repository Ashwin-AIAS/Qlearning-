# Imports:
# --------
from env_2057 import create_env
from q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

learning_rate = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1000  # Number of episodes

goal_coordinates = (6, 6)
# Define all hell state coordinates as a tuple within a list
hell_state_coordinates = [(2, 1), (1, 4), (4, 3), (5, 5)]

# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(goal_coordinates=goal_coordinates,
                     hell_state_coordinates=hell_state_coordinates)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy")
