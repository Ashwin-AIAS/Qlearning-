# Imports:
# --------
import gymnasium as gym
import numpy as np
import pygame
import sys

# Class 1: Define a custom environment
# --------
class RoadtoMrolympia(gym.Env):
    def __init__(self, grid_size=7, goal_coordinates=(6, 6)) -> None:
        super(RoadtoMrolympia, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 800 // grid_size
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.hell_states = []

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        pygame.display.set_caption("Ashwin")

        # Load images:
        self.background_image = pygame.image.load('background.jpg').convert()
        self.agent_image = pygame.image.load('player.png').convert_alpha()
        self.goal_image = pygame.image.load('goal.png').convert_alpha()
        self.hell_images = [
            pygame.image.load('hell1.png').convert_alpha(),
            pygame.image.load('hell2.png').convert_alpha(),
            pygame.image.load('hell3.png').convert_alpha(),
            pygame.image.load('hell4.png').convert_alpha()
        ]

        # Scale images to fit the grid cells:
        self.background_image = pygame.transform.scale(self.background_image, (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))
        self.hell_images = [pygame.transform.scale(image, (self.cell_size, self.cell_size)) for image in self.hell_images]

    # Method 1: .reset()
    # ---------
    def reset(self):
        """
        Everything must be reset
        """
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.info

    # Method 2: Add hell states
    # ---------
    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    # Method 3: .step()
    # ---------
    def step(self, action):
        # Actions:
        # --------
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size-1:
            self.state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size-1:
            self.state[1] += 1
        elif action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # Reward:
        # -------
        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward = 100
            self.done = True
        elif True in [np.array_equal(self.state, each_hell) for each_hell in self.hell_states]:
            self.reward = -10
            self.done = True
        else:  # Every other state
            self.reward = -0.01 
            self.done = False

        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.reward, self.done, self.info

    # Method 3: .render()
    # ---------
    def render(self):
        # Code for closing the window:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the background image:
        self.screen.blit(self.background_image, (0, 0))

        # Draw the grid:
        for x in range(0, self.cell_size * self.grid_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.cell_size * self.grid_size))
        for y in range(0, self.cell_size * self.grid_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.cell_size * self.grid_size, y))

        # Draw the Goal-state:
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))

        # Draw the hell-states:
        for i, each_hell in enumerate(self.hell_states):
            self.screen.blit(self.hell_images[i % len(self.hell_images)], (each_hell[1] * self.cell_size, each_hell[0] * self.cell_size))

        # Draw the agent:
        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))

        # Update contents on the window:
        pygame.display.flip()

    # Method 4: .close()
    # ---------
    def close(self):
        pygame.quit()

# Function 1: Create an instance of the environment
# -----------
def create_env(goal_coordinates, hell_state_coordinates):
    # Create the environment:
    # -----------------------
    env = RoadtoMrolympia(goal_coordinates=goal_coordinates)

    for i in range(len(hell_state_coordinates)):
        env.add_hell_states(hell_state_coordinates=hell_state_coordinates[i])

    return env

# Testing the Environment
import time

def main():
    # Define goal coordinates and hell state coordinates
    goal_coordinates = (6, 6)
    hell_state_coordinates = [(3, 3), (2, 5), (5, 2)]

    # Create the environment
    env = create_env(goal_coordinates, hell_state_coordinates)

    # Reset the environment
    state, info = env.reset()
    print(f"Initial State: {state}, Info: {info}")

    done = False

    # Run the environment loop
    try:
        while not done:
            # Sample a random action
            action = env.action_space.sample()

            # Take a step in the environment
            state, reward, done, info = env.step(action)
            print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}, Info: {info}")

            # Render the environment
            env.render()

            # Wait for a short period to visually see the agent's movement
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
