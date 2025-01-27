# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # Parameters
# grid_size = 10
# num_episodes = 5000
# learning_rate = 0.1
# discount_factor = 0.9
# epsilon = 0.1  # exploration rate

# # Possible actions
# actions = ['up', 'down', 'left', 'right']

# # Initialize Q-table
# Q = np.zeros((grid_size, grid_size, len(actions)))

# # Set a goal position
# goal_position = (9, 9)

# # Define the environment and the rewards
# def get_reward(current_position):
#     if current_position == goal_position:
#         return 10  # Reward for reaching the goal
#     else:
#         return -1  # Small penalty for each step

# # Move agent in the grid
# def move(position, action):
#     x, y = position
#     if action == 'up':
#         x = max(0, x - 1)
#     elif action == 'down':
#         x = min(grid_size - 1, x + 1)
#     elif action == 'left':
#         y = max(0, y - 1)
#     elif action == 'right':
#         y = min(grid_size - 1, y + 1)
#     return (x, y)

# # Training loop
# for episode in range(num_episodes):
#     state = (0, 0)  # Start position
#     done = False
    
#     while not done:
#         # Epsilon-greedy action selection
#         if random.uniform(0, 1) < epsilon:
#             action_index = random.randint(0, len(actions) - 1)  # Explore
#         else:
#             action_index = np.argmax(Q[state[0], state[1]])  # Exploit

#         action = actions[action_index]
#         new_state = move(state, action)
        
#         # Get reward
#         reward = get_reward(new_state)
        
#         # Update Q-value
#         best_future_q = np.max(Q[new_state[0], new_state[1]])
#         Q[state[0], state[1], action_index] += learning_rate * (reward + discount_factor * best_future_q - Q[state[0], state[1], action_index])
        
#         # Transition to the new state
#         state = new_state
#         if state == goal_position:
#             done = True

# # Visualization
# def visualize_policy(Q):
#     policy = np.ones((grid_size, grid_size), dtype=str)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             action_index = np.argmax(Q[i, j])
#             policy[i, j] = actions[action_index]
    
#     plt.figure(figsize=(8, 8))
#     plt.imshow(np.zeros((grid_size, grid_size)), cmap='Blues', alpha=0.5)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             plt.text(j, i, policy[i, j], ha='center', va='center', fontsize=12)
#     plt.title('Learned Policy')
#     plt.xticks(np.arange(grid_size))
#     plt.yticks(np.arange(grid_size))
#     plt.grid()
#     plt.show()

# # Visualize the learned policy
# visualize_policy(Q)
# 


#code-2

import numpy as np
import matplotlib.pyplot as plt

# Define constants
NUM_EPISODES = 1000
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
GRID_SIZE = 5  # Size of the grid environment

# Define the GridEnvironment class
class GridEnvironment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.goal_state = (self.grid_size - 1, self.grid_size - 1)

    def reset(self):
        self.agent_state = (0, 0)
        return self.agent_state

    def step(self, action):
        x, y = self.agent_state

        if action == 0:  # Move up
            x = max(0, x - 1)
        elif action == 1:  # Move down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Move left
            y = max(0, y - 1)
        elif action == 3:  # Move right
            y = min(self.grid_size - 1, y + 1)

        next_state = (x, y)
        reward = 1 if next_state == self.goal_state else -0.1

        self.agent_state = next_state
        return next_state, reward

# Define the QLearningAgent class
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

    def choose_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.randint(4)  # Explore: choose random action
        x, y = state
        return np.argmax(self.q_table[x, y])  # Exploit: choose best action

    def update_q_value(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y])

        # Q-learning update rule
        td_target = reward + GAMMA * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += ALPHA * td_error

# Define the train_agent function
def train_agent():
    env = GridEnvironment()
    agent = QLearningAgent()

    for episode in range(NUM_EPISODES):
        state = env.reset()

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update_q_value(state, action, reward, next_state)

            if next_state == env.goal_state:
                done = True

            state = next_state

    print("Training completed!")
    return agent  # Return the trained agent

# Define the simulate_agent function
def simulate_agent(agent):
    env = GridEnvironment()
    state = env.reset()

    path_x, path_y = [state[1]], [state[0]]  # Store path for visualization

    while state != env.goal_state:
        action = agent.choose_action(state)  # Use learned policy to choose action
        next_state, _ = env.step(action)

        path_x.append(next_state[1])
        path_y.append(next_state[0])

        state = next_state

    return path_x, path_y

# Define the plot_simulation function
def plot_simulation(path_x, path_y):
    plt.figure(figsize=(10, 10))
    plt.plot(path_x, path_y, marker='o', label='Path')
    plt.grid(True)
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.gca().invert_yaxis()  # Optional: invert y-axis for better visualization
    plt.title("Agent Path Simulation")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent()

    # Simulate the agent's learned behavior
    path_x, path_y = simulate_agent(trained_agent)

    # Plot the simulation results
    plot_simulation(path_x, path_y)
