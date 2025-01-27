# import numpy as np

# # Define the warehouse environment
# grid_size = 10  # 10x10 grid
# pickup_stations = [(5, 0), (6, 0)]  # Pickup stations
# shelves = [(5, i) for i in range(1, 10)] + [(6, i) for i in range(1, 10)]  # Shelf locations
# obstacles = []  # Define obstacles if needed

# # Q-learning parameters
# n_states = grid_size * grid_size  # Total states (10x10 grid flattened)
# n_actions = 4  # Actions: up, down, left, right
# Q_table = np.zeros((n_states, n_actions))

# learning_rate = 0.8  # alpha
# discount_factor = 0.95  # gamma
# exploration_prob = 0.2  # epsilon
# epochs = 1000

# # Helper functions
# def state_to_coords(state):
#     """Convert a state index to grid coordinates (row, col)."""
#     return divmod(state, grid_size)

# def coords_to_state(coords):
#     """Convert grid coordinates (row, col) to a state index."""
#     return coords[0] * grid_size + coords[1]

# def get_next_state(current_state, action):
#     """Simulate the environment's response to an action."""
#     row, col = state_to_coords(current_state)
#     if action == 0 and row > 0:  # Move up
#         row -= 1
#     elif action == 1 and row < grid_size - 1:  # Move down
#         row += 1
#     elif action == 2 and col > 0:  # Move left
#         col -= 1
#     elif action == 3 and col < grid_size - 1:  # Move right
#         col += 1
#     return coords_to_state((row, col))

# def get_reward(state, carrying_item):
#     """Reward function for the agent."""
#     coords = state_to_coords(state)
#     if coords in pickup_stations and not carrying_item:
#         return 10  # Reward for picking up an item
#     elif coords in shelves and carrying_item:
#         return 20  # Reward for placing the item
#     else:
#         return -1  # Small penalty for each step

# # Q-learning algorithm
# for epoch in range(epochs):
#     current_state = np.random.randint(0, n_states)  # Start at a random state
#     carrying_item = False  # Track whether the agent is carrying an item

#     for step in range(100):  # Limit the number of steps per episode
#         # Choose action using epsilon-greedy strategy
#         if np.random.rand() < exploration_prob:
#             action = np.random.randint(0, n_actions)  # Explore
#         else:
#             action = np.argmax(Q_table[current_state])  # Exploit

#         # Get the next state
#         next_state = get_next_state(current_state, action)
#         reward = get_reward(next_state, carrying_item)

#         # Update carrying status
#         if state_to_coords(next_state) in pickup_stations and not carrying_item:
#             carrying_item = True
#         elif state_to_coords(next_state) in shelves and carrying_item:
#             carrying_item = False

#         # Update Q-value
#         Q_table[current_state, action] += learning_rate * (
#             reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
#         )

#         current_state = next_state  # Move to the next state

#         # End the episode if the task is complete
#         if not carrying_item and state_to_coords(current_state) in shelves:
#             break

# # Display the learned Q-table
# print("Learned Q-table:")
# print(Q_table)


# 2nd Code
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the warehouse environment
grid_size = 10  # 10x10 grid
pickup_stations = [(5, 0), (6, 0)]  # Pickup stations
shelves = [(5, i) for i in range(1, 10)] + [(6, i) for i in range(1, 10)]  # Shelf locations
obstacles = []  # Add obstacles if needed

# Q-learning parameters
n_states = grid_size * grid_size  # Total states (10x10 grid flattened)
n_actions = 4  # Actions: up, down, left, right
Q_table = np.zeros((n_states, n_actions))

learning_rate = 0.8  # alpha
discount_factor = 0.95  # gamma
exploration_prob = 0.2  # epsilon
epochs = 500

# Helper functions
def state_to_coords(state):
    """Convert a state index to grid coordinates (row, col)."""
    return divmod(state, grid_size)

def coords_to_state(coords):
    """Convert grid coordinates (row, col) to a state index."""
    return coords[0] * grid_size + coords[1]

def get_next_state(current_state, action):
    """Simulate the environment's response to an action."""
    row, col = state_to_coords(current_state)
    if action == 0 and row > 0:  # Move up
        row -= 1
    elif action == 1 and row < grid_size - 1:  # Move down
        row += 1
    elif action == 2 and col > 0:  # Move left
        col -= 1
    elif action == 3 and col < grid_size - 1:  # Move right
        col += 1
    return coords_to_state((row, col))

def get_reward(state, carrying_item):
    """Reward function for the agent."""
    coords = state_to_coords(state)
    if coords in pickup_stations and not carrying_item:
        return 10  # Reward for picking up an item
    elif coords in shelves and carrying_item:
        return 20  # Reward for placing the item
    else:
        return -1  # Small penalty for each step

def render_warehouse(agent_coords, carrying_item):
    """Visualize the warehouse layout and agent."""
    grid = np.zeros((grid_size, grid_size))

    # Mark pickup stations
    for station in pickup_stations:
        grid[station] = 0.5  # Gray for pickup stations

    # Mark shelves
    for shelf in shelves:
        grid[shelf] = 0.7  # Light gray for shelves

    # Mark obstacles (if any)
    for obstacle in obstacles:
        grid[obstacle] = -1  # Black for obstacles

    # Mark agent's position
    grid[agent_coords] = 1.0 if carrying_item else 0.9  # Green if carrying, yellow otherwise

    # Plot the grid
    plt.imshow(grid, cmap="coolwarm", origin="upper")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(color="black", linestyle="-", linewidth=0.5)
    plt.title(f"Agent {'Carrying Item' if carrying_item else 'Empty-Handed'}")
    plt.pause(0.1)
    plt.clf()

# Q-learning algorithm with visualization
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start at a random state
    carrying_item = False  # Track whether the agent is carrying an item

    for step in range(100):  # Limit the number of steps per episode
        agent_coords = state_to_coords(current_state)
        render_warehouse(agent_coords, carrying_item)  # Visualize the environment

        # Choose action using epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Get the next state
        next_state = get_next_state(current_state, action)
        reward = get_reward(next_state, carrying_item)

        # Update carrying status
        if state_to_coords(next_state) in pickup_stations and not carrying_item:
            carrying_item = True
        elif state_to_coords(next_state) in shelves and carrying_item:
            carrying_item = False

        # Update Q-value
        Q_table[current_state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )

        current_state = next_state  # Move to the next state

        # End the episode if the task is complete
        if not carrying_item and state_to_coords(current_state) in shelves:
            break

plt.show()
