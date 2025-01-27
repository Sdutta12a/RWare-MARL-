import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the grid environment
grid_size = 10
start_position = (0, 0)
goal_position = (9, 9)

# Define obstacles (bottom-left corner coordinates and dimensions)
obstacles = [
    (2, 2, 3, 1),  
    (5, 5, 1, 3),  
    (7, 1, 2, 2),   
    (7, 9, 1, 2),
    (0, 4, 2, 1),
    (8, 7, 1, 1),
]

# Define actions
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}
num_actions = len(actions)

# Initialize Q-table
Q_table = np.zeros((grid_size, grid_size, num_actions))

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 1.0    # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 2500

# Function to check if the next state collides with any obstacles
def is_collision(next_state):
    for (x, y, width, height) in obstacles:
        if x <= next_state[1] < x + width and y <= next_state[0] < y + height:
            return True
    return False

# Function to get the next state considering obstacles
def get_next_state(state, action):
    row, col = state
    d_row, d_col = actions[action]
    next_row = max(0, min(grid_size - 1, row + d_row))
    next_col = max(0, min(grid_size - 1, col + d_col))
    
    next_state = (next_row, next_col)
    
    if is_collision(next_state):
        return state  # Stay in the same state if there's a collision
    
    return next_state

# Training the agent
for episode in range(num_episodes):
    state = start_position
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(list(actions.keys()))  # Explore
        else:
            action = np.argmax(Q_table[state[0], state[1]])  # Exploit

        # Take action and observe the next state and reward
        next_state = get_next_state(state, action)
        
        if next_state == state: 
            reward = -100  # Hit an obstacle or tried to move into it
        elif next_state == goal_position:
            reward = 100   # Reached goal position
            done = True 
        else:
            reward = -1    # Regular step penalty
        
        # Update Q-value
        best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])
        Q_table[state[0], state[1], action] += alpha * (
            reward + gamma * Q_table[next_state[0], next_state[1], best_next_action] - Q_table[state[0], state[1], action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

    # Logging the progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Testing the trained agent after training is complete
print("\nTesting the trained agent:")
state = start_position
path = [state]

while state != goal_position:
    action = np.argmax(Q_table[state[0], state[1]])
    state = get_next_state(state, action)
    path.append(state)

print("Path taken by the agent:")
print(path)

# Visualization using Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(grid_size - 0.5, -0.5)
ax.set_xticks(np.arange(0, grid_size, 1))
ax.set_yticks(np.arange(0, grid_size, 1))
ax.grid(True)

# Plotting obstacles on the grid.
for (x,y,width,height) in obstacles:
    rect = plt.Rectangle((x,y), width,height,color='gray', alpha=0.5)
    ax.add_patch(rect)

# Plot the start and goal positions.
ax.plot(start_position[1], start_position[0], 'go', markersize=12)   # Start in green.
ax.plot(goal_position[1], goal_position[0], 'ro', markersize=12)     # Goal in red.

# Function to update the plot for each step.
def update(frame):
    ax.clear()
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(True)

    for (x,y,width,height) in obstacles:
        rect = plt.Rectangle((x,y), width,height,color='gray', alpha=0.5)
        ax.add_patch(rect)

    ax.plot(start_position[1], start_position[0], 'go', markersize=12)   # Start in green.
    ax.plot(goal_position[1], goal_position[0], 'ro', markersize=12)     # Goal in red.

    agent_position = path[frame]
    ax.plot(agent_position[1], agent_position[0], 'bs', markersize=8)     # Agent in blue.

    path_so_far = path[:frame+1]
    path_x = [pos[1] for pos in path_so_far]
    path_y = [pos[0] for pos in path_so_far]
    
    ax.plot(path_x,path_y,'r:', linewidth=2)  

# Create the animation.
ani = animation.FuncAnimation(fig, update, frames=len(path), repeat=False)

plt.show()
