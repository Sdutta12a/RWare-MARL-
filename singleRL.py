# import numpy as np

# # Initialize a 4x4 grid environment with 4 actions (up, down, left, right)
# grid_size = 4
# action_space_size = 4
# state_space_size = grid_size * grid_size  # 4x4 grid = 16 states

# # Initialize Q-table with zeros
# q_table = np.zeros((state_space_size, action_space_size))

# # Visualize the initial Q-table
# print("Initial Q-table:")
# print(q_table)


# # Q-learning parameters
# alpha = 0.1      # Learning rate
# gamma = 0.9      # Discount factor
# epsilon = 0.1    # Exploration rate

# # Simulate a few steps of Q-learning updates

# # Example state transitions and rewards
# state = 0         # Starting at state 0 (top-left corner of the grid)
# action = 3        # Let's say the agent chooses action 'right' (action 3)
# reward = 1        # Reward for taking this action (e.g., moving to the next state)
# next_state = 1    # After taking the action, the agent moves to state 1

# # Update the Q-value for the chosen state-action pair (0, 3)
# q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

# # Print the Q-table after the update
# print("\nUpdated Q-table (after one step):")
# print(q_table)



# import matplotlib.pyplot as plt

# # Function to plot the Q-table as a heatmap
# def plot_q_table(q_table):
#     plt.imshow(q_table, cmap='coolwarm', aspect='auto')
#     plt.colorbar()
#     plt.title("Q-Table Heatmap")
#     plt.xlabel("Actions")
#     plt.ylabel("States")
#     plt.show()

# # Initial Q-table plot
# print("Visualizing Initial Q-table...")
# plot_q_table(q_table)

# # Simulate more updates (you can extend this further)
# # For simplicity, I will update Q-table for a few more states and actions
# for _ in range(5):  # Simulate 5 more updates
#     state = np.random.randint(0, state_space_size)
#     action = np.random.randint(0, action_space_size)
#     reward = np.random.choice([0, 1])  # Random reward (either 0 or 1)
#     next_state = np.random.randint(0, state_space_size)

#     q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

# # Plot updated Q-table
# print("Visualizing Updated Q-table...")
# plot_q_table(q_table)




import numpy as np
import matplotlib.pyplot as plt

# Initialize a 4x4 grid environment with 4 actions (up, down, left, right)
grid_size = 4
action_space_size = 4
state_space_size = grid_size * grid_size  # 4x4 grid = 16 states

# Initialize Q-table with zeros
q_table = np.zeros((state_space_size, action_space_size))

# Q-learning parameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 1.0    # Initial exploration rate
epsilon_decay = 0.99  # Decay rate for exploration
min_epsilon = 0.1     # Minimum exploration rate

def update_q_table(state, action, reward, next_state):
    """Update the Q-value for the given state-action pair."""
    best_next_action_value = np.max(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * best_next_action_value - q_table[state, action])

def choose_action(state):
    """Choose an action based on epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.randint(0, action_space_size)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def plot_q_table(q_table):
    """Plot the Q-table as a heatmap."""
    plt.imshow(q_table, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title("Q-Table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

# Simulate a few steps of Q-learning updates
for episode in range(100):  # Simulate multiple episodes
    state = np.random.randint(0, state_space_size)  # Random starting state
    
    for _ in range(10):  # Simulate steps within each episode
        action = choose_action(state)
        reward = np.random.choice([0, 1])  # Random reward (either 0 or 1)
        next_state = (state + 1) % state_space_size  # Simple transition logic for demonstration
        
        update_q_table(state, action, reward, next_state)
        
        state = next_state
    
    # Decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Visualize the initial and updated Q-table
print("Visualizing Updated Q-table...")
plot_q_table(q_table)
