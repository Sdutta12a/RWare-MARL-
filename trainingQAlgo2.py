# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # Parameters
# grid_size = 10
# num_episodes = 5000
# learning_rate = 0.1
# discount_factor = 0.9
# epsilon_start = 1.0
# epsilon_end = 0.1
# epsilon_decay = 0.995

# # Possible actions
# actions = ['up', 'down', 'left', 'right']
# action_mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

# # Initialize Q-table
# Q = np.zeros((grid_size, grid_size, len(actions)))

# # Set a goal position
# goal_position = (9, 9)

# # Set obstacles
# obstacles = {(3, 3), (3, 4), (4, 3), (6, 6), (6, 7), (7, 6)}

# # Define the reward function
# def get_reward(current_position):
#     if current_position == goal_position:
#         return 10  # Reward for reaching the goal
#     elif current_position in obstacles:
#         return -10  # Penalty for hitting an obstacle
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
# cumulative_rewards = []  # For tracking learning progress
# epsilon = epsilon_start

# for episode in range(num_episodes):
#     state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
#     while state in obstacles or state == goal_position:  # Ensure valid start position
#         state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    
#     done = False
#     total_reward = 0  # Cumulative reward for the episode
    
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
#         total_reward += reward
        
#         # Update Q-value
#         best_future_q = np.max(Q[new_state[0], new_state[1]])
#         Q[state[0], state[1], action_index] += learning_rate * (
#             reward + discount_factor * best_future_q - Q[state[0], state[1], action_index])
        
#         # Transition to the new state
#         state = new_state
#         if state == goal_position or state in obstacles:
#             done = True
    
#     # Decay epsilon
#     epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
#     # Track cumulative reward
#     cumulative_rewards.append(total_reward)

# # Visualization of learning progress
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Reward')
# plt.title('Learning Progress')
# plt.show()

# # Visualization of the learned policy
# def visualize_policy(Q):
#     policy = np.full((grid_size, grid_size), ' ')
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if (i, j) == goal_position:
#                 policy[i, j] = 'G'
#             elif (i, j) in obstacles:
#                 policy[i, j] = 'X'
#             else:
#                 action_index = np.argmax(Q[i, j])
#                 policy[i, j] = actions[action_index][0].upper()
    
#     plt.figure(figsize=(8, 8))
#     plt.imshow(np.zeros((grid_size, grid_size)), cmap='Blues', alpha=0.5)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             plt.text(j, i, policy[i, j], ha='center', va='center', fontsize=12, color='black')
#     plt.title('Learned Policy (G: Goal, X: Obstacle)')
#     plt.xticks(np.arange(grid_size))
#     plt.yticks(np.arange(grid_size))
#     plt.grid()
#     plt.show()

# # Visualize the learned policy
# visualize_policy(Q)



import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
grid_size = 10
num_episodes = 100
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # exploration rate (fixed)

# Possible actions
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, len(actions)))

# Set a goal position
goal_position = (9, 9)

# Define obstacles using (x, y, width, height)
obstacles = [
    (1, 1, 3, 3),  # A block from (1,1) to (3,3)
    (5, 2, 4, 4),  # A block from (5,2) to (8,5)
    (7, 7, 2, 2)   # A block from (7,7) to (8,8)
]

# Function to check if a position is in an obstacle
def is_obstacle(position):
    x, y = position
    for (ox, oy, w, h) in obstacles:
        if ox <= x < ox + w and oy <= y < oy + h:
            return True
    return False

# Define the environment and the rewards
def get_reward(current_position):
    if current_position == goal_position:
        return 10  # Reward for reaching the goal
    elif is_obstacle(current_position):
        return -10  # Heavy penalty for hitting an obstacle
    else:
        return -1  # Small penalty for each step

# Move agent in the grid
def move(position, action):
    x, y = position
    if action == 'up':
        new_position = (max(0, x - 1), y)
    elif action == 'down':
        new_position = (min(grid_size - 1, x + 1), y)
    elif action == 'left':
        new_position = (x, max(0, y - 1))
    elif action == 'right':
        new_position = (x, min(grid_size - 1, y + 1))
    
    # Only return new_position if it is not an obstacle
    return new_position if not is_obstacle(new_position) else position

# Training loop
for episode in range(num_episodes):
    state = (0, 0)  # Start position
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(actions) - 1)  # Explore
        else:
            action_index = np.argmax(Q[state[0], state[1]])  # Exploit

        action = actions[action_index]
        new_state = move(state, action)
        
        # Get reward
        reward = get_reward(new_state)
        
        # Update Q-value
        best_future_q = np.max(Q[new_state[0], new_state[1]])
        Q[state[0], state[1], action_index] += learning_rate * (reward + discount_factor * best_future_q - Q[state[0], state[1], action_index])
        
        # Transition to the new state
        state = new_state
        if state == goal_position:
            done = True

# Visualization of obstacles and path
def visualize_grid(Q):
    plt.figure(figsize=(8, 8))
    
    # Plotting the grid
    for (ox, oy, w, h) in obstacles:
        plt.gca().add_patch(plt.Rectangle((oy, ox), w, h, color='black'))  # Draw obstacles

    plt.scatter(goal_position[1], goal_position[0], color='green', s=100, label='Goal')  # Draw goal
    plt.scatter(0, 0, color='blue', s=100, label='Start')  # Draw start
    
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.grid()
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.gca().invert_yaxis()
    plt.title('Grid with Obstacles')
    plt.legend()
    plt.show()

# # Create a table containing Q-values
# def visualize_q_values(Q):
#     plt.figure(figsize=(12, 6))
#     q_values = Q.reshape((grid_size * grid_size, len(actions)))
#     plt.imshow(q_values, cmap='hot', interpolation='nearest')
#     plt.colorbar()
    
#     # Set ticks for action labels
#     action_labels = ['Up', 'Down', 'Left', 'Right']
#     plt.xticks(ticks=np.arange(len(actions)), labels=action_labels)
#     plt.yticks(ticks=np.arange(grid_size * grid_size), labels=[f'State {i}' for i in range(grid_size * grid_size)])
#     plt.title('Q-values for each State-Action Pair')
#     plt.xlabel('Actions')
#     plt.ylabel('States')
#     plt.show()

# Visualize the grid and the Q-values
visualize_grid(Q)
# visualize_q_values(Q)


# Create a heat map for Q-values
def visualize_q_heatmap(Q):
    plt.figure(figsize=(12, 6))
    
    # Calculate average Q-values for each state
    average_q_values = np.mean(Q, axis=2)  # Average over actions (3rd dimension)
    
    plt.imshow(average_q_values, cmap='hot', interpolation='nearest')
    plt.colorbar()
    
    plt.title('Average Q-values Heatmap')
    plt.xlabel('Grid Columns')
    plt.ylabel('Grid Rows')
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.gca().invert_yaxis()  # Invert the y-axis to match grid coordinates
    plt.show()

# Visualize the heat map
visualize_q_heatmap(Q)


