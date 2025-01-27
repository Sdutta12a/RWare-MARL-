import numpy as np

# Define the environment
n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 14  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8     #alpha
discount_factor = 0.95  #gamma
exploration_prob = 0.2  #epsilon
epochs = 300

# Q-learning algorithm
for epoch in range(epochs):
    # current_state = np.random.randint(0, n_states)  # Start from a random state
    current_state = 15
    while current_state != goal_state:
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # action = 1 
        # Move to the next state
        next_state = (current_state + 1) % n_states
        # next_state = (current_state + 1) 

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        reward = 1 if next_state == goal_state else 0

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
            np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state  # Move to the next state

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)