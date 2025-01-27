# import gymnasium as gym
# import rware
# import time

# # Define a custom rectangular layout for the large environment
# layout = """
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# g..........xxxxxxxxx.
# g..........xxxxxxxxx.
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................

# """

# # Create the environment with the custom layout
# env_layout = gym.make("rware:rware-large-2ag-v2", layout=layout)

# # Reset the environment before rendering
# env_layout.reset()

# # Render the environment in a loop to keep the window open
# try:
#     while True:
#         env_layout.render()  # Render the environment
#         time.sleep(0.1)  # Add a delay to stabilize the rendering (adjust as needed)
# except KeyboardInterrupt:
#     print("Exiting...")

# # Close the environment after use
# env_layout.close()



# from rware.warehouse import ObservationType, Warehouse, Direction, Action, RewardType


# import gymnasium as gym
# import rware
# import time
# import random

# # Define the custom layout
# layout = """
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# g..........xxxxxxxxx.
# g..........xxxxxxxxx.
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# """

# # Create the environment with the custom layout
# env_layout = gym.make("rware:rware-large-2ag-v2", layout=layout)

# # Reset the environment
# obs, info = env_layout.reset()
# print("Environment reset!")

# # List of all shelves 
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to the layout

# # Task buffer initialization
# task_buffer = [random.choice(shelves)]  # Randomly sample one task
# print("Initial task buffer:", task_buffer)

# def update_task_buffer(task_buffer, shelves):
#     """
#     Update the task buffer by adding a new task when one is completed.
#     """
#     new_task = random.choice(shelves)  # Randomly sample from available shelves
#     task_buffer.append(new_task)
#     print("Updated task buffer:", task_buffer)

# def update_environment_state(env, task_buffer):
#     """
#     Visualize the task buffer in the environment by marking requested shelves.
#     """
#     for task in task_buffer:
#         row, col = task
#         # Render requested shelves (use env's render or metadata functionality)
#         print(f"Requested shelf at ({row}, {col})")

# try:
#     while True:
#         # Update environment visualization
#         update_environment_state(env_layout, task_buffer)

#         # Example agent action (replace with actual logic)
#         actions = env_layout.action_space.sample()  # Random action

#         # Step through the environment
#         obs, reward, terminated, truncated, info = env_layout.step(actions)

#         # Check if the episode has ended (terminated or truncated)
#         if terminated or truncated:
#             print("Episode ended. Resetting environment...")
#             obs, info = env_layout.reset()

#         # Example: Complete task if agent reaches it (implement your task completion logic here)
#         if random.random() < 0.1:  # Simulate task completion
#             if task_buffer:  # Check if there are tasks to complete
#                 completed_task = task_buffer.pop(0)
#                 print(f"Completed task at {completed_task}")
#                 update_task_buffer(task_buffer, shelves)

#         # Render the environment
#         env_layout.render()
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Exiting...")

# # Close the environment
# env_layout.close()

# import gymnasium as gym
# import rware
# import time
# import random

# # Define the custom layout
# layout = """
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# g..........xxxxxxxxx.
# g..........xxxxxxxxx.
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# """

# # Create the environment with the custom layout
# env_layout = gym.make("rware:rware-large-2ag-v2", layout=layout)

# # Reset the environment
# obs, info = env_layout.reset()
# print("Environment reset!")

# # List of all shelves (initialize based on your layout)
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to the layout

# # Task buffer initialization
# task_buffer = [random.choice(shelves)]  # Randomly sample one task
# print("Initial task buffer:", task_buffer)

# def update_task_buffer(task_buffer, shelves):

#     new_task = random.choice(shelves)  # Randomly sample from available shelves
#     task_buffer.append(new_task)
#     print("Updated task buffer:", task_buffer)

# def update_environment_state(env, task_buffer):
#     for task in task_buffer:
#         row, col = task
#         # Render requested shelves (use env's render or metadata functionality)
#         print(f"Requested shelf at ({row}, {col})")

# try:
#     while True:
#         # Update environment visualization
#         update_environment_state(env_layout, task_buffer)

#         # Example agent action (replace with actual logic)
#         actions = env_layout.action_space.sample()  # Random action

#         # Step through the environment
#         obs, reward, terminated, truncated, info = env_layout.step(actions)

#         # Check if the episode has ended (terminated or truncated)
#         if terminated or truncated:
#             print("Episode ended. Resetting environment...")
#             obs, info = env_layout.reset()

#         # Example: Complete task if agent reaches it (implement your task completion logic here)
#         if random.random() < 0.1:  # Simulate task completion
#             if task_buffer:  # Check if there are tasks to complete
#                 completed_task = task_buffer.pop(0)
#                 print(f"Completed task at {completed_task}")
#                 update_task_buffer(task_buffer, shelves)

#         # Render the environment
#         env_layout.render()
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Exiting...")

# # Close the environment
# env_layout.close()


# import gymnasium as gym
# import rware
# import time
# import random
# import numpy as np

# # Define the custom layout
# layout = """
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# g..........xxxxxxxxx.
# g..........xxxxxxxxx.
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# """

# # Create the environment with the custom layout
# env_layout = gym.make("rware:rware-large-2ag-v2", layout=layout)

# # Reset the environment
# obs, info = env_layout.reset()
# print("Environment reset!")

# # List of all shelves (initialize based on your layout)
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to the layout

# # Task buffer initialization with all shelves
# task_buffer = shelves.copy()  # Start with all shelves
# print("Initial task buffer:", task_buffer)

# # Q-learning parameters
# alpha = 0.1  # Learning rate
# gamma = 0.9  # Discount factor
# epsilon = 0.1  # Exploration rate

# # Number of actions per agent (multi-agent environment)
# num_actions_per_agent = [space.n for space in env_layout.action_space]
# print("Number of actions per agent:", num_actions_per_agent)

# # Initialize Q-tables for each agent
# q_tables = [np.zeros((len(shelves), num_actions)) for num_actions in num_actions_per_agent]

# def update_environment_state(env, task_buffer):
#     for task in task_buffer:
#         row, col = task
#         print(f"Requested shelf at ({row}, {col})")

# def choose_action(agent_index, state_index):
#     if random.random() < epsilon:
#         return random.randint(0, num_actions_per_agent[agent_index] - 1)  # Explore
#     else:
#         return np.argmax(q_tables[agent_index][state_index])  # Exploit

# def update_task_buffer(task_buffer):
#     if task_buffer:  # Check if there are tasks to complete
#         completed_task = task_buffer.pop(0)
#         print(f"Completed task at {completed_task}")
#         # Add new task logic here if needed

# try:
#     while True:
#         current_state_indices = [0] * len(q_tables)  # Example: Initialize state indices for agents

#         # Update environment visualization
#         update_environment_state(env_layout, task_buffer)

#         # Choose actions for all agents using epsilon-greedy strategy
#         actions = [choose_action(agent_idx, current_state_indices[agent_idx]) for agent_idx in range(len(q_tables))]

#         # Step through the environment with multi-agent actions
#         obs, rewards, terminated, truncated, info = env_layout.step(actions)

#         # Update Q-tables based on rewards received and next state values
#         next_state_indices = current_state_indices  # Define how to get next state indices from obs
        
#         for agent_idx in range(len(q_tables)):
#             best_next_action = np.argmax(q_tables[agent_idx][next_state_indices[agent_idx]])
#             q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]] += alpha * (
#                 rewards[agent_idx] + gamma * q_tables[agent_idx][next_state_indices[agent_idx]][best_next_action] -
#                 q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]]
#             )

#         # Check if any episode has ended (terminated or truncated)
#         if terminated or truncated:
#             print("Episode ended. Resetting environment...")
#             obs, info = env_layout.reset()
#             current_state_indices = [0] * len(q_tables)  # Reset state indices

#         # Complete task logic (simulate completion)
#         if random.random() < 0.1:  
#             update_task_buffer(task_buffer)

#         # Render the environment
#         env_layout.render()
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Exiting...")

# # Close the environment
# env_layout.close()



# import gymnasium as gym
# import rware
# import time
# import random
# import numpy as np

# # Define the custom layout
# layout = """
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# g..........xxxxxxxxx.
# g..........xxxxxxxxx.
# .....................
# ..xxxxxxxx.xxxxxxxxx.
# ..xxxxxxxx.xxxxxxxxx.
# .....................
# """

# # Create the environment with the custom layout
# env = gym.make("rware:rware-large-2ag-v2", layout=layout)

# # Reset the environment
# obs, info = env.reset()
# print("Environment reset!")

# # List of all shelves (initialize based on your layout)
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to your layout

# # Initialize task buffer with all shelves as potential tasks
# task_buffer = shelves.copy()

# # Q-learning parameters
# alpha = 0.1  # Learning rate
# gamma = 0.9  # Discount factor
# epsilon = 0.1  # Exploration rate
# num_episodes = 1000  # Number of episodes for training

# # Number of actions per agent (multi-agent environment)
# num_actions_per_agent = [space.n for space in env.action_space]
# print("Number of actions per agent:", num_actions_per_agent)

# # Initialize Q-tables for each agent based on state-action pairs
# q_tables = [np.zeros((len(shelves), num_actions)) for num_actions in num_actions_per_agent]

# # Helper function: Get nearest requested shelf for an agent
# def get_nearest_shelf(agent_pos, task_buffer):
#     distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
#     nearest_shelf_index = np.argmin(distances)
#     return task_buffer[nearest_shelf_index], nearest_shelf_index

# # Helper function: Choose an action using epsilon-greedy strategy
# def choose_action(agent_index, state_index):
#     if random.random() < epsilon:
#         return random.randint(0, num_actions_per_agent[agent_index] - 1)  # Explore
#     else:
#         return np.argmax(q_tables[agent_index][state_index])  # Exploit

# # Helper function: Update task buffer after completing a task
# def update_task_buffer(task_buffer, completed_task):
#     if completed_task in task_buffer:
#         task_buffer.remove(completed_task)
#         print(f"Completed task at {completed_task}")
#     if not task_buffer:  # If all tasks are done, refill with new tasks (optional)
#         task_buffer.extend(shelves)

# # Training loop
# for episode in range(num_episodes):
#     obs, info = env.reset()
#     print(f"Episode {episode + 1}/{num_episodes} started.")
    
#     terminated = False
#     truncated = False
    
#     while not (terminated or truncated):
#         actions = []
#         current_state_indices = []
        
#         # Determine actions for each agent
#         for agent_idx in range(len(q_tables)):
#             # Access agent position from observations instead of info dictionary if necessary
#             agent_pos = obs[agent_idx][:2]  # Assuming position is part of observation
            
#             if task_buffer:  # If there are tasks in the buffer
#                 nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
#                 current_state_indices.append(shelf_index)  # Use shelf index as state index
                
#                 # Choose action using epsilon-greedy strategy
#                 action = choose_action(agent_idx, shelf_index)
#                 actions.append(action)
#             else:
#                 actions.append(0)  # No-op if no tasks available
        
#         # Step through the environment with multi-agent actions
#         obs, rewards, terminated, truncated, info = env.step(actions)
        
#         # Update Q-tables based on rewards received and next state values
#         next_state_indices = []
#         for agent_idx in range(len(q_tables)):
#             if task_buffer:  # If there are tasks in the buffer
#                 agent_pos = obs[agent_idx][:2]  # Get new position from observations
#                 nearest_shelf, next_shelf_index = get_nearest_shelf(agent_pos, task_buffer)
#                 next_state_indices.append(next_shelf_index)
                
#                 best_next_action = np.argmax(q_tables[agent_idx][next_shelf_index])
#                 q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]] += alpha * (
#                     rewards[agent_idx] + gamma * q_tables[agent_idx][next_shelf_index][best_next_action] -
#                     q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]]
#                 )
        
#         # Check if any tasks were completed and update the task buffer accordingly
#         for agent_idx in range(len(q_tables)):
#             agent_pos = obs[agent_idx][:2]  # Get new position from observations
#             if tuple(agent_pos) in task_buffer:  # If an agent reaches a requested shelf
#                 update_task_buffer(task_buffer, tuple(agent_pos))
        
#         # Render the environment (optional)
#         env.render()
#         time.sleep(0.1)

#     print(f"Episode {episode + 1} ended.")

# print("Training completed!")
# env.close()



import gymnasium as gym
import rware
import time
import random
import numpy as np

# Define the custom layout
layout = """
.....................
..xxxxxxxx.xxxxxxxxx.
..xxxxxxxx.xxxxxxxxx.
.....................
g..........xxxxxxxxx.
g..........xxxxxxxxx.
.....................
..xxxxxxxx.xxxxxxxxx.
..xxxxxxxx.xxxxxxxxx.
.....................
"""

# Create the environment with the custom layout
env = gym.make("rware:rware-large-2ag-v2", layout=layout)

# Reset the environment
obs, info = env.reset()
print("Environment reset!")

# List of all shelves 
shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to your layout

# Initialize task buffer with all shelves as potential tasks
task_buffer = shelves.copy()

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Initial exploration rate
epsilon_decay = 0.99  # Decay rate for epsilon after each episode
min_epsilon = 0.1  # Minimum value for epsilon
num_episodes = 500  # Reduced number of episodes for faster training

# Number of actions per agent (multi-agent environment)
num_actions_per_agent = [space.n for space in env.action_space]
print("Number of actions per agent:", num_actions_per_agent)

# Initialize Q-tables for each agent based on state-action pairs
q_tables = [np.zeros((len(shelves), num_actions)) for num_actions in num_actions_per_agent]

# Helper function: Get nearest requested shelf for an agent
def get_nearest_shelf(agent_pos, task_buffer):
    distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
    nearest_shelf_index = np.argmin(distances)
    return task_buffer[nearest_shelf_index], nearest_shelf_index

# Helper function: Choose an action using epsilon-greedy strategy
def choose_action(agent_index, state_index):
    if random.random() < epsilon:
        return random.randint(0, num_actions_per_agent[agent_index] - 1)  # Explore
    else:
        return np.argmax(q_tables[agent_index][state_index])  # Exploit

# Helper function: Update task buffer after completing a task
def update_task_buffer(task_buffer, completed_task):
    if completed_task in task_buffer:
        task_buffer.remove(completed_task)
        print(f"Completed task at {completed_task}")
    if not task_buffer:  # If all tasks are done, refill with new tasks (optional)
        task_buffer.extend(shelves)

# Training loop
for episode in range(num_episodes):
    obs, info = env.reset()
    print(f"Episode {episode + 1}/{num_episodes} started.")
    
    terminated = False
    truncated = False
    
    step_count = 0
    
    while not (terminated or truncated):
        actions = []
        current_state_indices = []
        
        # Determine actions for each agent
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  # Assuming position is part of observation
            
            if task_buffer:  # If there are tasks in the buffer
                nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                current_state_indices.append(shelf_index)  # Use shelf index as state index
                
                # Choose action using epsilon-greedy strategy
                action = choose_action(agent_idx, shelf_index)
                actions.append(action)
            else:
                actions.append(0)  # No-op if no tasks available
        
        # Step through the environment with multi-agent actions
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Update Q-tables based on rewards received and next state values
        next_state_indices = []
        for agent_idx in range(len(q_tables)):
            if task_buffer:  # If there are tasks in the buffer
                agent_pos = obs[agent_idx][:2]  # Get new position from observations
                nearest_shelf, next_shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                next_state_indices.append(next_shelf_index)
                
                best_next_action = np.argmax(q_tables[agent_idx][next_shelf_index])
                q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]] += alpha * (
                    rewards[agent_idx] + gamma * q_tables[agent_idx][next_shelf_index][best_next_action] -
                    q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]]
                )
        
        # Check if any tasks were completed and update the task buffer accordingly
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  # Get new position from observations
            if tuple(agent_pos) in task_buffer:  # If an agent reaches a requested shelf
                update_task_buffer(task_buffer, tuple(agent_pos))
        
        step_count += 1
        
        # Render occasionally (e.g., every 100 steps or at the end of an episode)
        if step_count % 100 == 0:
            env.render()
    
    print(f"Episode {episode + 1} ended.")
    
    # Decay epsilon after each episode to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training completed!")

# Testing phase
num_test_episodes = 100  # Number of episodes for testing
test_rewards = []  # To store rewards for each test episode

for episode in range(num_test_episodes):
    obs, info = env.reset()
    print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
    terminated = False
    truncated = False
    
    total_reward = 0  # To accumulate rewards for this episode
    
    while not (terminated or truncated):
        actions = []
        
        # Determine actions for each agent based on max Q-value
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  # Assuming position is part of observation
            
            if task_buffer:  # If there are tasks in the buffer
                nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                
                # Choose action using max Q-value (exploit)
                action = np.argmax(q_tables[agent_idx][shelf_index])
                actions.append(action)
            else:
                actions.append(0)  # No-op if no tasks available
        
        # Step through the environment with multi-agent actions
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        total_reward += sum(rewards)  # Sum rewards from all agents
        
        # Check if any tasks were completed and update the task buffer accordingly
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  # Get new position from observations
            if tuple(agent_pos) in task_buffer:  # If an agent reaches a requested shelf
                update_task_buffer(task_buffer, tuple(agent_pos))
    
    test_rewards.append(total_reward)  # Store total reward for this episode
    print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

print("Testing completed!")
print("Average reward over test episodes:", np.mean(test_rewards))
env.close()
