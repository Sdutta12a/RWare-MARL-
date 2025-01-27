# # import gymnasium as gym
# # import rware
# # import random
# # import numpy as np

# # # Define the custom layout
# # layout = """
# # .....................
# # ..xxxxxxxx.xxxxxxxxx.
# # ..xxxxxxxx.xxxxxxxxx.
# # .....................
# # g..........xxxxxxxxx.
# # g..........xxxxxxxxx.
# # .....................
# # ..xxxxxxxx.xxxxxxxxx.
# # ..xxxxxxxx.xxxxxxxxx.
# # .....................
# # """

# # # Create the environment with the custom layout
# # env = gym.make("rware:rware-large-1ag-v2", layout=layout)  # Ensure this is a single-agent version

# # # Reset the environment
# # obs, info = env.reset()
# # print("Environment reset!")

# # # Dynamically identify all shelf positions from the layout
# # shelves = [(i, j) for i in range(len(layout.splitlines())) 
# #            for j in range(len(layout.splitlines()[0])) 
# #            if layout.splitlines()[i][j] == 'x']

# # # Initialize task buffer with all shelves as potential tasks
# # task_buffer = shelves.copy()

# # # Q-learning parameters
# # alpha = 0.1  # Learning rate
# # gamma = 0.9  # Discount factor
# # epsilon = 0.5  # Initial exploration rate
# # epsilon_decay = 0.99  # Decay rate for epsilon after each episode
# # min_epsilon = 0.1  # Minimum value for epsilon
# # num_episodes = 500  # Number of episodes for training

# # # Number of actions for the single agent (accessing the first element of the tuple)
# # num_actions = env.action_space[0].n  
# # print("Number of actions for the agent:", num_actions)

# # # Initialize Q-table for the single agent based on state-action pairs
# # q_table = np.zeros((len(shelves), num_actions))

# # # Helper function: Get nearest requested shelf for the agent
# # def get_nearest_shelf(agent_pos, task_buffer):
# #     distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
# #     nearest_shelf_index = np.argmin(distances)
# #     return task_buffer[nearest_shelf_index], nearest_shelf_index

# # # Helper function: Choose an action using epsilon-greedy strategy
# # def choose_action(state_index):
# #     if random.random() < epsilon:
# #         return random.randint(0, num_actions - 1)  # Explore
# #     else:
# #         return np.argmax(q_table[state_index])  # Exploit

# # # Helper function: Update task buffer after completing a task
# # def update_task_buffer(task_buffer, completed_task):
# #     if completed_task in task_buffer:
# #         task_buffer.remove(completed_task)
# #         print(f"Completed task at {completed_task}")
# #     if not task_buffer:  # If all tasks are done, refill with new tasks (optional)
# #         task_buffer.extend(shelves)

# # # Training loop
# # for episode in range(num_episodes):
# #     obs, info = env.reset()
# #     print(f"Episode {episode + 1}/{num_episodes} started.")
    
# #     terminated = False
# #     truncated = False
    
# #     step_count = 0
    
# #     while not (terminated or truncated):
# #         agent_pos = obs[0][:2]  # Assuming position is part of observation
        
# #         if task_buffer:  # If there are tasks in the buffer
# #             nearest_shelf, current_state_index = get_nearest_shelf(agent_pos, task_buffer)
            
# #             # Choose action using epsilon-greedy strategy
# #             action = choose_action(current_state_index)
# #         else:
# #             action = 0  # No-op if no tasks available
        
# #         # Step through the environment with the agent's action (wrap action in a list)
# #         obs, reward, terminated, truncated, info = env.step([action]) 
        
# #         next_agent_pos = obs[0][:2]  # Get new position from observations
        
# #         if task_buffer:  # If there are tasks in the buffer after taking action
# #             nearest_shelf, next_shelf_index = get_nearest_shelf(next_agent_pos, task_buffer)
            
# #             best_next_action = np.argmax(q_table[next_shelf_index])
# #             q_table[current_state_index][action] += alpha * (
# #                 reward[0] + gamma * q_table[next_shelf_index][best_next_action] -
# #                 q_table[current_state_index][action]
# #             )
        
# #         # Check if the agent reached a requested shelf and update the task buffer accordingly
# #         if tuple(agent_pos) in task_buffer:  
# #             update_task_buffer(task_buffer, tuple(agent_pos))
        
# #         step_count += 1
        
# #         # Render occasionally (e.g., every 100 steps or at the end of an episode)
# #         if step_count % 100 == 0:
# #             env.render()
    
# #     print(f"Episode {episode + 1} ended.")
    
# #     # Decay epsilon after each episode to reduce exploration over time
# #     epsilon = max(min_epsilon, epsilon * epsilon_decay)

# # print("Training completed!")

# # # Testing phase with dynamic shelves identification again (to avoid stale data)
# # task_buffer = shelves.copy()  # Resetting task buffer for testing phase

# # num_test_episodes = 200  # Number of episodes for testing
# # test_rewards = []  # To store rewards for each test episode

# # for episode in range(num_test_episodes):
# #     obs, info = env.reset()
# #     print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
# #     terminated = False
# #     truncated = False
    
# #     total_reward = 0  
    
# #     while not (terminated or truncated):
# #         agent_pos = obs[0][:2]  
        
# #         if task_buffer:  
# #             nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
            
# #             action = np.argmax(q_table[shelf_index])  
# #         else:
# #             action = 0  
        
# #         obs, rewards, terminated, truncated, info = env.step([action]) 
        
# #         total_reward += rewards[0]  # Accessing first element of rewards list
        
# #         if tuple(agent_pos) in task_buffer:  
# #             update_task_buffer(task_buffer, tuple(agent_pos))
    
# #     test_rewards.append(total_reward)  
# #     print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

# # print("Testing completed!")
# # print("Average reward over test episodes:", np.mean(test_rewards))
# # env.close()





# import gymnasium as gym
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
# env = gym.make("rware:rware-large-1ag-v2", layout=layout)  # Ensure this is a single-agent version

# # Reset the environment
# obs, info = env.reset()
# print("Environment reset!")

# # Dynamically identify all shelf positions from the layout
# shelves = [(i, j) for i in range(len(layout.splitlines())) 
#            for j in range(len(layout.splitlines()[0])) 
#            if layout.splitlines()[i][j] == 'x']

# # Initialize task buffer with all shelves as potential tasks
# task_buffer = shelves.copy()

# # Q-learning parameters
# alpha = 0.1  # Learning rate
# gamma = 0.9  # Discount factor
# epsilon = 0.5  # Initial exploration rate
# epsilon_decay = 0.99  # Decay rate for epsilon after each episode
# min_epsilon = 0.1  # Minimum value for epsilon
# num_episodes = 500  # Number of episodes for training

# # Number of actions for the single agent (accessing the first element of the tuple)
# num_actions = env.action_space[0].n  
# print("Number of actions for the agent:", num_actions)

# # Initialize Q-table for the single agent based on state-action pairs
# q_table = np.zeros((len(shelves), num_actions))

# # Helper function: Get nearest requested shelf for the agent
# def get_nearest_shelf(agent_pos, task_buffer):
#     distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
#     nearest_shelf_index = np.argmin(distances)
#     return task_buffer[nearest_shelf_index], nearest_shelf_index

# # Helper function: Choose an action using epsilon-greedy strategy
# def choose_action(state_index):
#     if random.random() < epsilon:
#         return random.randint(0, num_actions - 1)  # Explore
#     else:
#         return np.argmax(q_table[state_index])  # Exploit

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
    
#     step_count = 0
    
#     while not (terminated or truncated):
#         agent_pos = obs[0][:2]  # Assuming position is part of observation
        
#         if task_buffer:  # If there are tasks in the buffer
#             nearest_shelf, current_state_index = get_nearest_shelf(agent_pos, task_buffer)
            
#             # Choose action using epsilon-greedy strategy
#             action = choose_action(current_state_index)
#         else:
#             action = 0  # No-op if no tasks available
        
#         # Step through the environment with the agent's action (wrap action in a list)
#         obs, reward, terminated, truncated, info = env.step([action]) 
        
#         next_agent_pos = obs[0][:2]  # Get new position from observations
        
#         if task_buffer:  # If there are tasks in the buffer after taking action
#             nearest_shelf, next_shelf_index = get_nearest_shelf(next_agent_pos, task_buffer)
            
#             best_next_action = np.argmax(q_table[next_shelf_index])
#             q_table[current_state_index][action] += alpha * (
#                 reward[0] + gamma * q_table[next_shelf_index][best_next_action] -
#                 q_table[current_state_index][action]
#             )
        
#         # Check if the agent reached a requested shelf and update the task buffer accordingly
#         if tuple(agent_pos) in task_buffer:  
#             update_task_buffer(task_buffer, tuple(agent_pos))
        
#         step_count += 1
        
#         # Render occasionally (e.g., every 100 steps or at the end of an episode)
#         if step_count % 100 == 0:
#             env.render()
    
#     print(f"Episode {episode + 1} ended.")
    
#     # Decay epsilon after each episode to reduce exploration over time
#     epsilon = max(min_epsilon, epsilon * epsilon_decay)

# print("Training completed!")

# # Testing phase with dynamic shelves identification again (to avoid stale data)
# task_buffer = shelves.copy()  # Resetting task buffer for testing phase

# num_test_episodes = 200  # Number of episodes for testing
# test_rewards = []  # To store rewards for each test episode

# for episode in range(num_test_episodes):
#     obs, info = env.reset()
#     print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
#     terminated = False
#     truncated = False
    
#     total_reward = 0  
    
#     while not (terminated or truncated):
#         agent_pos = obs[0][:2]  
        
#         if task_buffer:  
#             nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
            
#             action = np.argmax(q_table[shelf_index])  
#         else:
#             action = random.randint(0, num_actions - 1)  # Random action or no-op
        
#         obs, rewards, terminated, truncated, info = env.step([action]) 
        
#         total_reward += rewards[0]  # Accessing first element of rewards list
        
#         if tuple(agent_pos) in task_buffer:  
#             update_task_buffer(task_buffer, tuple(agent_pos))
    
#     test_rewards.append(total_reward)  
#     print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

# print("Testing completed!")
# print("Average reward over test episodes:", np.mean(test_rewards))
# env.close()


import gymnasium as gym
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
env = gym.make("rware:rware-large-1ag-v2", layout=layout)  # Ensure this is a single-agent version

# Reset the environment
obs, info = env.reset()
print("Environment reset!")

# Dynamically identify all shelf positions from the layout
shelves = [(i, j) for i in range(len(layout.splitlines())) 
           for j in range(len(layout.splitlines()[0])) 
           if layout.splitlines()[i][j] == 'x']

# Initialize task buffer with all shelves as potential tasks
task_buffer = shelves.copy()

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Initial exploration rate
epsilon_decay = 0.99  # Decay rate for epsilon after each episode
min_epsilon = 0.1  # Minimum value for epsilon
num_episodes = 500  # Number of episodes for training

# Number of actions for the single agent (accessing the first element of the tuple)
num_actions = env.action_space[0].n  
print("Number of actions for the agent:", num_actions)

# Initialize Q-table for the single agent based on state-action pairs
q_table = np.zeros((len(shelves), num_actions))

# Helper function: Get nearest requested shelf for the agent
def get_nearest_shelf(agent_pos, task_buffer):
    distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
    nearest_shelf_index = np.argmin(distances)
    return task_buffer[nearest_shelf_index], nearest_shelf_index

# Helper function: Choose an action using epsilon-greedy strategy
def choose_action(state_index):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # Explore
    else:
        return np.argmax(q_table[state_index])  # Exploit

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
        agent_pos = obs[0][:2]  # Assuming position is part of observation
        
        if task_buffer:  # If there are tasks in the buffer
            nearest_shelf, current_state_index = get_nearest_shelf(agent_pos, task_buffer)
            
            # Choose action using epsilon-greedy strategy
            action = choose_action(current_state_index)
        else:
            action = 0  # No-op if no tasks available
        
        # Step through the environment with the agent's action (wrap action in a list)
        obs, reward, terminated, truncated, info = env.step([action]) 
        
        next_agent_pos = obs[0][:2]  # Get new position from observations
        
        if task_buffer:  # If there are tasks in the buffer after taking action
            nearest_shelf, next_shelf_index = get_nearest_shelf(next_agent_pos, task_buffer)
            
            best_next_action = np.argmax(q_table[next_shelf_index])
            q_table[current_state_index][action] += alpha * (
                reward[0] + gamma * q_table[next_shelf_index][best_next_action] -
                q_table[current_state_index][action]
            )
        
        # Check if the agent reached a requested shelf and update the task buffer accordingly
        if tuple(agent_pos) in task_buffer:  
            update_task_buffer(task_buffer, tuple(agent_pos))
        
        step_count += 1
        
        # Render occasionally (e.g., every 100 steps or at the end of an episode)
        if step_count % 100 == 0:
            env.render()
    
    print(f"Episode {episode + 1} ended.")
    
    # Decay epsilon after each episode to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training completed!")

# Testing phase with dynamic shelves identification again (to avoid stale data)
task_buffer = shelves.copy()  # Resetting task buffer for testing phase

num_test_episodes = 200  # Number of episodes for testing
test_rewards = []  # To store rewards for each test episode

for episode in range(num_test_episodes):
    obs, info = env.reset()
    print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
    terminated = False
    truncated = False
    
    total_reward = 0  
    
    while not (terminated or truncated):
        agent_pos = obs[0][:2]  
        
        if task_buffer:  
            nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
            
            action = np.argmax(q_table[shelf_index])  
        else:
            action = random.randint(0, num_actions - 1)  # Random action or no-op
        
        obs, rewards, terminated, truncated, info = env.step([action]) 
        
        total_reward += rewards[0]  # Accessing first element of rewards list
        
        if tuple(agent_pos) in task_buffer:  
            update_task_buffer(task_buffer, tuple(agent_pos))
    
    test_rewards.append(total_reward)  
    print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

print("Testing completed!")
print("Average reward over test episodes:", np.mean(test_rewards))
env.close()
