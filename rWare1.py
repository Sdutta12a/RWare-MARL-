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

# # List of all shelves 
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions; update according to your layout

# # Initialize task buffer with all shelves as potential tasks
# task_buffer = shelves.copy()

# # Q-learning parameters
# alpha = 0.1  # Learning rate
# gamma = 0.9  # Discount factor
# epsilon = 0.5  # Initial exploration rate
# epsilon_decay = 0.99  # Decay rate for epsilon after each episode
# min_epsilon = 0.1  # Minimum value for epsilon
# num_episodes = 500  # Reduced number of episodes for faster training

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
    
#     step_count = 0
    
#     while not (terminated or truncated):
#         actions = []
#         current_state_indices = []
        
#         # Determine actions for each agent
#         for agent_idx in range(len(q_tables)):
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
        
#         step_count += 1
        
#         # Render occasionally (e.g., every 100 steps or at the end of an episode)
#         if step_count % 100 == 0:
#             env.render()
    
#     print(f"Episode {episode + 1} ended.")
    
#     # Decay epsilon after each episode to reduce exploration over time
#     epsilon = max(min_epsilon, epsilon * epsilon_decay)

# print("Training completed!")

# # Testing phase
# num_test_episodes = 100  # Number of episodes for testing
# test_rewards = []  # To store rewards for each test episode

# for episode in range(num_test_episodes):
#     obs, info = env.reset()
#     print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
#     terminated = False
#     truncated = False
    
#     total_reward = 0  # To accumulate rewards for this episode
    
#     while not (terminated or truncated):
#         actions = []
        
#         # Determine actions for each agent based on max Q-value
#         for agent_idx in range(len(q_tables)):
#             agent_pos = obs[agent_idx][:2]  # Assuming position is part of observation
            
#             if task_buffer:  # If there are tasks in the buffer
#                 nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                
#                 # Choose action using max Q-value (exploit)
#                 action = np.argmax(q_tables[agent_idx][shelf_index])
#                 actions.append(action)
#             else:
#                 actions.append(0)  # No-op if no tasks available
        
#         # Step through the environment with multi-agent actions
#         obs, rewards, terminated, truncated, info = env.step(actions)
        
#         # Accumulate rewards
#         total_reward += sum(rewards)  # Sum rewards from all agents
        
#         # Check if any tasks were completed and update the task buffer accordingly
#         for agent_idx in range(len(q_tables)):
#             agent_pos = obs[agent_idx][:2]  # Get new position from observations
#             if tuple(agent_pos) in task_buffer:  # If an agent reaches a requested shelf
#                 update_task_buffer(task_buffer, tuple(agent_pos))
    
#     test_rewards.append(total_reward)  # Store total reward for this episode
#     print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

# print("Testing completed!")
# print("Average reward over test episodes:", np.mean(test_rewards))
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
alpha = 0.2  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon after each episode
min_epsilon = 0.1  # Minimum value for epsilon
num_episodes = 100  # Reduced number of episodes for faster training

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

# Helper function: Compute reward based on agent's action and completed task
def compute_reward(agent_action_index, completed_task):
    if completed_task:
        return 10  # High reward for completing a task at a requested shelf
    if agent_action_index < len(shelves) and shelves[agent_action_index] in task_buffer:
        return -1  # Small penalty for ignoring a requested shelf action
    return 0  # Neutral reward for other actions

# Helper function: Choose an action using epsilon-greedy strategy
def choose_action(agent_index, state_index):
    if random.random() < epsilon:
        # Initialize action probabilities
        action_probabilities = np.zeros(num_actions_per_agent[agent_index])
        
        # Base probability for all actions
        base_probability = 0.1
        action_probabilities[:] = base_probability
        
        # Increase probability for actions related to green shelves (placeholder logic)
        for action in range(num_actions_per_agent[agent_index]):
            if is_green_shelf_action(action, state_index):  
                action_probabilities[action] += 0.9 / count_green_shelf_actions(state_index)
        
        # Normalize probabilities to sum to 1
        action_probabilities /= np.sum(action_probabilities)
        
        return np.random.choice(num_actions_per_agent[agent_index], p=action_probabilities)
    else:
        return np.argmax(q_tables[agent_index][state_index])  

def is_green_shelf_action(action, state_index):
    # Implement logic to check if the action corresponds to interacting with a green shelf
    return True  

def count_green_shelf_actions(state_index):
    return len([action for action in range(num_actions_per_agent[0]) if is_green_shelf_action(action, state_index)])

# Helper function: Update task buffer after completing a task
def update_task_buffer(task_buffer, completed_task):
    if completed_task in task_buffer:
        task_buffer.remove(completed_task)
        print(f"Completed task at {completed_task}")
    if not task_buffer:  
        print("All tasks completed! Refilling task buffer.")
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
            agent_pos = obs[agent_idx][:2]  
            
            if task_buffer:  
                nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                current_state_indices.append(shelf_index)  
                
                action = choose_action(agent_idx, shelf_index)
                actions.append(action)
            else:
                actions.append(0)  
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        next_state_indices = []
        for agent_idx in range(len(q_tables)):
            if task_buffer:  
                agent_pos = obs[agent_idx][:2]  
                nearest_shelf, next_shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                next_state_indices.append(next_shelf_index)
                
                completed_task = tuple(agent_pos) in task_buffer
                
                q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]] += alpha * (
                    compute_reward(actions[agent_idx], completed_task) + 
                    gamma * np.max(q_tables[agent_idx][next_shelf_index]) - 
                    q_tables[agent_idx][current_state_indices[agent_idx]][actions[agent_idx]]
                )
        
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  
            if tuple(agent_pos) in task_buffer:  
                update_task_buffer(task_buffer, tuple(agent_pos))
        
        step_count += 1
        
        if step_count % 100 == 0:
            env.render()
    
    print(f"Episode {episode + 1} ended.")
    
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training completed!")

# Testing phase
num_test_episodes = 100  
test_rewards = []  

for episode in range(num_test_episodes):
    obs, info = env.reset()
    print(f"Test Episode {episode + 1}/{num_test_episodes} started.")
    
    terminated = False
    truncated = False
    
    total_reward = 0  
    
    while not (terminated or truncated):
        actions = []
        
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  
            
            if task_buffer:  
                nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                
                action = np.argmax(q_tables[agent_idx][shelf_index])
                actions.append(action)
            else:
                actions.append(0)  
        
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Ensure rewards are numeric before summing them up.
        total_reward += sum(rewards) 
        
        for agent_idx in range(len(q_tables)):
            agent_pos = obs[agent_idx][:2]  
            if tuple(agent_pos) in task_buffer:  
                update_task_buffer(task_buffer, tuple(agent_pos))
    
    test_rewards.append(total_reward)  
    print(f"Test Episode {episode + 1} ended with total reward: {total_reward}")

print("Testing completed!")
print("Average reward over test episodes:", np.mean(test_rewards))
env.close()
