# actions = {
#     0: (-1, 0),  # Up
#     1: (1, 0),   # Down
#     2: (0, -1),  # Left
#     3: (0, 1),   # Right
# }
# num_actions = len(actions)
# print(num_actions)

# import numpy as np
# grid_size = 10
# Q_table = np.zeros((grid_size, grid_size, num_actions))
# print(Q_table)

# import gymnasium as gym
# import rware
# import random
# import numpy as np

# # Define the layout
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

# # Create the environment
# env = gym.make("rware:rware-large-2ag-v2", layout=layout)
# obs, info = env.reset()

# # Define shelves
# shelves = [(2, 6), (3, 7), (6, 11)]  # Example shelf positions
# task_buffer = [random.choice(shelves)]  # Initialize with one random task
# print("Initial task buffer:", task_buffer)

# # Q-Learning parameters
# num_agents = len(env.action_space)  # Number of agents
# num_actions_per_agent = [space.n for space in env.action_space]  # Actions per agent
# num_states = 1000  # Example: Replace with appropriate state encoding
# q_tables = [np.zeros((num_states, num_actions)) for num_actions in num_actions_per_agent]  # Separate Q-table for each agent
# alpha = 0.1       # Learning rate
# gamma = 0.95      # Discount factor
# epsilon = 1.0     # Exploration rate
# epsilon_decay = 0.995
# min_epsilon = 0.01

# # Function to encode the state
# def encode_state(observation):
#     # Simplified state encoding: Replace with a proper encoding scheme
#     return hash(tuple(map(tuple, observation))) % num_states

# # Update Q-table for a single agent
# def update_q_table(q_table, state, action, reward, next_state, done):
#     best_next_action = np.argmax(q_table[next_state])
#     td_target = reward + (gamma * q_table[next_state][best_next_action] * (1 - done))
#     td_error = td_target - q_table[state][action]
#     q_table[state][action] += alpha * td_error

# # Main training loop
# import random
# import time
# import numpy as np

# # Environment setup
# GRID_ROWS = 10
# GRID_COLS = 10
# SHELVES = [(2, 6), (3, 7), (6, 5)]  # Shelf locations
# GOALS = [(9, 0), (9, 9)]  # Goal locations
# NUM_AGENTS = 2
# ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]

# # Initialize the grid
# def initialize_grid():
#     grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
#     for shelf in SHELVES:
#         grid[shelf] = 1  # Mark shelves
#     for goal in GOALS:
#         grid[goal] = 2  # Mark goals
#     return grid

# # Agent class
# class Agent:
#     def __init__(self, id, start_pos):
#         self.id = id
#         self.pos = start_pos
#         self.carrying_shelf = None

#     def move(self, action):
#         if action == "UP":
#             self.pos = (max(0, self.pos[0] - 1), self.pos[1])
#         elif action == "DOWN":
#             self.pos = (min(GRID_ROWS - 1, self.pos[0] + 1), self.pos[1])
#         elif action == "LEFT":
#             self.pos = (self.pos[0], max(0, self.pos[1] - 1))
#         elif action == "RIGHT":
#             self.pos = (self.pos[0], min(GRID_COLS - 1, self.pos[1] + 1))
#         elif action == "PICK":
#             if self.carrying_shelf is None and self.pos in SHELVES:
#                 self.carrying_shelf = self.pos
#                 SHELVES.remove(self.pos)
#         elif action == "DROP":
#             if self.carrying_shelf is not None and self.pos in GOALS:
#                 self.carrying_shelf = None

# # Render the grid
# def render_grid(grid, agents):
#     grid_display = grid.copy()
#     for agent in agents:
#         grid_display[agent.pos] = 3  # Mark agents
#     print(grid_display)
#     print()

# # Main simulation
# def main():
#     grid = initialize_grid()
#     agents = [Agent(i, (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))) for i in range(NUM_AGENTS)]
#     task_buffer = [random.choice(SHELVES)]  # Initial task

#     try:
#         while True:
#             # Render environment
#             print("Task Buffer:", task_buffer)
#             render_grid(grid, agents)

#             # Agents take actions
#             for agent in agents:
#                 if agent.carrying_shelf:
#                     # Move toward a goal
#                     goal = random.choice(GOALS)
#                     if agent.pos != goal:
#                         if agent.pos[0] < goal[0]:
#                             agent.move("DOWN")
#                         elif agent.pos[0] > goal[0]:
#                             agent.move("UP")
#                         elif agent.pos[1] < goal[1]:
#                             agent.move("RIGHT")
#                         elif agent.pos[1] > goal[1]:
#                             agent.move("LEFT")
#                     else:
#                         agent.move("DROP")
#                 else:
#                     # Move toward a task
#                     task = task_buffer[0]
#                     if agent.pos != task:
#                         if agent.pos[0] < task[0]:
#                             agent.move("DOWN")
#                         elif agent.pos[0] > task[0]:
#                             agent.move("UP")
#                         elif agent.pos[1] < task[1]:
#                             agent.move("RIGHT")
#                         elif agent.pos[1] > task[1]:
#                             agent.move("LEFT")
#                     else:
#                         agent.move("PICK")
#                         if task_buffer:
#                             task_buffer.pop(0)
#                             if SHELVES:
#                                 task_buffer.append(random.choice(SHELVES))

#             # Delay for visualization
#             time.sleep(0.5)

#     except KeyboardInterrupt:
#         print("Simulation ended.")

# # Run the simulation
# if __name__ == "__main__":
#     main()


# import random
# import time
# import numpy as np

# # Environment setup
# GRID_ROWS = 10
# GRID_COLS = 10
# SHELVES = [(2, 6), (3, 7), (6, 5)]  # Shelf locations
# GOALS = [(9, 0), (9, 9)]  # Goal locations
# NUM_AGENTS = 2
# ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]

# # Initialize the grid
# def initialize_grid():
#     grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
#     for shelf in SHELVES:
#         grid[shelf] = 1  # Mark shelves
#     for goal in GOALS:
#         grid[goal] = 2  # Mark goals
#     return grid

# # Agent class
# class Agent:
#     def __init__(self, id, start_pos):
#         self.id = id
#         self.pos = start_pos
#         self.carrying_shelf = None

#     def move(self, action):
#         if action == "UP":
#             self.pos = (max(0, self.pos[0] - 1), self.pos[1])
#         elif action == "DOWN":
#             self.pos = (min(GRID_ROWS - 1, self.pos[0] + 1), self.pos[1])
#         elif action == "LEFT":
#             self.pos = (self.pos[0], max(0, self.pos[1] - 1))
#         elif action == "RIGHT":
#             self.pos = (self.pos[0], min(GRID_COLS - 1, self.pos[1] + 1))
#         elif action == "PICK":
#             if self.carrying_shelf is None and self.pos in SHELVES:
#                 self.carrying_shelf = self.pos
#                 SHELVES.remove(self.pos)
#         elif action == "DROP":
#             if self.carrying_shelf is not None and self.pos in GOALS:
#                 self.carrying_shelf = None

# # Render the grid
# def render_grid(grid, agents):
#     grid_display = grid.copy()
#     for agent in agents:
#         grid_display[agent.pos] = 3  # Mark agents
#     print(grid_display)
#     print()

# # Main simulation
# def main():
#     grid = initialize_grid()
#     agents = [Agent(i, (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))) for i in range(NUM_AGENTS)]
#     task_buffer = [random.choice(SHELVES)]  # Initial task

#     try:
#         while True:
#             # Render environment
#             print("Task Buffer:", task_buffer)
#             render_grid(grid, agents)

#             # Agents take actions
#             for agent in agents:
#                 if agent.carrying_shelf:
#                     # Move toward a goal
#                     goal = random.choice(GOALS)
#                     if agent.pos != goal:
#                         if agent.pos[0] < goal[0]:
#                             agent.move("DOWN")
#                         elif agent.pos[0] > goal[0]:
#                             agent.move("UP")
#                         elif agent.pos[1] < goal[1]:
#                             agent.move("RIGHT")
#                         elif agent.pos[1] > goal[1]:
#                             agent.move("LEFT")
#                     else:
#                         agent.move("DROP")
#                 else:
#                     # Move toward a task
#                     task = task_buffer[0]
#                     if agent.pos != task:
#                         if agent.pos[0] < task[0]:
#                             agent.move("DOWN")
#                         elif agent.pos[0] > task[0]:
#                             agent.move("UP")
#                         elif agent.pos[1] < task[1]:
#                             agent.move("RIGHT")
#                         elif agent.pos[1] > task[1]:
#                             agent.move("LEFT")
#                     else:
#                         agent.move("PICK")
#                         if task_buffer:
#                             task_buffer.pop(0)
#                             if SHELVES:
#                                 task_buffer.append(random.choice(SHELVES))

#             # Delay for visualization
#             time.sleep(0.5)

#     except KeyboardInterrupt:
#         print("Simulation ended.")

# # Run the simulation
# if __name__ == "__main__":
#     main()



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

# # Initialize task buffer with all shelves
# task_buffer = shelves.copy()
# print("Initial task buffer:", task_buffer)

# # Q-learning parameters
# Q = {}  # Initialize Q-table
# gamma = 0.9  # Discount factor
# alpha = 0.1  # Learning rate
# epsilon = 0.1  # Exploration rate

# def update_task_buffer(task_buffer):
#     if task_buffer:
#         completed_task = task_buffer.pop(0)
#         print(f"Completed task at {completed_task}")

# def get_state(obs):
#     # Convert observation to state representation (customize as needed)
#     return tuple(obs['agent_pos'])  # Example: using agent's position as state

# def choose_action(state):
#     if random.random() < epsilon:
#         return env_layout.action_space.sample()  # Explore
#     else:
#         return np.argmax(Q.get(state, np.zeros(env_layout.action_space.n)))  # Exploit

# try:
#     while True:
#         state = get_state(obs)
        
#         # Update environment visualization
#         print(f"Current state: {state}, Task buffer: {task_buffer}")

#         # Choose action based on current state
#         action = choose_action(state)

#         # Step through the environment
#         obs, reward, terminated, truncated, info = env_layout.step(action)

#         # Update Q-table using the Q-learning formula
#         next_state = get_state(obs)
#         old_value = Q.get(state, np.zeros(env_layout.action_space.n))[action]
#         next_max = np.max(Q.get(next_state, np.zeros(env_layout.action_space.n)))

#         # Update rule for Q-learning
#         new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        
#         if state not in Q:
#             Q[state] = np.zeros(env_layout.action_space.n)
#         Q[state][action] = new_value

#         # Check if the episode has ended (terminated or truncated)
#         if terminated or truncated:
#             print("Episode ended. Resetting environment...")
#             obs, info = env_layout.reset()

#         # Complete task if agent reaches it (implement your task completion logic here)
#         if random.random() < 0.1:  # Simulate task completion condition
#             update_task_buffer(task_buffer)

#         # Render the environment
#         env_layout.render()
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Exiting...")

# # Close the environment
# env_layout.close()




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
            # Access agent position from observations instead of info dictionary if necessary
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
env.close()




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
# epsilon = 0.5  # Initial exploration rate
# epsilon_decay = 0.99  # Decay rate for epsilon after each episode
# min_epsilon = 0.1  # Minimum value for epsilon
# num_episodes = 200  # Reduced number of episodes for faster training

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
        
#         step_count += 1
        
#         # Render occasionally (e.g., every step or at the end of an episode)
#         if step_count % 50 == 0 or terminated or truncated:
#             env.render()
        
#         time.sleep(0.05)  # Slight delay to allow observation

#     print(f"Episode {episode + 1} ended.")
    
#     # Decay epsilon after each episode to reduce exploration over time
#     epsilon = max(min_epsilon, epsilon * epsilon_decay)

# print("Training completed!")
# env.close()

