import gymnasium as gym
import rware
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Neural Network for DQN
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        # Initialize networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Copy initial weights to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(q_values, q_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Define custom layout
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

# List of all shelves (update according to your layout)
shelves = [(2, 6), (3, 7), (6, 11)]
task_buffer = shelves.copy()

# Initialize DQN agents for each agent
num_agents = 2
state_size = 4  # Example: agent position (x, y) + nearest shelf (x, y)
action_size = env.action_space[0].n
agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]

# Helper function: Get nearest requested shelf for an agent
def get_nearest_shelf(agent_pos, task_buffer):
    distances = [np.linalg.norm(np.array(agent_pos) - np.array(shelf)) for shelf in task_buffer]
    nearest_shelf_index = np.argmin(distances)
    return task_buffer[nearest_shelf_index], nearest_shelf_index

# Training loop
num_episodes = 500
target_update_freq = 50
batch_size = 64

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        actions = []
        experiences = []

        # Select actions for each agent
        for i, agent in enumerate(agents):
            agent_pos = obs[i][:2]
            if task_buffer:
                nearest_shelf, shelf_index = get_nearest_shelf(agent_pos, task_buffer)
                state = np.concatenate((agent_pos, nearest_shelf))
                action = agent.select_action(state)
                actions.append(action)
            else:
                actions.append(0)  # No-op if no tasks available

        # Perform actions
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # Store experiences and update task buffer
        for i, agent in enumerate(agents):
            next_agent_pos = next_obs[i][:2]
            nearest_shelf, next_shelf_index = get_nearest_shelf(next_agent_pos, task_buffer)
            next_state = np.concatenate((next_agent_pos, nearest_shelf))
            experience = (state, actions[i], rewards[i], next_state, terminated or truncated)
            agent.replay_buffer.add(experience)

            if tuple(next_agent_pos) in task_buffer:
                task_buffer.remove(tuple(next_agent_pos))

            # Train the agent
            agent.train(batch_size)

        total_reward += sum(rewards)

    # Update target networks periodically
    if episode % target_update_freq == 0:
        for agent in agents:
            agent.update_target_network()

    print(f"Episode {episode + 1} ended with total reward: {total_reward}")

# Testing phase can be added similarly
env.close()
