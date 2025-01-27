import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Custom Grid Environment Class
class CustomGridEnv(gym.Env):
    def __init__(self):
        super(CustomGridEnv, self).__init__()
        
        # Define grid size and positions
        self.grid_size = 10
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        
        # Define action space (4 possible actions: Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (grid coordinates)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size)))
        
        # Obstacles are defined as (x, y, width, height)
        self.obstacles = [
            (2, 2, 3, 1),  
            (5, 5, 1, 3),  
            (7, 1, 2, 2),   
            (7, 9, 1, 2),
            (0, 4, 2, 1),
            (8, 7, 1, 1),
        ]
        
        self.state = None
        self.done = False

    def reset(self):
        self.state = self.start_position
        self.done = False
        return self.state

    def step(self, action):
        row, col = self.state
        d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]  # Up(0), Down(1), Left(2), Right(3)
        
        next_row = max(0, min(self.grid_size - 1, row + d_row))
        next_col = max(0, min(self.grid_size - 1, col + d_col))
        
        next_state = (next_row, next_col)

        if self.is_collision(next_state):
            reward = -100
            next_state = self.state  # Stay in place on collision
        elif next_state == self.goal_position:
            reward = 100
            self.done = True
        else:
            reward = -1
        
        self.state = next_state
        
        return next_state, reward, self.done, {}

    def is_collision(self, next_state):
        for (x,y,width,height) in self.obstacles:
            if x <= next_state[1] < x + width and y <= next_state[0] < y + height:
                return True
        return False

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        for (x,y,width,height) in self.obstacles:
            grid[y:y+height,x:x+width] = -1
        
        grid[self.start_position[0], self.start_position[1]] = -2   # Start position
        grid[self.goal_position[0], self.goal_position[1]] = -3    # Goal position
        
        print("\nGrid:")
        print(grid)

# Q-Learning Implementation with Custom Environment
def train_agent():
    # Create environment instance
    env = CustomGridEnv()

    # Initialize Q-table
    Q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Hyperparameters
    alpha = 0.1      
    gamma = 0.99     
    epsilon = 1.0    
    epsilon_min = 0.01
    epsilon_decay = 0.995
    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q_table[state[0], state[1]])  # Exploit
                
            next_state, reward, done, _ = env.step(action)

            # Update Q-value
            best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])
            
            Q_table[state[0], state[1], action] += alpha * (
                reward + gamma * Q_table[next_state[0], next_state[1], best_next_action] - Q_table[state[0], state[1], action]
            )

            state = next_state
            total_reward += reward
            
            if done:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  

        # Logging progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    return Q_table

# Test the trained agent and visualize the path
def test_agent(Q_table):
    env = CustomGridEnv()
    state = env.reset()
    path = [state]

    while state != env.goal_position:
        action = np.argmax(Q_table[state[0], state[1]])
        state, _, done, _ = env.step(action)
        path.append(state)

    print("Path taken by the agent:")
    print(path)

    # Visualization using Matplotlib Animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)
        
        ax.set_xticks(np.arange(0, env.grid_size + 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1))
        
        ax.grid(True)

        # Plot obstacles
        for (x,y,width,height) in env.obstacles:
            rect = plt.Rectangle((x,y), width,height,color='gray', alpha=0.5)
            ax.add_patch(rect)

        # Plot start and goal positions
        ax.plot(env.start_position[1], env.start_position[0], 'go', markersize=12)   # Start in green.
        ax.plot(env.goal_position[1], env.goal_position[0], 'ro', markersize=12)     # Goal in red.

        agent_position = path[frame]
        
        ax.plot(agent_position[1], agent_position[0], 'bs', markersize=8)     # Agent in blue.

    ani=animation.FuncAnimation(fig ,update ,frames=len(path), repeat=False)
    
    plt.show()

# Main execution flow
if __name__ == "__main__":
    Q_table = train_agent()
    test_agent(Q_table)
