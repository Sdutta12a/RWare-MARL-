# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-11, 11)
# ax.set_ylim(-11, 11)

# # Add workspace constraints (rectangles)
# # rectangles = [
# #     [-3, 6, 6, 2],  # Top center rectangle
# #     [-4.5, -6, 1, 12],  # Left vertical rectangle
# #     [2.5, -6, 1, 12],  # Right vertical rectangle
# #     [4, -4, 2, 1],  # Right horizontal rectangles
# #     [4, -6, 2, 1],
# #     [4, -8, 2, 1],
# #     [6, -4, 2, 1],  # Left horizontal rectangles
# #     [6, -6, 2, 1],
# #     [6, -8, 2, 1],
# # ]
# rectangles = [
#     [-4, 7.5, 6, 3.5],     # Top center rectangle
#     [-2, -10.5, 3, 17],  # Left vertical rectangle
#     [4, -0.5, 4, 12],   # Right vertical rectangle
#     [4, -2, 4, 1],      # First right horizontal rectangle (topmost)
#     [4, -4, 4, 1],      # Second right horizontal rectangle
#     [4, -6, 4, 1],      # Third right horizontal rectangle
#     [4, -8, 4, 1],      # Fourth right horizontal rectangle
#     [4, -10, 4, 1],     # Fifth right horizontal rectangle
#     [4, -12, 4, 1],     # Sixth right horizontal rectangle (bottommost)
# ]

# # Draw the rectangles
# colors = ['black', 'yellow', 'pink', 'red', 'blue', 'green', 'orange', 'cyan', 'purple']
# for i, rect in enumerate(rectangles):
#     x, y, width, height = rect
#     ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor=colors[i], facecolor='none', linewidth=1.5))

# # Add start position
# start = (2.5, -7)
# ax.plot(start[0], start[1], 'o', color='blue', label='Start')

# # Add goal position
# goal = (0, 7)
# ax.plot(goal[0], goal[1], 'o', color='green', label='Goal')

# # Add an arrow for orientation
# arrow_dx, arrow_dy = 1, 2  # Arrow direction
# ax.arrow(start[0], start[1], arrow_dx, arrow_dy, head_width=0.5, head_length=0.5, fc='red', ec='red')

# # Labels and legend
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.legend(loc='upper right')
# ax.grid()

# # Show plot
# plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# from queue import PriorityQueue

# # Helper functions for A* algorithm
# def heuristic(a, b):
#     return np.linalg.norm(np.array(a) - np.array(b))

# def is_in_obstacle(x, y, obstacles):
#     """Check if a point (x, y) is inside any obstacle."""
#     for obs in obstacles:
#         ox, oy, ow, oh = obs
#         if ox <= x <= ox + ow and oy <= y <= oy + oh:
#             return True
#     return False

# # A* Algorithm for Path Finding

# def a_star(start, goal, obstacles, grid_size=0.5):

#     open_set = PriorityQueue()
#     open_set.put((0, start))
#     came_from = {}
#     cost_so_far = {start: 0}
    
#     while not open_set.empty():
#         _, current = open_set.get()
        
#         if current == goal:
#             # Reconstruct the path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]
        
#         x, y = current
#         for dx, dy in [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]:
#             next_point = (round(x + dx, 2), round(y + dy, 2))
#             if is_in_obstacle(next_point[0], next_point[1], obstacles):
#                 continue
            
#             new_cost = cost_so_far[current] + heuristic(current, next_point)
#             if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
#                 cost_so_far[next_point] = new_cost
#                 priority = new_cost + heuristic(next_point, goal)
#                 open_set.put((priority, next_point))
#                 came_from[next_point] = current
    
#     return []  # Return an empty path if no valid path is found

# # Main visualization and simulation
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-11, 11)
# ax.set_ylim(-11, 11)

# # Obstacles
# rectangles = [
#     [-4, 7.5, 6, 3.5],  # Top center rectangle
#     [-2, -10.5, 3, 17],  # Left vertical rectangle
#     [4, -0.5, 4, 12],  # Right vertical rectangle
#     [4, -2, 4, 1],  # First right horizontal rectangle
#     [4, -4, 4, 1],  # Second right horizontal rectangle
#     [4, -6, 4, 1],  # Third right horizontal rectangle
#     [4, -8, 4, 1],  # Fourth right horizontal rectangle
#     [4, -10, 4, 1],  # Fifth right horizontal rectangle
#     [4, -12, 4, 1],  # Sixth right horizontal rectangle
# ]

# # Draw obstacles
# colors = ['black', 'yellow', 'pink', 'red', 'blue', 'green', 'orange', 'cyan', 'purple']
# for i, rect in enumerate(rectangles):
#     x, y, width, height = rect
#     ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor=colors[i], facecolor='none', linewidth=1.5))

# # Define start and goal
# start = (2.5, -7)
# goal = (0, 7)

# # Path planning
# path = a_star(start, goal, rectangles)

# # Draw the path
# if path:
#     path_x, path_y = zip(*path)
#     ax.plot(path_x, path_y, linestyle='--', color='blue', label='Planned Path')
# else:
#     print("No valid path found!")

# # Add start and goal
# ax.plot(start[0], start[1], 'o', color='blue', label='Start')
# ax.plot(goal[0], goal[1], 'o', color='green', label='Goal')

# # Simulate arrow movement along the path
# for i in range(len(path) - 1):
#     current = path[i]
#     next_point = path[i + 1]
#     arrow_dx, arrow_dy = next_point[0] - current[0], next_point[1] - current[1]
    
#     ax.arrow(
#         current[0], current[1],
#         arrow_dx, arrow_dy,
#         head_width=0.3, head_length=0.3, fc='red', ec='red'
#     )

# # Labels and legend
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.legend(loc='upper right')
# ax.grid()

# # Show plot
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# from queue import PriorityQueue

# # Helper functions for A* algorithm
# def heuristic(a, b):
#     return np.linalg.norm(np.array(a) - np.array(b))

# def is_in_obstacle(x, y, obstacles):
#     """Check if a point (x, y) is inside any obstacle."""
#     for obs in obstacles:
#         ox, oy, ow, oh = obs
#         if ox <= x <= ox + ow and oy <= y <= oy + oh:
#             return True
#     return False

# # A* Algorithm for Path Finding

# def a_star(start, goal, obstacles, grid_size=0.5):

#     open_set = PriorityQueue()
#     open_set.put((0, start))
#     came_from = {}
#     cost_so_far = {start: 0}
    
#     while not open_set.empty():
#         _, current = open_set.get()
        
#         if current == goal:
#             # Reconstruct the path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]
        
#         x, y = current
#         for dx, dy in [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]:
#             next_point = (round(x + dx, 2), round(y + dy, 2))
#             if is_in_obstacle(next_point[0], next_point[1], obstacles):
#                 continue
            
#             new_cost = cost_so_far[current] + heuristic(current, next_point)
#             if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
#                 cost_so_far[next_point] = new_cost
#                 priority = new_cost + heuristic(next_point, goal)
#                 open_set.put((priority, next_point))
#                 came_from[next_point] = current
    
#     return []  # Return an empty path if no valid path is found

# # Main visualization and simulation
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-11, 11)
# ax.set_ylim(-11, 11)

# # Obstacles
# rectangles = [
#     [-4, 7.5, 6, 3.5],  # Top center rectangle
#     [-2, -10.5, 3, 17],  # Left vertical rectangle
#     [4, -0.5, 4, 12],  # Right vertical rectangle
#     [4, -2, 4, 1],  # First right horizontal rectangle
#     [4, -4, 4, 1],  # Second right horizontal rectangle
#     [4, -6, 4, 1],  # Third right horizontal rectangle
#     [4, -8, 4, 1],  # Fourth right horizontal rectangle
#     [4, -10, 4, 1],  # Fifth right horizontal rectangle
#     [4, -12, 4, 1],  # Sixth right horizontal rectangle
# ]

# # Draw obstacles
# colors = ['black', 'yellow', 'pink', 'red', 'blue', 'green', 'orange', 'cyan', 'purple']
# for i, rect in enumerate(rectangles):
#     x, y, width, height = rect
#     ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor=colors[i], facecolor='none', linewidth=1.5))

# # Define start and goal
# start = (2.5, -7)
# goal = (0, 7)

# # Path planning
# path = a_star(start, goal, rectangles)

# # Draw the path in black
# if path:
#     path_x, path_y = zip(*path)
#     ax.plot(path_x, path_y, linestyle='-', color='black', label='Planned Path')
# else:
#     print("No valid path found!")

# # Add start and goal
# ax.plot(start[0], start[1], 'o', color='blue', label='Start')
# ax.plot(goal[0], goal[1], 'o', color='green', label='Goal')

# # Labels and legend
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.legend(loc='upper right')
# ax.grid()

# # Show plot
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from queue import PriorityQueue
import matplotlib.animation as animation

# Helper functions for A* algorithm
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_in_obstacle(x, y, obstacles):
    """Check if a point (x, y) is inside any obstacle."""
    for obs in obstacles:
        ox, oy, ow, oh = obs
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            return True
    return False

# A* Algorithm for Path Finding
def a_star(start, goal, obstacles, grid_size=0.5):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}
    
    while not open_set.empty():
        _, current = open_set.get()
        
        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        x, y = current
        for dx, dy in [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]:
            next_point = (round(x + dx, 2), round(y + dy, 2))
            if is_in_obstacle(next_point[0], next_point[1], obstacles):
                continue
            
            new_cost = cost_so_far[current] + heuristic(current, next_point)
            if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
                cost_so_far[next_point] = new_cost
                priority = new_cost + heuristic(next_point, goal)
                open_set.put((priority, next_point))
                came_from[next_point] = current
    
    return []  # Return an empty path if no valid path is found

# Function to generate waypoints from a path
def generate_waypoints(path):
    waypoints = []
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        direction = end - start
        distance = np.linalg.norm(direction)
        num_waypoints = int(distance / 0.5)  # Adjust spacing between waypoints
        
        for j in range(num_waypoints + 1):
            waypoint = start + (direction / distance) * (j * (distance / num_waypoints))
            waypoints.append(tuple(waypoint))
    
    return waypoints

# Main visualization and simulation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)

# Obstacles
rectangles = [
    [-4, 7.5, 6, 3.5],  
    [-2, -10.5, 3, 17],  
    [4, -0.5, 4, 12],  
    [4, -2, 4, 1],  
    [4, -4, 4, 1],  
    [4, -6, 4, 1],  
    [4, -8, 4, 1],  
    [4, -10, 4, 1],  
    [4, -12, 4, 1],  
]

# Draw obstacles
colors = ['black', 'yellow', 'pink', 'red', 'blue', 'green', 'orange', 'cyan', 'purple']
for i, rect in enumerate(rectangles):
    x,y,width,height = rect
    ax.add_patch(patches.Rectangle((x,y), width,height,
                                    edgecolor=colors[i],
                                    facecolor='none',
                                    linewidth=1.5))

# Define start and goal
start = (2.5,-7)
goal = (0 ,7)

# Path planning
path = a_star(start , goal , rectangles)

# Generate waypoints
waypoints = generate_waypoints(path)

# Initialize arrow and dots
arrow_length = 0.5
arrow_head_length = 0.15

# Create an initial arrow at the first waypoint
arrow_line = ax.arrow(waypoints[0][0], waypoints[0][1],
                      waypoints[1][0] - waypoints[0][0],
                      waypoints[1][1] - waypoints[0][1],
                      head_width=arrow_head_length,
                      head_length=arrow_head_length,
                      fc='blue', ec='blue')

dots_plot = ax.plot([], [], 'o', color='orange')[0]

# Animation function
def update(frame):
    # Update arrow position and direction only if there are enough waypoints left
    if frame < len(waypoints) - 2:
        # Update arrow position and direction
        dx = waypoints[frame + 1][0] - waypoints[frame][0]
        dy = waypoints[frame + 1][1] - waypoints[frame][1]
        
        # Remove previous arrow and draw a new one at updated position
        for patch in ax.patches:
            if isinstance(patch , patches.FancyArrowPatch):
                patch.remove()
        
        ax.arrow(waypoints[frame][0], waypoints[frame][1],
                dx,
                dy,
                head_width=arrow_head_length,
                head_length=arrow_head_length,
                fc='blue', ec='blue')
        
        # Update dots positions (3 dots)
        dots_x = [waypoints[frame][0], waypoints[frame + 1][0], waypoints[frame + 2][0]]
        dots_y = [waypoints[frame][1], waypoints[frame + 1][1], waypoints[frame + 2][1]]
        
        dots_plot.set_data(dots_x , dots_y)
    
    return dots_plot,

# Create animation
ani = animation.FuncAnimation(fig , update , frames=len(waypoints)-2 , interval=300)

# Add start and goal markers
ax.plot(start[0] , start[1] , 'o' , color='blue' , label='Start')
ax.plot(goal[0] , goal[1] , 'o' , color='green' , label='Goal')

# Labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend(loc='upper right')
ax.grid()

# Show plot with animation
plt.show()
