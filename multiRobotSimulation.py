# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import heapq

# # Grid limits
# GRID_MIN, GRID_MAX = -4, 4

# # Robots' start and goal positions
# robots = [
#     {'start': (-4, -4), 'goal': (4, 4), 'color': 'red'},
#     {'start': (4, -4), 'goal': (-4, 4), 'color': 'blue'},
#     {'start': (4, 4), 'goal': (-4, -4), 'color': 'green'},
#     {'start': (-4, 4), 'goal': (4, -4), 'color': 'orange'},
# ]

# # A* algorithm
# def a_star(start, goal, grid_min, grid_max, obstacles):
#     def heuristic(a, b):
#         return abs(a[0] - b[0]) + abs(a[1] - b[1])

#     neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: heuristic(start, goal)}

#     while open_set:
#         _, current = heapq.heappop(open_set)
#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             return path[::-1]

#         for dx, dy in neighbors:
#             neighbor = (current[0] + dx, current[1] + dy)
#             if grid_min <= neighbor[0] <= grid_max and grid_min <= neighbor[1] <= grid_max and neighbor not in obstacles:
#                 tentative_g_score = g_score[current] + 1
#                 if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g_score
#                     f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
#     return []

# # Calculate paths for all robots
# paths = {robot['color']: a_star(robot['start'], robot['goal'], GRID_MIN, GRID_MAX, set()) for robot in robots}

# # Animation setup
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(GRID_MIN, GRID_MAX + 1)
# ax.set_ylim(GRID_MIN, GRID_MAX + 1)
# ax.set_xticks(np.arange(GRID_MIN, GRID_MAX + 1, 1))
# ax.set_yticks(np.arange(GRID_MIN, GRID_MAX + 1, 1))
# ax.grid(True)

# # Robots and their paths
# robot_patches = {robot['color']: plt.Circle(robot['start'], 0.3, color=robot['color']) for robot in robots}
# for patch in robot_patches.values():
#     ax.add_patch(patch)

# # Collision avoidance variables
# current_positions = {robot['color']: robot['start'] for robot in robots}
# next_positions = {}

# # Add start and goal points
# for robot in robots:
#     ax.plot(*robot['start'], marker='o', color=robot['color'], markersize=8)
#     ax.plot(*robot['goal'], marker='o', color=robot['color'], markersize=8, markerfacecolor='none')

# def update(frame):
#     global current_positions, next_positions
#     next_positions = {}

#     # Calculate next positions
#     for robot in robots:
#         color = robot['color']
#         path = paths[color]
#         if frame < len(path):
#             next_positions[color] = path[frame]

#     # Resolve collisions
#     occupied_positions = set(current_positions.values())
#     for color, next_pos in next_positions.items():
#         if next_pos in occupied_positions:
#             next_positions[color] = current_positions[color]  # Stay in place
#         else:
#             occupied_positions.add(next_pos)

#     # Update robot positions
#     for robot in robots:
#         color = robot['color']
#         if color in next_positions:
#             current_positions[color] = next_positions[color]
#             robot_patches[color].center = current_positions[color]

#     return list(robot_patches.values())

# frames = max(len(path) for path in paths.values())
# ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# Grid limits
GRID_MIN, GRID_MAX = -4, 4

# Robots' start and goal positions
robots = [
    {'start': (-4, -4), 'goal': (4, 4), 'color': 'red'},
    {'start': (4, -4), 'goal': (-4, 4), 'color': 'blue'},
    {'start': (4, 4), 'goal': (-4, -4), 'color': 'green'},
    {'start': (-4, 4), 'goal': (4, -4), 'color': 'orange'},
]

# A* algorithm with diagonal moves
def a_star(start, goal, grid_min, grid_max, obstacles):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
    ]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if grid_min <= neighbor[0] <= grid_max and grid_min <= neighbor[1] <= grid_max and neighbor not in obstacles:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# Calculate paths for all robots
paths = {robot['color']: a_star(robot['start'], robot['goal'], GRID_MIN, GRID_MAX, set()) for robot in robots}

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(GRID_MIN, GRID_MAX + 1)
ax.set_ylim(GRID_MIN, GRID_MAX + 1)
ax.set_xticks(np.arange(GRID_MIN, GRID_MAX + 1, 1))
ax.set_yticks(np.arange(GRID_MIN, GRID_MAX + 1, 1))
ax.grid(True)

# Robots and their paths
robot_patches = {robot['color']: plt.Circle(robot['start'], 0.3, color=robot['color']) for robot in robots}
for patch in robot_patches.values():
    ax.add_patch(patch)

# Collision avoidance variables
current_positions = {robot['color']: robot['start'] for robot in robots}
next_positions = {}

# Add start and goal points
for robot in robots:
    ax.plot(*robot['start'], marker='o', color=robot['color'], markersize=8)
    ax.plot(*robot['goal'], marker='o', color=robot['color'], markersize=8, markerfacecolor='none')

def update(frame):
    global current_positions, next_positions
    next_positions = {}

    # Calculate next positions
    for robot in robots:
        color = robot['color']
        path = paths[color]
        if frame < len(path):
            next_positions[color] = path[frame]

    # Resolve collisions
    occupied_positions = set(current_positions.values())
    for color, next_pos in next_positions.items():
        if next_pos in occupied_positions:
            next_positions[color] = current_positions[color]  # Stay in place
        else:
            occupied_positions.add(next_pos)

    # Update robot positions
    for robot in robots:
        color = robot['color']
        if color in next_positions:
            current_positions[color] = next_positions[color]
            robot_patches[color].center = current_positions[color]

    return list(robot_patches.values())

frames = max(len(path) for path in paths.values())
ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
plt.show()
