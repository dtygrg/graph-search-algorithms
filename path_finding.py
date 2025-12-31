import heapq
import math
import random
from collections import deque
from collections.abc import Generator

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage


# ================== MAZE GENERATOR ==================
def generate_maze(rows: int, cols: int) -> list[list[int]]:
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    def carve(r, c):
        maze[r][c] = 0
        dirs = [(2,0), (-2,0), (0,2), (0,-2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows-1 and 1 <= nc < cols-1 and maze[nr][nc] == 1:
                maze[r + dr//2][c + dc//2] = 0
                carve(nr, nc)

    carve(1, 1)
    return maze

def add_loops(maze: list[list[int]], chance: float=0.08):
    rows, cols = len(maze), len(maze[0])
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if maze[r][c] == 1 and random.random() < chance:
                maze[r][c] = 0

# ================== CONFIG ==================
ROWS, COLS = 51, 51
maze = generate_maze(ROWS, COLS)
add_loops(maze)

start: tuple[int, int] = (1, 1)
goal: tuple[int, int] = (ROWS-2, COLS-2)

DIRS: list[tuple[int, int, float]] = [
    (0, 1, 1),  # right
    (1, 0, 1),  # down
    (0, -1, 1), # left
    (-1, 0, 1), # up
    (1, 1, math.sqrt(2)),   # down-right
    (1, -1, math.sqrt(2)),  # down-left
    (-1, 1, math.sqrt(2)),  # up-right
    (-1, -1, math.sqrt(2))  # up-left
]

# ================== COLORS ==================
colors = [
    "white",      # EMPTY
    "black",      # WALL
    "#4C72B0",    # VISITED (blue)
    "#DD8452",    # FRONTIER (orange)
    "#55A868",    # START (green)
    "#C44E52",    # GOAL (red)
    "#FFD700"     # PATH (yellow)
]

cmap = ListedColormap(colors)

EMPTY, WALL, VISITED, FRONTIER, START, GOAL, PATH = range(7)

def base_grid() -> NDArray[np.int8]:
    g = np.zeros((ROWS, COLS), dtype=np.int8)
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == 1:
                g[r][c] = WALL
    g[start] = START
    g[goal] = GOAL
    return g

def neighbors(r: int, c: int) -> Generator[tuple[int, int, float]]:
    random.shuffle(DIRS)
    for dr, dc, cost in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr][nc] == 0:
            yield nr, nc, cost

def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ================== SEARCH RUNNER ==================
def run(alg: str) -> list[NDArray[np.int8]]:
    grid = base_grid()
    frames: list[NDArray[np.int8]] = []
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    if alg == "BFS":
        frontier = deque([start])
        pop = frontier.popleft
        push = frontier.append

    elif alg == "DFS":
        frontier = [start]
        pop = frontier.pop
        push = frontier.append

    elif alg == "Dijkstra":
        frontier = [(0, start)]
        dist = {start: 0}
        pop = lambda: heapq.heappop(frontier)[1]
        push = lambda n, c: heapq.heappush(frontier, (c, n))

    elif alg == "A*":
        frontier = [(heuristic(start, goal), start)]
        g_cost = {start: 0}
        pop = lambda: heapq.heappop(frontier)[1]
        push = lambda n, f: heapq.heappush(frontier, (f, n))

    else:
        return []

    while frontier:
        node = pop()
        if node == goal:
            break
        if node in visited:
            continue

        visited.add(node)
        if node != start:
            grid[node] = VISITED

        for (step_x, step_y, step_cost) in neighbors(*node):
            n = (step_x, step_y)
            if n in visited:
                continue
            parent[n] = node

            if alg in ("BFS", "DFS"):
                push(n)
            elif alg == "Dijkstra":
                cost = dist[node] + step_cost
                if n not in dist or cost < dist[n]:
                    dist[n] = cost
                    push(n, cost)
            elif alg == "A*":
                g = g_cost[node] + step_cost
                if n not in g_cost or g < g_cost[n]:
                    g_cost[n] = g
                    push(n, g + heuristic(n, goal))

            if grid[n] == EMPTY:
                grid[n] = FRONTIER

        frames.append(grid.copy())

    # Path
    node = goal
    while (node is not None) and (node != start):
        node = parent.get(node)
        if node and node != start:
            grid[node] = PATH
    frames.append(grid.copy())

    return frames

# ================== ANIMATION ==================
algorithms = ["BFS", "DFS", "Dijkstra", "A*"]
frames = {a: run(a) for a in algorithms}
max_len = max(len(f) for f in frames.values())

for a in algorithms:
    while len(frames[a]) < max_len:
        frames[a].append(frames[a][-1])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
imgs: list[AxesImage] = []

for ax, alg in zip(axes, algorithms):
    img = ax.imshow(frames[alg][0], cmap=cmap, vmin=0, vmax=6)
    ax.set_title(alg)
    ax.axis("off")
    imgs.append(img)

def update(i: int) -> list[AxesImage]:
    for img, alg in zip(imgs, algorithms):
        img.set_data(frames[alg][i])
    return imgs

ani = animation.FuncAnimation(
    fig, update, frames=max_len, interval=5, repeat=False
)

plt.show()
