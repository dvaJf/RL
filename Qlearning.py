import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np


maze = np.array([
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1, 2, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1,-1, 0,-1,-1],
    [-1, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0,-1],
    [-1,-1, 0,-1,-1,-1, 0,-1,-1,-1, 0,-1,-1,-1, 0,-1,-1, 0,-1,-1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1, 0,-1,-1,-1, 0,-1,-1, 0,-1, 0,-1,-1,-1, 0,-1,-1,-1, 0,-1],
    [-1, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1],
    [-1,-1, 0,-1,-1,-1,-1,-1, 0,-1,-1,-1, 0,-1,-1,-1,-1,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1, 0,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1],
    [-1,-1,-1, 0,-1,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1, 0,-1,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1],
    [-1, 0,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
])


alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
actions = 4

xlen, ylen = maze.shape
Q = np.zeros((xlen, ylen, actions))

def plot_visits_heatmap(maze, visits):
    h, w = maze.shape
    visits_masked = np.ma.masked_where(maze == -1, visits)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(visits_masked)
    for y in range(h):
        for x in range(w):
                ax.text(x,y,visits[y, x],ha="center",va="center",color="red",fontsize=8)
    plt.show()

def visualize(maze, Q):
    pos = [1,1]

    plt.ion()
    fig, ax = plt.subplots()

    for step_i in range(100):
        ax.clear()
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if maze[y, x] == -1:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color="black"))
                elif maze[y, x] == 1:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color="green"))
                else:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color="white", ec="gray"))
        ay, axx = pos
        ax.add_patch(plt.Circle((axx + 0.5, ay + 0.5), 0.3, color="red"))
        ax.set_xlim(0, maze.shape[1])
        ax.set_ylim(maze.shape[0], 0)
        ax.set_title(f"шаг {step_i}")
        ax.set_aspect("equal")
        action = np.argmax(Q[pos[0], pos[1]])
        pos, _, done = step(pos, action)
        if done:
            print("выход")
            break

    plt.ioff()
    plt.show()

def test(maze, Q):
    pos = [1,1]
    for step_i in range(100):
        action = np.argmax(Q[pos[0], pos[1]])
        pos, _, done = step(pos, action)
        if done:
            return 1
    return 0
    
def step(pos, action):
    y, x = pos
    moves = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    dy, dx = moves[action]
    ny, nx = y + dy, x + dx

    if maze[ny, nx] == -1:
        return pos, -1, False

    if maze[ny, nx] == 1:
        return (ny, nx), 10, True
    
    return (ny, nx), -0.1, False

episodes = 1000
max_steps = 500
visits = np.zeros(maze.shape, dtype=np.int32)

for ep in range(episodes):
    pos = [1,1]
    visits[pos[0], pos[1]] += 1
    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[pos[0], pos[1]])
        next_pos, reward, a = step(pos, action)
        y, x = pos
        ny, nx = next_pos
        Q[y, x, action] = Q[y, x, action] + alpha * (reward + gamma * np.max(Q[ny, nx]) - Q[y, x, action])
        pos = next_pos
        visits[ny, nx] += 1
        if a:
            break
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

np.save("Q_table.npy", Q)
Q = np.load("Q_table.npy")
plot_visits_heatmap(maze, visits)
s=0
for x in range(100):
    s+=test(maze, Q)
print(f"процент правильного прохождения лабиринта {s/100}")
