import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

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
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
])

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super().__init__()
        self.maze = maze
        self.start = np.argwhere(maze == 2)[0]
        self.goal = np.argwhere(maze == 1)[0]
        self.pos = self.start
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=2, shape=maze.shape, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        info = {}
        return self._get_obs(), info

    def step(self, action):
        y, x = self.pos
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        dy, dx = moves[int(action)]
        ny, nx = y + dy, x + dx

        if self.maze[ny, nx] == -1:
            reward = -5 
            done = False
        elif self.maze[ny, nx] == 1:
            reward = 100 
            done = True
            y, x = ny, nx
        else:
            distance_to_goal = np.linalg.norm(np.array([ny, nx]) - np.array(self.goal))
            reward = max(0, 1 - distance_to_goal / np.linalg.norm(np.array(self.maze.shape)))
            y, x = ny, nx
            done = False

        self.pos = (y, x)
        terminated = done
        truncated = False
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = self.maze.copy()
        y, x = self.pos
        obs[y, x] = 3 
        return obs

def plot_path(maze, path):
    fig, ax = plt.subplots(figsize=(8,8))
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y,x] == -1:
                ax.add_patch(plt.Rectangle((x,y),1,1,color="black"))
            elif maze[y,x] == 1:
                ax.add_patch(plt.Rectangle((x,y),1,1,color="green"))
            else:
                ax.add_patch(plt.Rectangle((x,y),1,1,color="white", ec="gray"))
    for y, x in path:
        ax.add_patch(plt.Circle((x+0.5, y+0.5), 0.3, color="red"))
    ax.set_xlim(0, maze.shape[1])
    ax.set_ylim(maze.shape[0], 0)
    plt.show()

env = MazeEnv(maze)

model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=0.001,
            buffer_size=5000,
            learning_starts=100,
            batch_size=1024,
            gamma=0.95,
            train_freq=4,
            target_update_interval=50,
            exploration_fraction=1,
            exploration_final_eps=0.05,
            )
model.learn(total_timesteps=20000)

done = False
obs, info = env.reset()  
path = [env.pos]
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    path.append(env.pos) 
    done = terminated or truncated
model.save("dqn_model")
plot_path(maze, path)