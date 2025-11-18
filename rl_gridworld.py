import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


# =====================================================
#  GRIDWORLD ENVIRONMENT
# =====================================================

class GridWorldEnv:
    def __init__(self, nrows=6, ncols=6, start=(0, 0), goal=(5, 5), holes=None):
        self.nrows = nrows
        self.ncols = ncols
        self.start = start
        self.goal = goal
        self.holes = holes if holes else []

        self.state = start

        # actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        if action == 0:     # up
            r = max(0, r - 1)
        elif action == 1:   # down
            r = min(self.nrows - 1, r + 1)
        elif action == 2:   # left
            c = max(0, c - 1)
        elif action == 3:   # right
            c = min(self.ncols - 1, c + 1)

        new_state = (r, c)
        self.state = new_state

        # rewards
        if new_state == self.goal:
            return new_state, 10, True
        if new_state in self.holes:
            return new_state, -5, True

        return new_state, -1, False

    def get_state_space_size(self):
        return self.nrows * self.ncols

    def state_to_index(self, state):
        return state[0] * self.ncols + state[1]


# =====================================================
#  Q-LEARNING TRAINING
# =====================================================

def train_q_learning(env, episodes=500, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
    Q = np.zeros((env.get_state_space_size(), 4))
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            s_idx = env.state_to_index(state)

            # Exploration vs exploitation
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space)
            else:
                action = np.argmax(Q[s_idx])

            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_index(next_state)

            # Q-learning update
            Q[s_idx, action] = Q[s_idx, action] + alpha * (
                    reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        epsilon *= epsilon_decay

    return Q, rewards


# =====================================================
#  SAVE & LOAD MODEL
# =====================================================

def save_model(Q, path='models/q_table.pkl'):
    os.makedirs('models', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(Q, f)


def load_model(path='models/q_table.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


# =====================================================
#  PLOT REWARDS
# =====================================================

def plot_rewards(rewards, savepath='static/images/rewards.png'):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("Recompensa por episodio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa acumulada")
    plt.grid()
    os.makedirs('static/images', exist_ok=True)
    plt.savefig(savepath)
    plt.close()


# =====================================================
#  SIMULATE EPISODE
# =====================================================

def simulate_episode(env, Q):
    state = env.reset()
    done = False
    trajectory = [state]
    total_reward = 0

    while not done:
        s_idx = env.state_to_index(state)
        action = np.argmax(Q[s_idx])
        next_state, reward, done = env.step(action)
        trajectory.append(next_state)
        total_reward += reward
        state = next_state

    return trajectory, total_reward, done


# =====================================================
#  PLOT TRAJECTORY
# =====================================================

def plot_trajectory(trajectory, env, savepath='static/images/traj.png'):
    grid = np.zeros((env.nrows, env.ncols))

    for h in env.holes:
        grid[h] = -1
    grid[env.goal] = 2

    traj_r = [s[0] for s in trajectory]
    traj_c = [s[1] for s in trajectory]

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='coolwarm_r')
    plt.plot(traj_c, traj_r, marker='o', linestyle='-', linewidth=2)
    plt.title("Trayectoria del Agente")
    plt.gca().invert_yaxis()

    os.makedirs('static/images', exist_ok=True)
    plt.savefig(savepath)
    plt.close()
