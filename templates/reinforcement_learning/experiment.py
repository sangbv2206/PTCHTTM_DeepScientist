import argparse
import json
import math
import os
import random
import time
from collections import deque
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================================================================
# SELF-CONTAINED CONTINUOUS-STATE MARKOV DECISION PROCESS ENVIRONMENT
# (Cart-Pole Inverted Pendulum Dynamics - Pure Python/NumPy, Zero Dependency)
# =========================================================================
class CartPoleMDP:
    def __init__(self, seed: int = 42):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_steps = 200

        self.rng = np.random.RandomState(seed)
        self.state = None
        self.steps_beyond_done = None
        self.step_count = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.step_count = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        self.step_count += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count >= self.max_steps
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}


# =========================================================================
# REPLAY BUFFER & DEEP Q-NETWORK ARCHITECTURE
# =========================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================================
# REINFORCEMENT LEARNING TRAINING LOOP
# =========================================================================
def train_rl_agent(
    num_episodes: int = 100,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.96,
    target_update_freq: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CartPoleMDP(seed=seed)
    q_network = DeepQNetwork(state_dim=4, action_dim=2, hidden_dim=64)
    target_network = DeepQNetwork(state_dim=4, action_dim=2, hidden_dim=64)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = epsilon_start
    episode_rewards = []
    td_losses = []
    convergence_episode = None

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_loss = 0.0
        loss_steps = 0

        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_vals = q_network(s_tensor)
                    action = q_vals.argmax(dim=1).item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Mini-batch gradient step
            if len(replay_buffer) >= batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)

                # Compute Q(s, a)
                current_q = q_network(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Compute Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    max_next_q = target_network(b_next_states).max(dim=1)[0]
                    target_q = b_rewards + (1.0 - b_dones) * gamma * max_next_q

                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                loss_steps += 1

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(float(total_reward))
        avg_loss = float(ep_loss / loss_steps) if loss_steps > 0 else 0.0
        td_losses.append(avg_loss)

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Check for convergence (average reward >= 180 over 10 consecutive episodes)
        if convergence_episode is None and len(episode_rewards) >= 10:
            if np.mean(episode_rewards[-10:]) >= 180.0:
                convergence_episode = episode

    mean_last_20 = float(np.mean(episode_rewards[-20:]))
    max_reward = float(np.max(episode_rewards))
    final_loss = float(np.mean(td_losses[-10:]))

    return {
        "mean_episode_reward": round(mean_last_20, 2),
        "max_episode_reward": round(max_reward, 2),
        "convergence_episode": convergence_episode if convergence_episode else num_episodes,
        "final_td_loss": round(final_loss, 4),
        "episodes_trained": num_episodes,
        "training_dynamics": {
            "episode_rewards": episode_rewards,
            "td_losses": td_losses,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning DQN Benchmark")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Directory to save experimental artifacts")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_time = time.time()
    results = train_rl_agent(num_episodes=args.episodes, lr=args.lr)
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)

    # Save final info JSON
    info_path = os.path.join(args.out_dir, "final_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[RL Experiment] Training complete in {elapsed:.2f}s.")
    print(f"Mean Reward (Last 20): {results['mean_episode_reward']}")
    print(f"Max Episode Reward: {results['max_episode_reward']}")
    print(f"Convergence Episode: {results['convergence_episode']}")
    print(f"Final TD Loss: {results['final_td_loss']}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
