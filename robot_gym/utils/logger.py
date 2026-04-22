import numpy as np
from collections import defaultdict


class Logger:
    def __init__(self, dt: float):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0

    def log_state(self, key, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        self.state_log[key].append(value)

    def log_states(self, data: dict):
        for key, value in data.items():
            self.log_state(key, value)

    def log_rewards(self, rewards: dict, num_episodes: int):
        for key, value in rewards.items():
            if hasattr(value, "item"):
                value = value.item()
            self.rew_log[key].append(float(value) * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
        self.num_episodes = 0

    def get_mean_rewards(self):
        if self.num_episodes == 0:
            return {}
        out = {}
        for key, values in self.rew_log.items():
            out[key] = float(np.sum(np.array(values)) / self.num_episodes)
        return out

    def print_rewards(self):
        mean_rewards = self.get_mean_rewards()
        print("Average rewards per episode:")
        for key, value in mean_rewards.items():
            print(f" - {key}: {value:.6f}")
        print(f"Total number of episodes: {self.num_episodes}")