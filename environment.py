import game
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import globals as g

class AirHockeyEnv(gym.Env):
    def __init__(self, training=True):
        super(AirHockeyEnv, self).__init__()
        if training:
            self.game = game.Game(training=True)

        self.observation_space = spaces.Dict({
            "paddle_1_pos": spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32),
            "paddle_2_pos": spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32),
            "puck_pos":     spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32),
            "paddle_1_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "paddle_2_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_vel":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_dir":     spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-g.PADDLE_ACC, high=g.PADDLE_ACC, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        print("Resetting environment")
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation(1)
        return observation, {}

    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        if done:
            print(f'Total reward: {info['cumulative_reward']}')
        truncated = False
        return observation, reward, done, truncated, info

    def close(self):
        self.game.close()
