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
            "paddle_2_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_pos":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "paddle_1_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "paddle_2_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_vel":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),                                    
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        print("Resetting environment")
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation(1)
        return {k: v.numpy() for k, v in observation.items()}, {}

    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        observation = {k: v.numpy() for k, v in observation.items()}
        if done:
            print(f'Total reward: {info['cumulative_reward']}')
        truncated = False
        return observation, reward, done, truncated, info

    def close(self):
        self.game.close()
