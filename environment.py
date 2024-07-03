import game
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pprint import pprint

class AirHockeyEnv(gym.Env):
    def __init__(self, players_per_team=2):
        super(AirHockeyEnv, self).__init__()
        self.game = game.Game(players_per_team)

        self.observation_space = {
            "puck_pos":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_vel":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }

        other_players_positions_team_1 = { f"paddle_{1}_{i}_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(2, players_per_team + 1) }
        other_players_positions_team_2 = { f"paddle_{2}_{i}_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, players_per_team + 1) }
        other_players_velocities_team_1 = { f"paddle_{1}_{i}_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, players_per_team + 1) }
        other_players_velocities_team_2 = { f"paddle_{2}_{i}_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, players_per_team + 1) }

        self.observation_space |= {
            **other_players_positions_team_1,
            **other_players_positions_team_2,
            **other_players_velocities_team_1,
            **other_players_velocities_team_2
        }

        self.observation_space = spaces.Dict(self.observation_space)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        print("Resetting environment")
        super().reset(seed=seed)
        self.game.reset()
        observation = self.game.get_observation(1, 1)
        return observation, {}

    def step(self, action):
        observation, reward, done, info = self.game.step_training(action)
        if done:
            print("Team 1 reward breakdown:")
            pprint(info['reward_breakdown_1'], width=1)
            print("")
            print("Team 2 reward breakdown:")
            pprint(info['reward_breakdown_2'], width=1)
            print("")
            print("")
        truncated = False
        return observation, reward, done, truncated, info

    def close(self):
        self.game.close()
