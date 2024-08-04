import gymnasium as gym
from gymnasium import spaces
import numpy as np
import constants as c
import globals as g

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        team_size = c.settings["team_size"]

        self.observation_space = {
            "self_pos":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_pos":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_vel":     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_rot_vel": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "charging_alpha": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "goal_1_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_top_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_bot_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_finding_practice": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "shooting_practice": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "passing_practice": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defensive_practice": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "scoring_practice": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "full_game": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        other_players_positions_team_1 = { f"paddle_{1}_{i}_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(2, team_size + 1) }
        other_players_positions_team_2 = { f"paddle_{2}_{i}_pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }
        other_players_velocities_team_1 = { f"paddle_{1}_{i}_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }
        other_players_velocities_team_2 = { f"paddle_{2}_{i}_vel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }

        self.observation_space |= {
            **other_players_positions_team_1,
            **other_players_positions_team_2,
            **other_players_velocities_team_1,
            **other_players_velocities_team_2
        }

        self.observation_space = spaces.Dict(self.observation_space)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        print("Resetting environment")
        super().reset(seed=seed)
        g.game.reset()
        observation = g.game.get_observation(1, 1)
        return observation, {}

    def step(self, action):
        observation, reward, done, info = g.game.step_training(action)
        truncated = False
        if done:
            print(g.game.round_reward)
        return observation, reward, done, truncated, info

    def close(self):
        g.game.close()
