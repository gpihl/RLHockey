import gymnasium as gym
from gymnasium import spaces
import numpy as np
import constants as c
import globals as g

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        team_size = c.settings["team_size"]

        self.observation_space = {}

        # for j in range(1, 6):
        j = 1
        self.observation_space |= {
            "self_pos_absolute_" + str(j):     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_pos_absolute_" + str(j):     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_vel_absolute_" + str(j):     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_rot_vel_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "charging_alpha_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "puck_pos_relative_" + str(j):     spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_top_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_bot_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_top_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_bot_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }

        other_players_positions_team_1 = { f"paddle_{1}_{i}_pos_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(2, team_size + 1) }
        other_players_positions_team_2 = { f"paddle_{2}_{i}_pos_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }
        other_players_velocities_team_1 = { f"paddle_{1}_{i}_vel_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }
        other_players_velocities_team_2 = { f"paddle_{2}_{i}_vel_absolute_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }

        self.observation_space |= {
            **other_players_positions_team_1,
            **other_players_positions_team_2,
            **other_players_velocities_team_1,
            **other_players_velocities_team_2
        }

        other_players_positions_team_1 = { f"paddle_{1}_{i}_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(2, team_size + 1) }
        other_players_positions_team_2 = { f"paddle_{2}_{i}_pos_relative_" + str(j): spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) for i in range(1, team_size + 1) }

        self.observation_space |= {
            **other_players_positions_team_1,
            **other_players_positions_team_2,
            **other_players_velocities_team_1,
            **other_players_velocities_team_2
        }

        self.observation_space |= {
            "goal_1_top_pos_absolute_1": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_1_bot_pos_absolute_1": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_top_pos_absolute_1": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "goal_2_bot_pos_absolute_1": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "puck_finding_practice_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "shooting_practice_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "passing_practice_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defensive_practice_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "scoring_practice_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "full_game_1": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        self.observation_space = spaces.Dict(self.observation_space)
        g.observation_space = self.observation_space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        print("Resetting environment")
        super().reset(seed=seed)
        g.game.reset()
        observation = g.game.get_observation(g.game.paddles_1[0])
        return observation, {}

    def step(self, action):
        observation, reward, done, info = g.game.step_training(action)
        truncated = False

        if done:
            print(f"Round reward: {g.game.round_reward}")

        return observation, reward, done, truncated, info

    # def process_observation(self, obs):
    #     filtered_obs = { key: value for key, value in obs.items() if key in self.observation_space.spaces }
    #     return filtered_obs

    def close(self):
        g.game.close()
