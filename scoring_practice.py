from practice import Practice
import constants as c
import helpers as h
import numpy as np
import random
import globals as g

class ScoringPractice(Practice):
    def __init__(self):
        super().__init__()
        self.difficulty_increase = 0.02
        self.starting_difficulty = 0.3
        self.difficulty_alpha = self.starting_difficulty
        self.init_params()
        self.consecutive_goal_req = 3
        self.consecutive_win_req = 4
        self.max_failure_atempts = 8
        self.name = "scoring"
        # c.settings["goal_1_blocked"] = True

        # self.reward = 0
        self.goal_reward = 7500
        # self.level_reward = 0
        self.regular_start = True
        c.training["learning_rate"] = 0.15e-4
        c.training["ent_coef"] = 0.0005

        self.reward_structure = {
            # "puck_proximity": 0.5,
            # "puck_vel_toward_goal": 0.2,
            # "goal_puck_proximity": 0.5,
            # # "shot": 0.5,
            # "shot_toward_goal": 1.5,
            # # "dash_shot": 1.0,
            # "pre_shot_placement": 0.3,
            # # "speed_dash": 0.5
        }

        c.model_names = ["15Aug", "Goalie", "MixScorer", "MixGoalie"]

        c.settings["round_time"] = 60
        # c.settings["goal_1_blocked"] = True

    def init_params(self):
        paddle_radius = 72
        self.min_params = {
            "paddle_min_dist": paddle_radius * 8.0,
            "paddle_max_dist": paddle_radius * 10.0,
            "paddle_angular_range": 40,
            "puck_min_dist": h.field_width() * 0.1,
            "puck_max_dist": h.field_width() * 0.2,
            "puck_angular_range": 30,
            "puck_init_speed_range": 0,
            "puck_init_spin_range": 0,
        }

        self.max_params = {
            "paddle_min_dist": paddle_radius * 8.0,
            "paddle_max_dist": paddle_radius * 30.0,
            "paddle_angular_range": 360,
            "puck_min_dist": h.field_width() * 0.5,
            "puck_max_dist": h.field_width() * 1.0,
            "puck_angular_range": 180,
            "puck_init_speed_range": 40,
            "puck_init_spin_range": 40,
        }

        super().update_params()

    def get_paddle_starting_pos(self, paddle, puck):
        super().seed_rngs(self.seed + paddle.get_idx())

        if paddle.team == 1 and paddle.player == 1:
            pos = np.array([random.uniform(c.settings["field_width"] * 0.0, c.settings["field_width"] * 1.0),
                                random.uniform(c.settings["field_height"] * 0.0, c.settings["field_height"] * 1.0)],
                                dtype=np.float32)

            # puck_pos = np.copy(puck.pos)
            # goal_to_puck = puck_pos - h.goal_pos(2)
            # goal_to_puck_dir = goal_to_puck / np.linalg.norm(goal_to_puck)
            # pos = h.random_vector_within_cone(puck_pos, goal_to_puck_dir, self.params["paddle_min_dist"], self.params["paddle_max_dist"], self.params["paddle_angular_range"])
        else:
            while True:
                pos = np.array([random.uniform(2*paddle.radius, c.settings["field_width"] * 1.0 - 2*paddle.radius),
                                        random.uniform(2*paddle.radius, c.settings["field_height"] - 2*paddle.radius)],
                                        dtype=np.float32)

                dist_to_agent = np.linalg.norm(g.game.paddles_1[0].pos - pos)
                dist_to_puck = np.linalg.norm(g.game.puck.pos - pos)

                if dist_to_agent > 500 and dist_to_puck > 120:
                    break

        return pos

    def get_puck_starting_pos(self):
        super().seed_rngs(self.seed + 20)
        pos = np.array([random.uniform(c.settings["field_width"] * 0.0, c.settings["field_width"] * 1.0),
                                random.uniform(c.settings["field_height"] * 0.0, c.settings["field_height"] * 1.0)],
                                dtype=np.float32)
        # pos = h.random_vector_within_cone(h.goal_pos(2), np.array([-1.0, 0.0]), self.params["puck_min_dist"], self.params["puck_max_dist"], self.params["puck_angular_range"])

        return pos

    def handle_goal_achieved(self):
        # time_bonus = g.game.time_left() * 500
        # self.collectable_reward += time_bonus
        # power_bonus = (np.linalg.norm(g.game.puck.vel) / c.gameplay["max_puck_speed"]) * 1600
        # self.collectable_reward += power_bonus
        # print(f"time_bonus {time_bonus}")
        # print(f"power_bonus {power_bonus}")
        pass
        # # self.change_level()

    def handle_goal_failed(self):
        # self.change_level()
        pass

    def get_puck_starting_vel(self, puck):
        self.seed_rngs(self.seed + 40)
        init_speed = random.random() * self.params["puck_init_speed_range"]
        vel = h.generate_random_2d_dir_vector()
        return vel * init_speed

    def get_puck_starting_rot_vel(self):
        self.seed_rngs(self.seed + 40)
        init_spin = (random.random() - 0.5) * 2 * self.params["puck_init_spin_range"]
        return init_spin

    def specific_update(self):
        pass
        # if "paddle_collision" in self.events:
        #     # self.collectable_reward -= 1000
        #     self.events.append("round_end")
        #     g.game.reset()

    def specific_level_change(self):
        pass

    def goal_failed(self):
        if "round_end" in self.events:
            if "2_scored" not in self.events:
                self.collectable_reward += 1600
            self.events = []
            return True
        else:
            return False

    def goal_achieved(self):
        if self.name in self.events:
            self.events = []
            print(f"Difficulty: {self.difficulty_alpha}")
            return True
        else:
            return False