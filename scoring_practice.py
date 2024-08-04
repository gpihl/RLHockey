from practice import Practice
import constants as c
import helpers as h
import numpy as np
import random

class ScoringPractice(Practice):
    def __init__(self):
        super().__init__()
        self.init_params()
        self.consecutive_goal_req = 3
        self.consecutive_win_req = 7
        self.name = "scoring"
        self.difficulty_increase = 0.02

        self.reward_structure = {
            "velocity": -0.0,
            "opponents_goal": 4000,
            "own_goal": -4000,
            "team_mate_proximity": 0.0,
            "wrong_side_of_puck": -0.2,
            "puck_proximity": 0.5,
            "puck_vel_toward_goal": 0.2,
            "goal_puck_proximity": 0.5,
            "shot": 1.0,
            "shot_toward_goal": 1.5,
            "dash_shot": 4.0,
            "puck_own_goal_prox": -0.0,
            "self_goal_prox": 0.0
        }

    def init_params(self):
        paddle_radius = 72
        self.min_params = {
            "paddle_min_dist": paddle_radius * 2.0,
            "paddle_max_dist": paddle_radius * 10.0,
            "paddle_angular_range": 80,
            "puck_min_dist": h.field_width() * 0.2,
            "puck_max_dist": h.field_width() * 0.4,
            "puck_angular_range": 30,
        }

        self.max_params = {
            "paddle_min_dist": paddle_radius * 2.0,
            "paddle_max_dist": paddle_radius * 30.0,
            "paddle_angular_range": 360,
            "puck_min_dist": h.field_width() * 0.2,
            "puck_max_dist": h.field_width() * 0.7,
            "puck_angular_range": 180,
        }

        super().update_params()

    def get_paddle_starting_pos(self, paddle, puck):
        super().seed_rngs(self.seed + paddle.get_idx())

        if paddle.team == 1 and paddle.player == 1:
            # pos = np.array([random.uniform(c.settings["field_width"] * 0.7, c.settings["field_width"] * 1.0),
            #                     random.uniform(c.settings["field_height"] * 0.0, c.settings["field_height"] * 1.0)],
            #                     dtype=np.float32)

            puck_pos = np.copy(puck.pos)
            goal_to_puck = puck_pos - h.goal_pos(2)
            goal_to_puck_dir = goal_to_puck / np.linalg.norm(goal_to_puck)
            pos = h.random_vector_within_cone(puck_pos, goal_to_puck_dir, self.params["paddle_min_dist"], self.params["paddle_max_dist"], self.params["paddle_angular_range"])
        else:
            pos = np.array([random.uniform(2*paddle.radius, c.settings["field_width"] * 0.5 - 2*paddle.radius),
                                    random.uniform(2*paddle.radius, c.settings["field_height"] - 2*paddle.radius)],
                                    dtype=np.float32)

        return pos

    def get_puck_starting_pos(self):
        super().seed_rngs(self.seed + 20)
        # starting_pos = np.array([random.uniform(c.settings["field_width"] * 0.7, c.settings["field_width"] * 1.0),
        #                         random.uniform(c.settings["field_height"] * 0.0, c.settings["field_height"] * 1.0)],
        #                         dtype=np.float32)

        pos = h.random_vector_within_cone(h.goal_pos(2), np.array([-1.0, 0.0]), self.params["puck_min_dist"], self.params["puck_max_dist"], self.params["puck_angular_range"])

        return pos

    def handle_goal_achieved(self):
        pass

