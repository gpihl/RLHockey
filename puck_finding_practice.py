from practice import Practice
import constants as c
import helpers as h
import numpy as np
import random
import globals as g

class PuckFindingPractice(Practice):
    def __init__(self):
        super().__init__()
        self.init_params()
        self.consecutive_goal_req = 3
        self.consecutive_win_req = 7
        self.name = "puck_finding"
        self.difficulty_increase = 0.02
        self.difficulty_alpha = 0.8

        self.reward_structure = {
            "velocity": -0.0,
            "opponents_goal": 0,
            "own_goal": -0,
            "team_mate_proximity": 0.0,
            "wrong_side_of_puck": 0.0,
            "puck_proximity": 2.0,
            "puck_vel_toward_goal": 0.0,
            "goal_puck_proximity": 0.0,
            "shot": 0.0,
            "shot_toward_goal": 0.0,
            "speed_dash": 15.0,
            "dash_shot": 0.0,
            "puck_own_goal_prox": -0.0,
            "self_goal_prox": 0.0
        }

    def init_params(self):
        self.min_params = {
            "puck_distance": 60,
            "puck_init_speed_range": 0,
            "puck_init_spin_range": 0,
        }

        self.max_params = {
            "puck_distance": h.field_width(),
            "puck_init_speed_range": 75,
            "puck_init_spin_range": 70,
        }

        super().update_params()

    def get_paddle_starting_pos(self, paddle, puck):
        self.seed_rngs(self.seed + paddle.get_idx())

        if paddle.team == 1 and paddle.player == 1:
            pos = h.random_vector_within_cone(puck.pos, np.array([1.0, 0]), 100, self.params["puck_distance"], 360)
        else:
            pos = np.array([random.uniform(c.settings["field_width"] * 0.0 + 100, c.settings["field_width"] * 1.0 - 100),
                            random.uniform(c.settings["field_height"] * 0.0 + 100, c.settings["field_height"] * 1.0 - 100)],
                            dtype=np.float32)

        return pos

    def get_puck_starting_pos(self):
        self.seed_rngs(self.seed + 20)
        pos = np.array([random.uniform(c.settings["field_width"] * 0.0 + 100, c.settings["field_width"] * 1.0 - 100),
                        random.uniform(c.settings["field_height"] * 0.0 + 100, c.settings["field_height"] * 1.0 - 100)],
                        dtype=np.float32)

        return pos

    def get_puck_starting_vel(self):
        self.seed_rngs(self.seed + 40)
        init_speed = random.random() * self.params["puck_init_speed_range"]
        vel = h.generate_random_2d_dir_vector()
        return vel * init_speed

    def get_puck_starting_rot_vel(self):
        self.seed_rngs(self.seed + 40)
        init_spin = (random.random() - 0.5) * 2 * self.params["puck_init_spin_range"]
        return init_spin

    def handle_goal_achieved(self):
        g.game.reset()

    def specific_update(self):
        if "goal" in self.events:
            self.seed += 1
