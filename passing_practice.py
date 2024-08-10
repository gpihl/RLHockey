from practice import Practice
import constants as c
import helpers as h
import numpy as np
import random
import globals as g

class PassingPractice(Practice):
    def __init__(self):
        super().__init__()
        self.init_params()
        self.consecutive_goal_req = 3
        self.consecutive_win_req = 7
        self.name = "passing"
        self.difficulty_increase = 0.02
        self.difficulty_alpha = 0.06
        c.settings["goal_1_blocked"] = True
        c.settings["goal_2_blocked"] = True
        c.settings["round_time"] = 10

        self.reward_structure = {
            "puck_proximity": 0.6,
            "shot_toward_team_mate": 1.6,
            "speed_dash": 0.3,
            "dash_shot": 0.3,
            "team_mate_puck_proximity": 1.0,
            "pre_pass_placement": 0.3,
        }

    def init_params(self):
        self.min_params = {
            "distance": 800,
            "angle": 0,
            "puck_dist_from_mid": h.field_height() * 0.40,
            "puck_init_speed_range": 0,
            "puck_init_spin_range": 0,
        }

        self.max_params = {
            "distance": h.max_dist(),
            "angle": 90,
            "puck_dist_from_mid": h.field_height() * 0.6,
            "puck_init_speed_range": 20,
            "puck_init_spin_range": 10,
        }

        super().update_params()

    def get_paddle_starting_pos(self, paddle, puck):
        self.seed_rngs(self.seed)
        random_dir = h.generate_random_2d_dir_vector()
        self.seed_rngs(self.seed + paddle.get_idx())

        if paddle.team == 1 and paddle.player == 1:
            pos = h.random_vector_within_cone(puck.pos, random_dir, self.min_params["distance"], self.params["distance"], self.params["angle"] * 2)
        elif paddle.team == 1 and paddle.player == 2:
            pos = h.random_vector_within_cone(puck.pos, -random_dir, self.min_params["distance"], self.params["distance"], self.params["angle"] * 2)
        else:
            pos = h.generate_random_2d_dir_vector() * 10000
            # pos = np.array([random.uniform(c.settings["field_width"] * 0.0 + 100, c.settings["field_width"] * 1.0 - 100),
            #                 random.uniform(c.settings["field_height"] * 0.0 + 100, c.settings["field_height"] * 1.0 - 100)],
            #                 dtype=np.float32)

        return pos

    def get_puck_starting_pos(self):
        self.seed_rngs(self.seed + 20)
        pos = h.random_vector_within_cone(h.field_mid(), np.array([1.0,0.0]), 0, self.params["puck_dist_from_mid"], 360)
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
        pass
        # if "goal" in self.events:
        #     self.seed += 1
