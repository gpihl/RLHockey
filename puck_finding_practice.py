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
        self.starting_difficulty = 0.5
        self.difficulty_alpha = self.starting_difficulty
        c.settings["goal_1_blocked"] = True
        c.settings["goal_2_blocked"] = True
        c.settings["round_time"] = 1.2

        self.reward_structure = {
            # "puck_proximity": 4.0,
            # "speed_dash": 8.0,
            # "other_paddles_prox": -4.0,
        }
        self.specific_level_change()

    def init_params(self):
        self.min_params = {
            "puck_distance": 60,
            "puck_init_speed_range": 0,
            "puck_init_spin_range": 0,
            "other_paddles_speed_range": 0,
        }

        self.max_params = {
            "puck_distance": h.field_width(),
            "puck_init_speed_range": 75,
            "puck_init_spin_range": 70,
            "other_paddles_speed_range": 1.0,
        }

        super().update_params()

    def get_paddle_starting_pos(self, paddle, puck):
        self.seed_rngs(self.seed + paddle.get_idx())

        if paddle.team == 1 and paddle.player == 1:
            pos = h.random_vector_within_cone(puck.pos, np.array([1.0, 0]), 100, self.params["puck_distance"], 360)
        else:
            while True:
                pos = np.array([random.uniform(c.settings["field_width"] * 0.0 + 100, c.settings["field_width"] * 1.0 - 100),
                                random.uniform(c.settings["field_height"] * 0.0 + 100, c.settings["field_height"] * 1.0 - 100)],
                                dtype=np.float32)

                dist_to_agent = np.linalg.norm(g.game.paddles_1[0].pos - pos)

                if dist_to_agent > 500:
                    break

        return pos

    def get_puck_starting_pos(self):
        self.seed_rngs(self.seed + 20)
        pos = np.array([random.uniform(c.settings["field_width"] * 0.0 + 100, c.settings["field_width"] * 1.0 - 100),
                        random.uniform(c.settings["field_height"] * 0.0 + 100, c.settings["field_height"] * 1.0 - 100)],
                        dtype=np.float32)

        return pos

    def get_puck_starting_vel(self, puck):
        self.seed_rngs(self.seed + 40)
        init_speed = random.random() * self.params["puck_init_speed_range"]
        vel = h.generate_random_2d_dir_vector()
        return vel * init_speed

    def get_puck_starting_rot_vel(self):
        self.seed_rngs(self.seed + 40)
        init_spin = (random.random() - 0.5) * 2 * self.params["puck_init_spin_range"]
        return init_spin

    def handle_goal_achieved(self):
        time_bonus = g.game.time_left() * 800
        self.collectable_reward += time_bonus

    def handle_goal_failed(self):
        dist_to_puck = np.linalg.norm(g.game.paddles_1[0].pos - g.game.puck.pos)
        prox_reward = h.map_value_to_range(dist_to_puck, 0, h.max_dist()) * 1000
        prox_reward = max(0.0, prox_reward)
        self.collectable_reward += prox_reward
        print("Prox reward: " + str(int(prox_reward)))

    def specific_update(self):
        if "goal" in self.events:
            self.seed += 1

        # if "paddle_collision" in self.events:
        #     # self.collectable_reward -= 1000
        #     self.events.append("round_end")
        #     g.game.reset()

    def specific_level_change(self):
        c.settings["random_paddle_speed"] = self.params["other_paddles_speed_range"]

    def goal_failed(self):
        if "round_end" in self.events:
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
