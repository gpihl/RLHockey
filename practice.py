from abc import ABC, abstractmethod
import globals as g
import constants as c
import time
import random
import numpy as np
import helpers as h
import sys

class Practice(ABC):
    def __init__(self):
        self.seed = int(time.time())
        self.consecutive_wins = 0
        self.consecutive_goals = 0
        self.consecutive_goal_req = 0
        self.consecutive_win_req = 0
        self.consecutive_failures = 0
        self.collectable_reward = 0
        self.reward = 4000
        self.reward_structure = dict()
        self.events = []
        self.name = ""
        self.params = dict()
        self.min_params = dict()
        self.max_params = dict()
        self.difficulty_alpha = 0.0
        self.difficulty_increase = 0.0

    def change_level(self):
        self.seed += 1

    def update(self):
        self.specific_update()

        if self.goal_achieved():
            self.collectable_reward = self.reward
            print("goal achieved")
            self.consecutive_goals += 1
            if self.consecutive_goals == self.consecutive_goal_req:
                print("win achieved")
                self.consecutive_wins += 1
                self.consecutive_goals = 0
                self.change_level()

            if self.consecutive_wins == self.consecutive_win_req:
                self.increase_difficulty()
                self.consecutive_wins = 0
                self.consecutive_goals = 0

            self.consecutive_failures == 0
            self.handle_goal_achieved()
        elif self.goal_failed():
            print("goal failed")
            self.consecutive_goals = 0
            self.consecutive_wins = 0
            self.consecutive_failures += 1
            if self.consecutive_failures > 200:
                self.change_level()
                self.consecutive_failures = 0



    def collect_reward(self):
        reward = self.collectable_reward
        self.collectable_reward = 0
        return reward

    def goal_achieved(self):
        if self.name in self.events:
            self.events = []
            print(self.difficulty_alpha)
            return True
        else:
            return False

    def goal_failed(self):
        if "round_end" in self.events:
            self.events = []
            return True
        else:
            return False

    def update_params(self):
        print("updating params")
        self.params = h.interpolate_dicts(self.min_params, self.max_params, self.difficulty_alpha)
        print(self.params)
        print(self.difficulty_alpha)

    def increase_difficulty(self):
        print("increasing difficulty")
        self.difficulty_alpha += self.difficulty_increase
        self.update_params()
        if self.difficulty_alpha > 1.0:
            print("Training finished, good job!")
            sys.exit()

    def seed_rngs(self, seed=None):
        if seed is None:
            seed = self.seed

        random.seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def handle_goal_achieved(self):
        pass

    @abstractmethod
    def get_paddle_starting_pos(self, paddle):
        pass

    @abstractmethod
    def get_puck_starting_pos(self):
        pass

    @abstractmethod
    def get_puck_starting_vel(self):
        pass

    @abstractmethod
    def get_puck_starting_rot_vel(self):
        pass

    @abstractmethod
    def specific_update(self):
        pass


