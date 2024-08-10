from abc import ABC, abstractmethod
import time
import random
import numpy as np
import helpers as h
import globals as g
import sys

class Practice(ABC):
    def __init__(self):
        self.seed = int(time.time())
        self.consecutive_wins = 0
        self.consecutive_goals = 0
        self.consecutive_goal_req = 0
        self.consecutive_win_req = 0
        self.consecutive_failures = 0
        self.consecutive_level_fails = 0
        self.collectable_reward = 0
        self.reward = 4000
        self.goal_reward = 1000
        self.level_reward = 16000
        self.reward_structure = dict()
        self.events = []
        self.name = ""
        self.params = dict()
        self.min_params = dict()
        self.max_params = dict()
        self.starting_difficulty = 0.0
        self.difficulty_alpha = 0.0
        self.difficulty_increase = 0.0
        self.max_failure_atempts = 30
        self.max_level_fails = 3

    def change_level(self):
        self.seed += 1
        self.specific_level_change()
        g.game.reset()

    def update(self):
        self.specific_update()

        if self.goal_achieved():
            print("goal achieved")
            self.collectable_reward += self.goal_reward
            self.consecutive_goals += 1
            if self.consecutive_goals == self.consecutive_goal_req:
                print("win achieved")
                self.collectable_reward += self.reward
                self.consecutive_wins += 1
                self.consecutive_goals = 0
                self.consecutive_level_fails = 0
                self.change_level()

            if self.consecutive_wins == self.consecutive_win_req:
                self.collectable_reward += self.level_reward
                self.increase_difficulty()
                self.consecutive_wins = 0
                self.consecutive_goals = 0

            self.consecutive_failures = 0
            self.handle_goal_achieved()
        elif self.goal_failed():
            print("goal failed")
            self.consecutive_goals = 0
            self.consecutive_wins = 0
            self.consecutive_failures += 1
            if self.consecutive_failures > self.max_failure_atempts:
                self.consecutive_level_fails += 1
                if self.consecutive_level_fails > self.max_level_fails:
                    self.decrease_difficulty()
                    self.consecutive_level_fails = 0

                self.change_level()
                self.consecutive_failures = 0

            self.handle_goal_failed()



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
            self.difficulty_alpha = self.starting_difficulty
            # sys.exit()

    def decrease_difficulty(self):
        print("decreasing difficulty")
        self.difficulty_alpha =  max(0.0, self.difficulty_alpha - 2 * self.difficulty_increase)
        self.update_params()

    def seed_rngs(self, seed=None):
        if seed is None:
            seed = self.seed

        random.seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def handle_goal_achieved(self):
        pass

    @abstractmethod
    def handle_goal_failed(self):
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

    @abstractmethod
    def specific_level_change(self):
        pass


