import numpy as np
import helpers as h
import globals as g

class Reward:
    # rewards_start = {
    #     "velocity": -0.03,
    #     "goal": 1000,
    #     "team_mate_proximity": 0.0,
    #     "wrong_side_of_puck": -0.0,
    #     "puck_proximity": 0.0,
    #     "puck_vel_toward_goal": 0.0,
    #     "goal_puck_proximity": 0.,
    #     "shot": 0.0,
    #     "shot_toward_goal": 0.0,
    #     "dash": 0.0,
    #     "defender_goal_prox": 0.0,
    #     "defensive_positioning": 0.0,
    # }

    # rewards_start = {"velocity": -0.01, "goal": 1000, "team_mate_proximity": 0.8, "wrong_side_of_puck": -0.1, "puck_proximity": 1.3, "puck_vel_toward_goal": 0.05, "goal_puck_proximity": 0.2, "shot": 0.0, "shot_toward_goal": 7.0, "dash": 150.0, "defender_goal_prox": 0.3, "defensive_positioning": 0.5}

    rewards_start = {
        "velocity": -0.01,
        "goal": 1000,
        "team_mate_proximity": 1.0,
        "wrong_side_of_puck": -0.8,
        "puck_proximity": 3.0,
        "puck_vel_toward_goal": 0.07,
        "goal_puck_proximity": 0.2,
        "shot": 0.0,
        "shot_toward_goal": 4.0,
        "dash": 20.0,
        "defender_goal_prox": 1.0,
        "defensive_positioning": 1.0
        }

    rewards_end = None

    # rewards_end = {
    #     "velocity": -0.03,
    #     "goal": 1000,
    #     "team_mate_proximity": 0.0,
    #     "wrong_side_of_puck": -0.0,
    #     "puck_proximity": 0.0,
    #     "puck_vel_toward_goal": 0.0,
    #     "goal_puck_proximity": 0.0,
    #     "shot": 0.0,
    #     "shot_toward_goal": 0.0,
    #     "dash": 0.0,
    #     "defender_goal_prox": 0.0,
    #     "defensive_positioning": 0.0,
    # }


    # bra!
    # rewards_start = {
    #     "time_reward": -0.0,
    #     "velocity": -0.0,
    #     "goal": 1000,
    #     "team_mate_proximity": -0.0,
    #     "wrong_side_of_puck": -0.01,
    #     "puck_proximity": 0.5,
    #     "puck_vel_toward_goal": 0.0,
    #     "goal_puck_proximity": 0.0,
    #     "shot": 5.0,
    #     "shot_toward_goal": 2.0,
    #     "dash": 0.0,
    # }

    rewards = rewards_start
    time_to_reach_end_reward = 60 * 7 * 2 * 10

    def __init__(self, paddle, puck, team_mates):
        self.paddle = paddle
        self.puck = puck
        self.team_mates = list(filter(lambda x: x.agent_control != "off", team_mates))
        self.reward_breakdown = { "total": (0, 0) }
        self.action = None
        self.scorer = None

    def reset(self):
        for key, _ in self.reward_breakdown.items():
            self.reward_breakdown[key] = (0, 0)

        self.action = None
        self.scorer = None

    @staticmethod
    def update():
        if Reward.rewards_end is None:
            return

        t = g.current_time / Reward.time_to_reach_end_reward
        Reward.rewards = h.interpolate_dicts(Reward.rewards_start, Reward.rewards_end, t)

    def calculate_total_reward(self, action, scorer):
        if self.paddle.agent_control == "off":
            return 0

        self.action = action
        self.scorer = scorer

        self.reward_breakdown["total"] = (self.reward_breakdown["total"][0], 0)

        rewards = [
            (self.velocity, "velocity"),
            (self.goal, "goal"),
            (self.team_mate_proximity, "team_mate_proximity"),
            (self.wrong_side_of_puck, "wrong_side_of_puck"),
            (self.puck_proximity, "puck_proximity"),
            (self.puck_vel_toward_goal, "puck_vel_toward_goal"),
            (self.goal_puck_proximity, "goal_puck_proximity"),
            (self.shot, "shot"),
            (self.shot_toward_goal, "shot_toward_goal"),
            (self.dash, "dash"),
            (self.defender_goal_prox, "defender_goal_prox"),
            (self.defensive_positioning, "defensive_positioning")
        ]

        total_reward = sum([self.calculate_specific_reward(reward_fn, reward_name) for reward_fn, reward_name in rewards])
        return total_reward

    def register_reward(self, reward, name):
        if name not in self.reward_breakdown:
            self.reward_breakdown[name] = (0, 0)

        self.reward_breakdown[name] = (self.reward_breakdown[name][0] + reward, reward)
        self.reward_breakdown["total"] = (self.reward_breakdown["total"][0] + reward, reward)

    def calculate_specific_reward(self, reward_fn, reward_name):
        specific_reward = reward_fn() * Reward.rewards[reward_name]
        self.register_reward(specific_reward, reward_name)
        return specific_reward

    def velocity(self):
        reward = np.linalg.norm(self.paddle.vel)
        return reward

    def goal(self):
        reward = 0
        if self.scorer == self.paddle.team:
            reward = 1
        elif self.scorer != 0:
            reward = -1

        return reward

    def puck_proximity(self):
        dist_to_puck = min([np.linalg.norm(self.puck.pos - team_player.pos) for team_player in self.team_mates + [self.paddle]])
        reward = h.map_value_to_range(dist_to_puck, 0, h.max_dist() / 2)
        # reward = (dist_to_puck / h.field_width())
        return reward

    def team_mate_proximity(self):
        if len(self.team_mates) == 0:
            return 0

        closest_team_mate_dist = min([np.linalg.norm(team_mate.pos - self.paddle.pos) for team_mate in self.team_mates])
        optimal_team_mate_dist = h.field_height() * 0.5
        optimal_diff = np.abs(optimal_team_mate_dist - closest_team_mate_dist)
        reward = h.map_value_to_range(optimal_diff, 0, h.field_height() * 0.5)
        return reward

    def wrong_side_of_puck(self):
        puck_delta_x = self.puck.pos[0] - self.paddle.pos[0]
        if self.paddle.team == 1:
            reward = 1 if puck_delta_x < 0 else -1
        else:
            reward = 1 if puck_delta_x > 0 else -1
        return reward

    def goal_puck_proximity(self):
        goal_pos = h.goal_pos(2) if self.paddle.team == 1 else h.goal_pos(1)
        puck_to_goal_dist = np.linalg.norm(goal_pos - self.puck.pos)
        reward = ((h.field_width() - puck_to_goal_dist) / h.field_width())
        return reward

    def puck_vel_toward_goal(self):
        goal_pos = h.goal_pos(2) if self.paddle.team == 1 else h.goal_pos(1)
        puck_to_goal_dir = (goal_pos - self.puck.pos) / np.linalg.norm(goal_pos - self.puck.pos)
        reward = np.dot(self.puck.vel, puck_to_goal_dir)
        return reward

    def pointless(self):
        acceleration = self.action["acceleration"]
        reward = self.paddle.pointless_motion(acceleration)
        return reward

    def dash(self):
        reward = 0
        if self.paddle.collect_dash_reward() > 0:
            reward -= 1

        reward += self.paddle.collect_dash_shot_reward() * 6
        # if reward != 0:
        #     print(reward)
        return reward

    def shot_toward_goal(self):
        reward = self.puck.collect_shot_reward("shot_toward_goal", self.paddle)
        return reward

    def shot(self):
        reward = self.puck.collect_shot_reward("shot", self.paddle)
        return reward

    def defender_goal_prox(self):
        goal_pos = h.goal_pos(self.paddle.team)
        team_paddles = self.team_mates + [self.paddle]
        closest_dist_from_goal = min([np.linalg.norm(paddle.pos - goal_pos) for paddle in team_paddles])
        reward = h.map_value_to_range(closest_dist_from_goal, h.field_height() / 4, h.max_dist() * 0.7)
        return reward

    def defensive_positioning(self):
        goal_pos = h.goal_pos(self.paddle.team)
        team_paddles = self.team_mates + [self.paddle]
        min_dist = h.max_dist()
        closest_paddle = None
        for paddle in team_paddles:
            dist_from_goal = np.linalg.norm(paddle.pos - goal_pos)
            if dist_from_goal < min_dist:
                min_dist = dist_from_goal
                closest_paddle = paddle

        if closest_paddle.player == self.paddle.player:
            reward = 1
            dist_from_optimal = h.point_to_line_distance(goal_pos, self.puck.pos, closest_paddle.pos)
            reward = h.map_value_to_range(dist_from_optimal, 0, h.field_height())
        else:
            reward = 0

        return reward






