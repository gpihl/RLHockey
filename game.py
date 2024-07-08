import numpy as np
import math
import globals as g
import constants as c
import helpers as h
import argparse
from paddle import Paddle
from puck import Puck
import cProfile
from stats import Stats

class Game:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Game, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ai=False):
        if self._initialized:
            self.total_steps = 0
            return

        self._initialized = True
        self.total_steps = 0
        self.prev_t = g.current_time
        self.curr_t = g.current_time
        self.last_scorer = 2
        self.paddles_1 = []
        self.paddles_2 = []
        self.reward_breakdown_1 = {"total": 0}
        self.reward_breakdown_2 = {"total": 0}
        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.start_time = 0
        self.total_reward = 0.0
        self.score = [0, 0]
        self.stats = Stats()
        self.match_steps = 60 * c.settings["fps"]
        self.create_objects()
        # g.controls.stick = g.controls.init_controls()
        self.reset()
        self.max_paddle_speed = 0
        self.max_puck_speed = 0
        self.max_puck_spin = 0
        self.player_1_observation = {}
        print("Game initialization done")

    def create_objects(self):
        for i in range(c.settings["team_size"]):
            paddle = Paddle(1, i+1)
            paddle.team_mates = c.settings["team_size"]
            self.paddles_1.append(paddle)

        for i in range(c.settings["team_size"]):
            paddle = Paddle(2, i+1)
            paddle.team_mates = c.settings["team_size"]
            self.paddles_2.append(paddle)

        g.paddles = self.paddles_1 + self.paddles_2

        self.puck = Puck()

    def reset(self):
        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.start_time = g.current_time

        for key, _ in self.reward_breakdown_1.items():
            self.reward_breakdown_1[key] = 0

        for key, _ in self.reward_breakdown_2.items():
            self.reward_breakdown_2[key] = 0

        for paddle in self.paddles_1 + self.paddles_2:
            paddle.reset()
        self.puck.reset(self.last_scorer)
        self.stats = Stats()
        g.sound_handler.reset()
        g.field.reset()

    def register_reward(self, reward, name, team):
        if team == 1:
            if name in self.reward_breakdown_1:
                self.reward_breakdown_1[name] += reward / c.rewards["normalization"]
                self.reward_breakdown_1["total"] += reward / c.rewards["normalization"]
            else:
                self.reward_breakdown_1[name] = 0
        else:
            if name in self.reward_breakdown_2:
                self.reward_breakdown_2[name] += reward / c.rewards["normalization"]
                self.reward_breakdown_2["total"] += reward / c.rewards["normalization"]
            else:
                self.reward_breakdown_2[name] = 0


    def get_reward(self, paddle, team_mates, action, scorer):
        if paddle.model is None:
            return 0

        acceleration = action["acceleration"]
        reward = 0

        vel_reward = np.linalg.norm(paddle.vel) * c.rewards["vel_reward"]
        self.register_reward(vel_reward, "vel_reward", paddle.team)
        reward += vel_reward

        if scorer == paddle.team:
            print("goal reward")
            goal_reward = c.rewards["goal"]
            self.register_reward(goal_reward, "goal", paddle.team)
            reward += goal_reward
        elif scorer != 0:
            print("negative goal reward")
            goal_reward = -c.rewards["goal"]
            self.register_reward(goal_reward, "goal", paddle.team)
            reward += goal_reward

        dist_to_puck = min([np.linalg.norm(self.puck.pos - team_player.pos) for team_player in team_mates + [paddle]])
        proximity_reward = ((c.settings["field_width"] - dist_to_puck) / c.settings["field_width"]) * c.rewards["puck_proximity"]
        self.register_reward(proximity_reward, "puck_proximity", paddle.team)
        reward += proximity_reward

        closest_team_mate_dist = min([np.linalg.norm(team_mate.pos - paddle.pos) for team_mate in team_mates])
        team_mate_proximity_reward = ((c.settings["field_width"] - closest_team_mate_dist) / c.settings["field_width"]) * c.rewards["team_mate_proximity"]
        self.register_reward(team_mate_proximity_reward, "team_mate_proximity", paddle.team)
        reward += team_mate_proximity_reward

        puck_delta_x = self.puck.pos[0] - paddle.pos[0]
        if paddle.team == 1:
            wrong_side_of_puck_reward = c.rewards["wrong_side_of_puck"] if puck_delta_x < 0 else -c.rewards["wrong_side_of_puck"]
        else:
            wrong_side_of_puck_reward = c.rewards["wrong_side_of_puck"] if puck_delta_x > 0 else -c.rewards["wrong_side_of_puck"]
        self.register_reward(wrong_side_of_puck_reward, "wrong_side_of_puck", paddle.team)
        reward += wrong_side_of_puck_reward

        goal_pos = h.goal_pos(2) if paddle.team == 1 else h.goal_pos(1)
        puck_to_goal = goal_pos - self.puck.pos
        puck_to_goal_dist = np.linalg.norm(puck_to_goal)
        puck_to_goal_reward = ((c.settings["field_width"] - puck_to_goal_dist) / c.settings["field_width"]) * c.rewards["goal_puck_proximity"]
        self.register_reward(puck_to_goal_reward, "goal_puck_proximity", paddle.team)
        reward += puck_to_goal_reward

        puck_to_goal_dir = puck_to_goal / puck_to_goal_dist
        puck_vel_toward_goal = (np.dot(self.puck.vel, puck_to_goal_dir) + np.linalg.norm(self.puck.vel))* c.rewards["puck_vel_toward_goal"]
        self.register_reward(puck_vel_toward_goal, "puck_vel_toward_goal", paddle.team)
        reward += puck_vel_toward_goal

        # pointless_reward = paddle.pointless_motion(acceleration) * c.rewards["pointless_motion"]
        # self.register_reward(pointless_reward, "pointless_motion", paddle.team)
        # reward += pointless_reward

        dash_reward = paddle.collect_dash_reward() ** 2
        dash_reward = dash_reward * c.rewards["dash"]
        self.register_reward(dash_reward, "dash", paddle.team)
        reward += dash_reward

        shot_toward_goal_reward = self.puck.collect_shot_reward("shot_toward_goal", paddle) * c.rewards["shot_toward_goal"]
        self.register_reward(shot_toward_goal_reward, "shot_toward_goal", paddle.team)
        reward += shot_toward_goal_reward

        shot_reward = self.puck.collect_shot_reward("shot", paddle) * c.rewards["shot"]
        self.register_reward(shot_reward, "shot", paddle.team)
        reward += shot_reward

        reward /= c.rewards["normalization"]

        return reward

    def non_player_1_team_1_paddles(self):
        return list(filter(lambda x: not (x.player == 1 and x.team == 1), self.paddles_1 + self.paddles_2))

    def step_training(self, player_1_model_action):
        running = g.framework.handle_events()
        if not running:
            exit()

        player_1_model_action = self.paddles_1[0].model.process_action(player_1_model_action)
        player_1_action = g.controls.game_action_from_model_action(player_1_model_action)
        self.player_1_observation = self.get_observation(1, 1)

        team_1_actions = [player_1_action]
        for paddle in list(filter(lambda x: not (x.player == 1 and x.team == 1), self.paddles_1)):
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_1_actions.append(action)

        team_2_actions = []
        for paddle in self.paddles_2:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_2_actions.append(action)

        scorer = self.update(team_1_actions, team_2_actions)
        reward = self.handle_rewards(team_1_actions, team_2_actions, scorer)

        return self.player_1_observation, reward, self.is_done(scorer), {}

    def step(self):
        self.curr_t = g.current_time
        delta_t = self.curr_t - self.prev_t
        c.settings["delta_t"] = min(92 * delta_t, 3)
        self.prev_t = self.curr_t

        team_1_actions = []
        for paddle in self.paddles_1:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_1_actions.append(action)

        team_2_actions = []
        for paddle in self.paddles_2:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_2_actions.append(action)

        scorer = self.update(team_1_actions, team_2_actions)

        if scorer != 0:
            self.goal_scored_sequence(scorer)

        return self.is_done(scorer)

    def update(self, team_1_actions, team_2_actions):
        self.handle_game_paused()
        self.current_step += 1
        self.total_steps += 1

        for idx, action in enumerate(team_1_actions):
            self.paddles_1[idx].update(self.puck, action)

        for idx, action in enumerate(team_2_actions):
            self.paddles_2[idx].update(self.puck, action)

        all_paddles = self.paddles_1 + self.paddles_2
        for i in range(len(all_paddles)):
            for j in range(i+1, len(all_paddles)):
                all_paddles[i].handle_collision(all_paddles[j])

        self.puck.update(all_paddles)

        scorer = 0
        if g.field.team_1_scored(self.puck):
            scorer = 1
            self.last_scorer = 1
            self.score[0] += 1
        elif g.field.team_2_scored(self.puck):
            scorer = 2
            self.last_scorer = 2
            self.score[1] += 1

        g.sound_handler.update(scorer)
        g.field.update(self.puck)
        if not c.settings["is_training"]:
            g.framework.update_particles()
            g.framework.update_paddle_data(all_paddles)

            g.framework.update_object_data(self.paddles_1 + self.paddles_2 + [self.puck])
            paddle_lights = list(map(lambda x: x.light, all_paddles))
            g.framework.update_light_data(g.field.lights + paddle_lights + [self.puck.light])

        self.handle_rendering()
        g.framework.tick()
        # self.print_max_speeds()

        return scorer

    def handle_rewards(self, team_1_actions, team_2_actions, scorer):
        for idx, paddle in enumerate(self.paddles_1):
            paddle.current_reward = self.get_reward(paddle, list(filter(lambda x: x.player != paddle.player, self.paddles_1)), team_1_actions[idx], scorer)

        for idx, paddle in enumerate(self.paddles_2):
            paddle.current_reward = self.get_reward(paddle, list(filter(lambda x: x.player != paddle.player, self.paddles_2)), team_2_actions[idx], scorer)

        team1_reward = sum([paddle.current_reward for paddle in self.paddles_1])
        team2_reward = sum([paddle.current_reward for paddle in self.paddles_2])
        reward = team1_reward - team2_reward

        time_reward = c.rewards["time_reward"]
        self.register_reward(time_reward, "time_reward", 1)
        self.register_reward(time_reward, "time_reward", 2)
        reward += time_reward

        self.current_reward = reward
        self.round_reward += reward
        self.total_reward += reward
        self.stats.update(team1_reward - team2_reward - time_reward)
        self.stats.update(team2_reward - team1_reward - time_reward)
        return reward

    def handle_rendering(self):
        if not c.settings["no_render"]:
            if c.settings["is_training"] and (not g.framework.fps_locked):
                if self.current_step % 10 == 0:
                    self.render()
            else:
                self.render()
        else:
            if self.current_step % 60 == 0:
                g.framework.rendering_off_message()

    def handle_game_paused(self):
        i = 0
        while c.settings["paused"]:
            g.clock.pause()
            i += 1
            running = g.framework.handle_events()
            if not running:
                exit()

            if i % 60 == 0:
                g.framework.paused()

        g.clock.unpause()

    def print_max_speeds(self):
        self.max_puck_speed = max(self.max_puck_speed, np.linalg.norm(self.puck.vel))
        for paddle in self.paddles_1 + self.paddles_2:
            self.max_paddle_speed = max(self.max_paddle_speed, np.linalg.norm(paddle.vel))

        self.max_puck_spin = max(self.max_puck_spin, np.abs(self.puck.rot_vel))

        if self.current_step % 60 == 0:
            print(f"puck_speed: {self.max_puck_speed}")
            print(f"puck_spin: {self.max_puck_spin}")
            print(f"paddle_speed: {self.max_paddle_speed}")


    def goal_scored_sequence(self, scorer):
        if not c.settings["is_training"]:
            goal_time = g.current_time
            scorer = self.paddles_1[0] if scorer == 1 else self.paddles_2[0]
            position = h.field_mid()
            radius = c.settings["field_height"] / 5.4
            g.framework.begin_drawing()
            scorer.draw_paddle(position, radius, scorer.color, draw_indicator=False)
            g.framework.end_drawing()
            prev_time = None
            while g.current_time - goal_time < 0.9:
                if g.current_time - goal_time > 0.65:
                    if prev_time is None:
                        prev_time = g.current_time
                    delta_t = g.current_time - prev_time
                    prev_time = g.current_time
                    opacity = int(1400 * delta_t)
                    g.framework.fill_screen_semiopaque_black(opacity)

                g.framework.tick()

    def scale_absolute_position(self, pos):
        centered_origin_pos = pos - h.field_mid()
        scaled_pos = np.array([centered_origin_pos[0] / (c.settings["field_width"] / 2), centered_origin_pos[1] / (c.settings["field_height"] / 2)])
        return scaled_pos

    # Absolute positions
    def get_observation(self, team, player):
        player = player - 1
        max_puck_speed = 60
        max_paddle_speed = 110
        max_puck_spin = 130

        if team == 1:
            charging_alpha = -1 if not self.paddles_1[player].charging_dash else self.paddles_1[player].charging_alpha() * 2 - 1
            charging_alpha = np.array([charging_alpha])

            obs = {
                "self_pos":         self.scale_absolute_position(self.paddles_1[player].pos),
                "puck_pos":         self.scale_absolute_position(self.puck.pos),
                "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
                "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
                "charging_alpha":   charging_alpha,
                "goal_1_top_pos":   self.scale_absolute_position(h.goal_top_pos(1)),
                "goal_1_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(1)),
                "goal_2_top_pos":   self.scale_absolute_position(h.goal_top_pos(2)),
                "goal_2_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(2)),
            }

            other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_1))
            all_paddles_on_team = [self.paddles_1[player]] + other_paddles_on_team

            players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(other_paddles_on_team) }
            players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(self.paddles_2) }
            players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
            players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_2) }

            obs |= {
                **players_positions_team_1,
                **players_positions_team_2,
                **players_velocities_team_1,
                **players_velocities_team_2
            }

        elif team == 2:
            charging_alpha = -1 if not self.paddles_2[player].charging_dash else self.paddles_2[player].charging_alpha() * 2 - 1
            charging_alpha = np.array([charging_alpha])

            obs = {
                "self_pos":         self.scale_absolute_position(self.paddles_2[player].pos),
                "puck_pos":         self.scale_absolute_position(self.puck.pos),
                "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
                "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
                "charging_alpha":   charging_alpha,
                "goal_1_top_pos":   self.scale_absolute_position(h.goal_top_pos(2)),
                "goal_1_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(2)),
                "goal_2_top_pos":   self.scale_absolute_position(h.goal_top_pos(1)),
                "goal_2_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(1)),
            }

            other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_2))
            all_paddles_on_team = [self.paddles_2[player]] + other_paddles_on_team

            players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(other_paddles_on_team) }
            players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(self.paddles_1) }
            players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
            players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_1) }

            obs |= {
                **players_positions_team_1,
                **players_positions_team_2,
                **players_velocities_team_1,
                **players_velocities_team_2
            }

            obs = { k: v if v.size == 1 else np.array([-v[0], v[1]]) for k, v in obs.items() }

        return obs

    # Relative positions
    # def get_observation(self, team, player):
    #     player = player - 1
    #     max_puck_speed = 60
    #     max_paddle_speed = 110
    #     max_puck_spin = 130

    #     if team == 1:
    #         charging_alpha = -1 if not self.paddles_1[player].charging_dash else self.paddles_1[player].charging_alpha() * 2 - 1
    #         charging_alpha = np.array([charging_alpha])

    #         obs = {
    #             "puck_pos":         self.paddles_1[player].get_relative_pos_of_puck_obs(self.puck),
    #             "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
    #             "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
    #             "charging_alpha":   charging_alpha,
    #             "goal_1_top_pos":   self.paddles_1[player].get_relative_pos_of_goal_1_top(),
    #             "goal_1_bot_pos":   self.paddles_1[player].get_relative_pos_of_goal_1_bot(),
    #             "goal_2_top_pos":   self.paddles_1[player].get_relative_pos_of_goal_2_top(),
    #             "goal_2_bot_pos":   self.paddles_1[player].get_relative_pos_of_goal_2_bot(),
    #         }

    #         other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_1))
    #         all_paddles_on_team = [self.paddles_1[player]] + other_paddles_on_team

    #         players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.paddles_1[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(other_paddles_on_team) }
    #         players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.paddles_1[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(self.paddles_2) }
    #         players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
    #         players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_2) }

    #         obs |= {
    #             **players_positions_team_1,
    #             **players_positions_team_2,
    #             **players_velocities_team_1,
    #             **players_velocities_team_2
    #         }

    #     elif team == 2:
    #         charging_alpha = -1 if not self.paddles_2[player].charging_dash else self.paddles_2[player].charging_alpha() * 2 - 1
    #         charging_alpha = np.array([charging_alpha])

    #         obs = {
    #             "puck_pos":         self.paddles_2[player].get_relative_pos_of_puck_obs(self.puck),
    #             "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
    #             "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
    #             "charging_alpha":   charging_alpha,
    #             "goal_1_top_pos":   self.paddles_2[player].get_relative_pos_of_goal_2_top(),
    #             "goal_1_bot_pos":   self.paddles_2[player].get_relative_pos_of_goal_2_bot(),
    #             "goal_2_top_pos":   self.paddles_2[player].get_relative_pos_of_goal_1_top(),
    #             "goal_2_bot_pos":   self.paddles_2[player].get_relative_pos_of_goal_1_bot(),
    #         }

    #         other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_2))
    #         all_paddles_on_team = [self.paddles_2[player]] + other_paddles_on_team

    #         players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.paddles_2[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(other_paddles_on_team) }
    #         players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.paddles_2[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(self.paddles_1) }
    #         players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
    #         players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_1) }

    #         obs |= {
    #             **players_positions_team_1,
    #             **players_positions_team_2,
    #             **players_velocities_team_1,
    #             **players_velocities_team_2
    #         }

    #         obs = { k: v if v.size == 1 else np.array([-v[0], v[1]]) for k, v in obs.items() }

    #     return obs

    def render(self):
        g.framework.begin_drawing()
        g.field.draw_bottom_layer(self.puck)
        self.puck.draw()
        if c.settings["is_training"]:
            reward_alpha_1 = self.get_reward_alpha(self.paddles_1, self.paddles_2)
            for paddle in self.paddles_1:
                paddle.draw(reward_alpha=reward_alpha_1)
            reward_alpha_2 = self.get_reward_alpha(self.paddles_2, self.paddles_1)
            for paddle in self.paddles_2:
                paddle.draw(reward_alpha=reward_alpha_2)
        else:
            for paddle in self.paddles_1 + self.paddles_2:
                paddle.draw()
        g.field.draw_top_layer(self.puck)
        if not c.settings["is_training"]:
            g.framework.draw_particles()
        self.draw_ui()
        g.framework.end_drawing()

    def get_reward_alpha(self, paddles, other_paddles):
        mean, std = self.stats.get_stats()
        if std == 0:
            return

        p = ((sum([paddle.current_reward for paddle in paddles]) - sum([other_paddle.current_reward for other_paddle in other_paddles])) - mean) / std
        alpha = (p / 4)  + 0.5
        alpha = max(0.0, min(1.0, p))
        return alpha

    def draw_ui(self):
        g.ui.draw_time_left(self.seconds_left())
        g.ui.draw_score(self.score, self.paddles_1[0], self.paddles_2[0])
        # g.framework.draw_fps(0,0)
        if c.settings["is_training"]:
            g.ui.draw_steps_left(str(self.total_training_steps_left()))
            g.ui.draw_reward_breakdown(self.reward_breakdown_1, self.reward_breakdown_2)
            g.ui.draw_observation(self.player_1_observation)

    def total_training_steps_left(self):
        return c.training["training_steps"] - self.total_steps

    def seconds_left(self):
        if c.settings["is_training"]:
            seconds_left = math.ceil((self.match_steps - self.current_step) / c.settings["fps"])
        else:
            # seconds_left = math.ceil((self.match_steps - self.current_step) / c.settings["fps"])
            seconds_left = math.ceil((self.match_steps / c.settings["fps"]) - (g.current_time - self.start_time))
        return seconds_left

    def is_done(self, scorer):
        if c.settings["is_training"]:
            return self.current_step > self.match_steps or scorer != 0
        else:
            return self.seconds_left() < 0 or scorer != 0

    def close(self):
        g.framework.close()

def main():
    g.initialize()

    if c.settings["is_training"]:
        training_model = g.game.paddles_1[0].model
        non_training_paddles = g.game.non_player_1_team_1_paddles()
        while True:
            total_training_steps = c.training["training_steps"]
            # total_training_steps = 1000
            training_model.train_model(total_training_steps)
            training_model.save_model()
            for paddle in non_training_paddles:
                paddle.load_new_model()

            g.game.total_steps = 0
    else:
        running = True
        while running:
            running = g.framework.handle_events()
            done = g.game.step()

            if done:
                g.game.reset()

        g.game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the game with optional profiling.")
    parser.add_argument("-p", "--profile", action="store_true", help="Run the game with profiling enabled")
    parser.add_argument("-a", "--ai", action="store_true", help="AI vs AI")
    parser.add_argument("-t", "--training", action="store_true", help="Perform training of RL agents")
    parser.add_argument("-n", "--number", type=int, default=2, help="Number of players per team")
    args = parser.parse_args()
    c.settings["team_size"] = args.number
    c.settings["is_training"] = args.training
    c.settings["no_sound"] = args.training
    c.settings["all_ai"] = args.ai

    if args.profile:
        print("Running game with profiling...")
        cProfile.run("main()", "profile_output.prof")
        print("Profiling complete. Results saved to profile_output.prof")
        print("You can visualize the results using snakeviz: snakeviz profile_output.prof")
    else:
        main()
        # main(args.ai)
