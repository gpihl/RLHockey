import torch
import numpy as np
import time
import math
import globals as g
import constants as c
import helpers as h
import argparse
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
import environment
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

    def __init__(self):
        if self._initialized:
            self.total_steps = 0
            return

        self._initialized = True
        self.total_steps = 0
        self.prev_t = time.time()
        self.curr_t = time.time()
        self.last_scorer = 2
        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.total_reward = 0.0
        self.player_2_model = None
        self.player_1_model = None
        self.score = [0, 0]
        self.background_color = g.sound_handler.current_color()
        self.stats = Stats()
        self.match_steps = 60 * c.settings['fps']
        self.create_objects()
        self.reset()

        print("Game initialization done")

    def create_objects(self):
        self.paddle1 = Paddle(1)
        self.paddle2 = Paddle(2)
        self.puck = Puck()

    def reset(self):
        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.paddle1.reset()
        self.paddle2.reset()
        self.puck.reset(self.last_scorer)
        self.stats = Stats()
        g.sound_handler.reset()

    def get_reward(self, paddle, action):
        acceleration = action['acceleration']
        reward = 0
        reward += c.rewards["time_reward"]

        reward += np.linalg.norm(acceleration) * c.rewards["acc_reward"]

        if (self.player_1_scored() and paddle.player == 1) or (self.player_2_scored() and paddle.player == 2):
            reward += c.rewards["goal"]

        relative_pos = self.puck.pos - paddle.pos
        dist_to_puck = np.linalg.norm(relative_pos)
        proximity_reward = ((c.settings['field_width'] - dist_to_puck) / c.settings['field_width']) * c.rewards["puck_proximity"]
        reward += proximity_reward

        goal_pos = h.goal_pos(2) if paddle.player == 1 else h.goal_pos(1)
        puck_to_goal_dist = np.linalg.norm(goal_pos - self.puck.pos)
        puck_to_goal_reward = ((c.settings['field_width'] - puck_to_goal_dist) / c.settings['field_width']) * c.rewards["goal_puck_proximity"]
        reward += puck_to_goal_reward

        pointless_reward = paddle.pointless_motion(acceleration) * c.rewards["pointless_motion"]
        reward += pointless_reward

        reward += self.puck.collect_shot_reward('shot_toward_goal', player=paddle.player) * c.rewards["shot_toward_goal"]

        reward += self.puck.collect_shot_reward('shot', player=paddle.player) * c.rewards["shot"]

        reward /= c.rewards["normalization"]

        return reward

    def step(self, action=None):
        if c.settings['is_training']:
            g.framework.handle_keyboard_input(g.framework.get_events())

        if self.player_1_model is not None:
            observation = self.get_observation(1)
            player_1_model_action = self.player_1_model.predict(observation)[0]
        else:
            player_1_model_action = action

        player_1_action = None
        player_2_action = None

        human_action = g.controls.get_human_action()

        if player_1_model_action is not None:
            player_1_action = g.controls.game_action_from_model_action(player_1_model_action)
        else:
            player_1_action = human_action

        if c.training['player_2_active']:
            if self.player_2_model is not None and not c.settings['player_2_human']:
                observation = self.get_observation(2)
                player_2_model_action = self.player_2_model.predict(observation)[0]
                player_2_action = g.controls.game_action_from_model_action(player_2_model_action)

            elif c.settings['is_training']:
                player_2_action = human_action

        if player_2_action is None:
            player_2_action = g.controls.empty_action()

        self.prev_t = self.curr_t
        self.curr_t = g.current_time
        self.current_step += 1
        self.total_steps += 1

        g.sound_handler.update()

        self.paddle1.update(self.puck, player_1_action)
        self.paddle2.update(self.puck, player_2_action)
        self.paddle1.handle_collision(self.paddle2)
        self.puck.update([self.paddle1, self.paddle2])

        if self.player_1_scored():
            self.last_scorer = 1
            self.score[0] += 1
        elif self.player_2_scored():
            self.last_scorer = 2
            self.score[1] += 1

        if not c.settings['no_render']:
            self.render()

        g.framework.tick()

        scored_1 = self.player_1_scored()
        scored_2 = self.player_2_scored()

        if scored_1 or scored_2:
            if scored_1:
                g.sound_handler.play_goal_sound(c.settings['field_width'])
            elif scored_2:
                g.sound_handler.play_goal_sound(0)

            if not c.settings['is_training']:
                goal_time = g.current_time
                scorer = self.paddle1 if scored_1 else self.paddle2
                if scorer is not None:
                    position = h.field_mid()
                    radius = c.settings['field_height'] / 4.5
                    color = self.paddle1.color if scored_1 else self.paddle2.color
                    scorer.draw_paddle(position, radius, color)

                while g.current_time - goal_time < 1:
                    if g.current_time - goal_time > 0.7:
                        g.framework.fill_screen_semiopaque_black(20)

                    g.framework.render()
                    g.framework.tick()

        self.paddle1.current_reward = self.get_reward(self.paddle1, player_1_action)
        self.paddle2.current_reward = self.get_reward(self.paddle2, player_2_action)
        reward = self.paddle1.current_reward - self.paddle2.current_reward
        self.current_reward = reward
        self.round_reward += reward
        self.total_reward += reward
        self.stats.update(self.paddle1.current_reward - self.paddle2.current_reward)
        self.stats.update(self.paddle2.current_reward - self.paddle1.current_reward)

        return self.get_observation(1), reward, self.is_done(), { 'cumulative_reward': self.round_reward }

    def get_observation(self, player):
        if player == 1:
            obs = {
                "paddle_2_pos":     self.paddle1.get_relative_pos_of_paddle_obs(self.paddle2),
                "puck_pos":         self.paddle1.get_relative_pos_of_puck_obs(self.puck),
                "paddle_1_vel":     h.scale(self.paddle1.vel, c.gameplay['max_paddle_speed'], c.gameplay['max_paddle_speed']),
                "paddle_2_vel":     h.scale(self.paddle2.vel, c.gameplay['max_paddle_speed'], c.gameplay['max_paddle_speed']),
                "puck_vel":         h.scale(self.puck.vel, c.gameplay['max_puck_speed'], c.gameplay['max_puck_speed']),
                "goal_1_top_pos":   self.paddle1.get_relative_pos_of_goal_1_top(),
                "goal_1_bot_pos":   self.paddle1.get_relative_pos_of_goal_1_bot(),
                "goal_2_top_pos":   self.paddle1.get_relative_pos_of_goal_2_top(),
                "goal_2_bot_pos":   self.paddle1.get_relative_pos_of_goal_2_bot(),
            }
        elif player == 2:
            obs = {
                "paddle_2_pos":     self.paddle2.get_relative_pos_of_paddle_obs(self.paddle1),
                "puck_pos":         self.paddle2.get_relative_pos_of_puck_obs(self.puck),
                "paddle_1_vel":     h.scale(self.paddle2.vel, c.gameplay['max_paddle_speed'], c.gameplay['max_paddle_speed']),
                "paddle_2_vel":     h.scale(self.paddle1.vel, c.gameplay['max_paddle_speed'], c.gameplay['max_paddle_speed']),
                "puck_vel":         h.scale(self.puck.vel, c.gameplay['max_puck_speed'], c.gameplay['max_puck_speed']),
                "goal_1_top_pos":   self.paddle2.get_relative_pos_of_goal_2_top(),
                "goal_1_bot_pos":   self.paddle2.get_relative_pos_of_goal_2_bot(),
                "goal_2_top_pos":   self.paddle2.get_relative_pos_of_goal_1_top(),
                "goal_2_bot_pos":   self.paddle2.get_relative_pos_of_goal_1_bot(),
            }
            obs = {k: np.array([-v[0], v[1]]) for k, v in obs.items()}

        return obs

    def render(self):
        self.draw_background()
        self.draw_field_lines()
        self.puck.draw()
        if c.settings['is_training']:
            self.paddle1.draw(self.puck, reward_alpha=self.get_reward_alpha(self.paddle1, self.paddle2))
            self.paddle2.draw(self.puck, reward_alpha=self.get_reward_alpha(self.paddle2, self.paddle1))
        else:
            self.paddle1.draw(self.puck, None)
            self.paddle2.draw(self.puck, None)
        self.draw_goals()
        self.draw_ui()
        g.framework.render()

    def get_reward_alpha(self, paddle, other_paddle):
        mean, std = self.stats.get_stats()
        if std == 0:
            return

        p = ((paddle.current_reward - other_paddle.current_reward) - mean) / std
        alpha = (p / 4)  + 0.5
        alpha = max(0.0, min(1.0, p))
        return alpha

    def draw_background(self):
        self.background_color = g.sound_handler.target_color()
        g.framework.fill_screen(self.background_color, (c.settings['field_width'], c.settings['field_height']))

    def draw_ui(self):
        g.ui.draw_time_left(self.seconds_left())
        g.ui.draw_score(self.score, self.paddle1, self.paddle2)
        if c.settings['is_training']:
            g.ui.draw_reward(self.current_reward, self.round_reward)
            g.ui.draw_steps_left(str(self.total_training_steps_left()))

    def draw_goals(self):
        goal1_color = h.modify_hsl(self.background_color, 0.15, 0, 0)
        goal2_color = h.modify_hsl(self.background_color, -0.15, 0, 0)

        puck_to_goal_1_dist = np.linalg.norm(self.puck.pos - h.goal_pos(1))
        alpha = h.dist_alpha(puck_to_goal_1_dist) ** 2
        goal1_color = h.modify_hsl(goal1_color, 0, 0, 0.45 * alpha)

        puck_to_goal_2_dist = np.linalg.norm(self.puck.pos - h.goal_pos(2))
        alpha = h.dist_alpha(puck_to_goal_2_dist) ** 2
        goal2_color = h.modify_hsl(goal2_color, 0, 0, 0.45 * alpha)

        goal_width = 40
        goal1_pos = (0, (h.field_bot() - c.settings['goal_height']) / 2)
        goal1_size = (goal_width + 4, c.settings['goal_height'])
        goal2_pos = (h.field_right() - goal_width, (h.field_bot() - c.settings['goal_height']) / 2)
        goal2_size = (goal_width + 4, c.settings['goal_height'])

        g.framework.draw_transparent_rectangle(goal1_color, goal1_pos, goal1_size, 0.7)
        g.framework.draw_transparent_rectangle(goal2_color, goal2_pos, goal2_size, 0.7)

    def draw_field_lines(self):
        color = h.interpolate_color_rgb((255,255,255), self.background_color, 0.95)
        line_thickness = 40

        puck_to_mid_dist = np.abs(self.puck.pos[0] - c.settings['field_width'] / 2)
        alpha = h.dist_alpha(puck_to_mid_dist) ** 2
        color = h.modify_hsl(color, 0, 0, 0.15 * alpha)

        mid_circle_color = h.modify_hsl(self.background_color, 0.05, 0, -0.04)
        mid_circle_radius = int(6.75 * line_thickness)
        mid_point_radius = int(2.125 * line_thickness)
        g.framework.draw_circle(h.field_mid(), mid_circle_radius, color)
        g.framework.draw_circle(h.field_mid(), mid_circle_radius - line_thickness, mid_circle_color)
        g.framework.draw_circle(h.field_mid(), mid_point_radius, color)

        mid_line_size = (line_thickness, c.settings['field_height'])
        mid_line_pos = (h.field_mid_x() - mid_line_size[0] / 2, 0)
        g.framework.draw_rectangle(color, mid_line_pos, mid_line_size)

    def total_training_steps_left(self):
        return c.training['training_steps'] - self.total_steps

    def seconds_left(self):
        seconds_left = math.ceil((self.match_steps - self.current_step) / c.settings['fps'])
        return seconds_left

    def is_done(self):
        return self.current_step > self.match_steps or self.player_1_scored() or self.player_2_scored()

    def player_1_scored(self):
        if c.settings['blocked_goals']:
            return False

        return self.puck.pos[0] >= c.settings['field_width'] - self.puck.radius and self.puck.pos[1] > h.goal_top() and self.puck.pos[1] < h.goal_bottom()

    def player_2_scored(self):
        if c.settings['blocked_goals']:
            return False

        return self.puck.pos[0] <= self.puck.radius and self.puck.pos[1] > h.goal_top() and self.puck.pos[1] < h.goal_bottom()

    def close(self):
        g.framework.close()

def main(ai=False):
    g.initialize()
    c.settings['no_sound'] = False
    c.settings['is_training'] = False

    game = Game()

    latest_model_path, opponent_algorithm = h.get_latest_model_path_with_algorithm(c.training['base_path'], 'PPO')

    if opponent_algorithm == 'PPO':
        opponent_algorithm = PPO
    elif opponent_algorithm == 'SAC':
        opponent_algorithm = SAC
    elif opponent_algorithm == 'TD3':
        opponent_algorithm = TD3

    if latest_model_path:
        env = make_vec_env(lambda: environment.AirHockeyEnv(False), n_envs=1)
        print(f"loading model: {latest_model_path}")
        game.player_2_model = opponent_algorithm.load(latest_model_path, env=env)
        if ai:
            print(f"loading model: {latest_model_path}")
            game.player_1_model = opponent_algorithm.load(latest_model_path, env=env)

    running = True
    while running:
        running = g.framework.handle_events()
        _, _, done, _ = game.step()

        if done:
            game.reset()

    game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the game with optional profiling.")
    parser.add_argument("-p", "--profile", action="store_true", help="Run the game with profiling enabled")
    parser.add_argument("-a", "--ai", action="store_true", help="AI vs AI")
    args = parser.parse_args()

    if args.profile:
        print("Running game with profiling...")
        cProfile.run('main()', 'profile_output.prof')
        print("Profiling complete. Results saved to 'profile_output.prof'")
        print("You can visualize the results using snakeviz: snakeviz profile_output.prof")
    else:
        main(args.ai)
