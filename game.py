import torch
import numpy as np
import time
import math
import globals as g
import argparse
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
import environment
from paddle import Paddle
from puck import Puck
from framework import Framework
import cProfile

class Game:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Game, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
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
        self.score = [0, 0]
        self.background_color = g.sound_handler.current_color()
        self.running_stats = g.RunningStats()
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
        self.running_stats = g.RunningStats()
        g.sound_handler.reset()

    def get_reward(self, paddle, action):
        acceleration = action['acceleration']
        reward = g.REWARD_POLICY["time_reward"]
        reward += np.linalg.norm(acceleration) * g.REWARD_POLICY["acc_reward"]

        if (self.player_1_scored() and paddle.player == 1) or (self.player_2_scored() and paddle.player == 2):
            reward += g.REWARD_POLICY["goal"]

        relative_pos = self.puck.pos - paddle.pos
        dist_to_puck = np.linalg.norm(relative_pos)
        # side_multiplier = np.dot(relative_pos / dist_to_puck, np.array([1.0, 0.0]))
        side_multiplier = 1.0
        proximity_reward = ((g.WIDTH - dist_to_puck) / g.WIDTH) * side_multiplier * g.REWARD_POLICY["ball_proximity"]
        reward += proximity_reward

        if paddle.player == 1:
            goal_pos = np.array([g.WIDTH, g.HEIGHT / 2])
        else:
            goal_pos = np.array([0, g.HEIGHT / 2])

        puck_to_goal_dist = np.linalg.norm(goal_pos - self.puck.pos)
        puck_to_goal_reward = ((g.WIDTH - puck_to_goal_dist) / g.WIDTH) * g.REWARD_POLICY["goal_proximity"]
        reward += puck_to_goal_reward

        # dist_to_center = np.abs(paddle.pos[0] - g.WIDTH/2)
        # center_reward = ((g.WIDTH / 2 - dist_to_center) / (g.WIDTH / 2)) * g.REWARD_POLICY["center"]
        # reward += center_reward

        # dist_to_player = np.linalg.norm(paddle.pos - self.paddle2.pos) / g.WIDTH
        # reward += dist_to_player * g.REWARD_POLICY['dist_to_player']

        pointless_reward = paddle.pointless_motion(acceleration) * g.REWARD_POLICY["pointless_motion"]
        reward += pointless_reward

        reward += self.puck.collect_shot_reward('vel_2_goal', player=paddle.player) * g.REWARD_POLICY["ball_vel_2_goal"]
        reward += self.puck.collect_shot_reward('ball_velocity', player=paddle.player) * g.REWARD_POLICY["ball_velocity"]
        reward /= g.REWARD_POLICY["normalization"]

        # dash_reward = action['dash'] * g.REWARD_POLICY['dash']
        # reward += dash_reward

        # reward += paddle.wall_collision_factor(acceleration) * g.REWARD_POLICY["wall_acc"]
        return reward
    
    def step(self, action=None):
        # print(self.background_color)
        # print(g.rgb_to_hsl(self.background_color[0], self.background_color[1], self.background_color[2]))

        player_1_model_action = action
        player_1_action = None
        player_2_action = None

        human_action = g.get_human_action()

        if player_1_model_action is not None:
            player_1_action = g.game_action_from_model_action(player_1_model_action)
        else:
            player_1_action = human_action

        if g.TRAINING_PARAMS['player_2_active']:
            if self.player_2_model is not None and not g.SETTINGS['player_2_human']:
                player_2_model_action = self.player_2_model.predict(self.get_observation(2))[0]
                player_2_action = g.game_action_from_model_action(player_2_model_action)
            elif g.SETTINGS['is_training']:
                player_2_action = human_action

        if player_2_action is None:
            player_2_action = g.empty_action()

        self.prev_t = self.curr_t
        self.curr_t = g.current_time
        self.current_step += 1
        self.total_steps += 1

        g.framework.handle_keyboard_input()
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

        if not g.TRAINING_PARAMS['no_render']:
            self.render()

        g.framework.tick()

        scored_1 = self.player_1_scored()
        scored_2 = self.player_2_scored()

        if scored_1 or scored_2:
            if scored_1:
                g.sound_handler.play_goal_sound(g.WIDTH)
            elif scored_2:
                g.sound_handler.play_goal_sound(0)

            if not g.SETTINGS['is_training']:
                goal_time = g.current_time
                # scorer = self.puck.last_collider
                scorer = self.paddle1 if scored_1 else self.paddle2
                if scorer is not None:
                    position = np.array([g.WIDTH / 2, g.HEIGHT / 2])
                    radius = g.HEIGHT / 4
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
        self.running_stats.update(self.paddle1.current_reward - self.paddle2.current_reward)
        self.running_stats.update(self.paddle2.current_reward - self.paddle1.current_reward)

        # if self.current_step % 15 == 0:
        #     mean, std = self.running_stats.get_stats()
        #     print(f"mean: {mean}, std: {std}")
        #     print(f"player1 reward: {self.paddle1.current_reward}, player2 reward: {self.paddle2.current_reward}")

        return self.get_observation(1), reward, self.is_done(), { 'cumulative_reward': self.round_reward }

    def get_observation(self, player):
        if player == 1:
            obs = {
                "paddle_2_pos": self.paddle1.get_relative_pos_of_paddle_obs(self.paddle2),
                "puck_pos":     self.paddle1.get_relative_pos_of_puck_obs(self.puck),
                "paddle_1_vel": self.scale(self.paddle1.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
                "paddle_2_vel": self.scale(self.paddle2.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
                "puck_vel":     self.scale(self.puck.vel, g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED),
                "goal_1_top_pos":     self.paddle1.get_relative_pos_of_goal_1_top(),
                "goal_1_bot_pos":     self.paddle1.get_relative_pos_of_goal_1_bot(),
                "goal_2_top_pos":     self.paddle1.get_relative_pos_of_goal_2_top(),
                "goal_2_bot_pos":     self.paddle1.get_relative_pos_of_goal_2_bot(),
            }
        elif player == 2:
            obs = {
                "paddle_2_pos": self.paddle2.get_relative_pos_of_paddle_obs(self.paddle1),
                "puck_pos":     self.paddle2.get_relative_pos_of_puck_obs(self.puck),
                "paddle_1_vel": self.scale(self.paddle2.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
                "paddle_2_vel": self.scale(self.paddle1.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
                "puck_vel":     self.scale(self.puck.vel, g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED),
                "goal_1_top_pos":     self.paddle2.get_relative_pos_of_goal_2_top(),
                "goal_1_bot_pos":     self.paddle2.get_relative_pos_of_goal_2_bot(),
                "goal_2_top_pos":     self.paddle2.get_relative_pos_of_goal_1_top(),
                "goal_2_bot_pos":     self.paddle2.get_relative_pos_of_goal_1_bot(),
            }
            obs = {k: np.array([-v[0], v[1]]) for k, v in obs.items()}

        return {k: torch.tensor(v, device=g.device).cpu() for k, v in obs.items()}

    def render(self):
        self.draw_background()
        self.draw_field_lines()
        self.puck.draw()
        # self.paddle1.draw(self.puck, self.running_stats if g.SETTINGS['is_training'] else None)
        # self.paddle2.draw(self.puck, self.running_stats if g.SETTINGS['is_training'] else None)
        if g.SETTINGS['is_training']:
            self.paddle1.draw(self.puck, reward_alpha=self.get_reward_alpha(self.paddle1, self.paddle2))
            self.paddle2.draw(self.puck, reward_alpha=self.get_reward_alpha(self.paddle2, self.paddle1))
        else:
            self.paddle1.draw(self.puck, None)
            self.paddle2.draw(self.puck, None)
        self.draw_goals()
        self.draw_ui()
        g.framework.render()

    def get_reward_alpha(self, paddle, other_paddle):
        mean, std = self.running_stats.get_stats()
        if std == 0:
            return
        
        p = ((paddle.current_reward - other_paddle.current_reward) - mean) / std
        # print(f"player {paddle.player} p: {p}")
        alpha = (p / 4)  + 0.5
        alpha = max(0.0, min(1.0, p))
        return alpha
            

    def draw_background(self):
        self.background_color = g.sound_handler.target_color()
        # final_color = g.interpolate_color(g.BG_COLOR, target_color, 0.2)
        
        g.framework.fill_screen(self.background_color, (g.WIDTH, g.HEIGHT))

    def draw_ui(self):
        g.ui.draw_time_left(self.seconds_left())
        g.ui.draw_score(self.score, self.paddle1, self.paddle2)
        if g.SETTINGS['is_training']:
            g.ui.draw_reward(self.current_reward, self.round_reward)
            g.ui.draw_steps_left(str(self.total_training_steps_left()))        

    def draw_goals(self):
        goal1_color = g.modify_hsl(self.background_color, 0.15, 0, 0)
        goal2_color = g.modify_hsl(self.background_color, -0.15, 0, 0)

        puck_to_goal_1_dist = np.linalg.norm(self.puck.pos - np.array([0, g.HEIGHT / 2]))
        alpha = 1.0 - min(1.0, puck_to_goal_1_dist / g.WIDTH)
        alpha = alpha ** 2
        goal1_color = g.modify_hsl(goal1_color, 0, 0, 0.45 * alpha)

        puck_to_goal_2_dist = np.linalg.norm(self.puck.pos - np.array([g.WIDTH, g.HEIGHT / 2]))
        alpha = 1.0 - min(1.0, puck_to_goal_2_dist / g.WIDTH)
        alpha = alpha ** 2
        goal2_color = g.modify_hsl(goal2_color, 0, 0, 0.45 * alpha)        

        goal_width = 40
        goal1_pos = (0, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal1_size = (goal_width + 4, g.GOAL_HEIGHT)
        goal2_pos = (g.WIDTH - goal_width, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal2_size = (goal_width + 4, g.GOAL_HEIGHT)

        g.framework.draw_transparent_rectangle(goal1_color, goal1_pos, goal1_size, 0.7)
        g.framework.draw_transparent_rectangle(goal2_color, goal2_pos, goal2_size, 0.7)

    def draw_field_lines(self):
        color = g.interpolate_color_rgb((255,255,255), self.background_color, 0.95)
        line_thickness = 40

        puck_to_mid_dist = np.abs(self.puck.pos[0] - g.WIDTH / 2)
        alpha = 1.0 - min(1.0, puck_to_mid_dist / g.WIDTH)
        alpha = alpha ** 2
        color = g.modify_hsl(color, 0, 0, 0.15 * alpha)

        mid_circle_color = g.modify_hsl(self.background_color, 0.05, 0, -0.04)
        mid_circle_radius = 270
        mid_point_radius = 85    
        g.framework.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius, color)
        g.framework.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius - line_thickness, mid_circle_color)
        g.framework.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_point_radius, color)

        mid_line_size = (line_thickness, g.HEIGHT)
        mid_line_pos = (g.WIDTH / 2 - mid_line_size[0] / 2, 0)
        g.framework.draw_rectangle(color, mid_line_pos, mid_line_size)

    def total_training_steps_left(self):
        return g.TRAINING_PARAMS['training_steps'] - self.total_steps

    def seconds_left(self):
        seconds_left = math.ceil((g.TIME_LIMIT - self.current_step) / g.LOW_FPS)
        return seconds_left
    
    def scale(self, vec, x_max, y_max):
        return np.array([vec[0] / x_max, vec[1] / y_max])
    
    def is_done(self):
        return self.current_step > g.TIME_LIMIT or self.player_1_scored() or self.player_2_scored()
    
    def goal_top(self):
        return (g.HEIGHT - g.GOAL_HEIGHT) / 2
    
    def goal_bottom(self):
        return g.GOAL_HEIGHT + (g.HEIGHT - g.GOAL_HEIGHT) / 2
    
    def player_1_scored(self):
        if g.TRAINING_PARAMS['blocked_goals']:
            return False
        
        return self.puck.pos[0] >= g.WIDTH - g.PUCK_RADIUS and self.puck.pos[1] > self.goal_top() and self.puck.pos[1] < self.goal_bottom()
    
    def player_2_scored(self):
        if g.TRAINING_PARAMS['blocked_goals']:
            return False
                
        return self.puck.pos[0] <= g.PUCK_RADIUS and self.puck.pos[1] > self.goal_top() and self.puck.pos[1] < self.goal_bottom()

    def close(self):
        g.framework.close()


def main():
    g.TRAINING_PARAMS['no_sound'] = False
    g.SETTINGS['is_training'] = False
    g.framework = Framework()
    game = Game()

    latest_model_path, opponent_algorithm = g.get_latest_model_path_with_algorithm(g.TRAINING_PARAMS['base_path'], 'PPO')

    if opponent_algorithm == 'PPO':
        opponent_algorithm = PPO
    elif opponent_algorithm == 'SAC':
        opponent_algorithm = SAC
    elif opponent_algorithm == 'TD3':
        opponent_algorithm = TD3

    if latest_model_path:
        env = make_vec_env(lambda: environment.AirHockeyEnv(False), n_envs=1)
        game.player_2_model = opponent_algorithm.load(latest_model_path, env=env)

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
    args = parser.parse_args()

    if args.profile:
        print("Running game with profiling...")
        cProfile.run('main()', 'profile_output.prof')
        print("Profiling complete. Results saved to 'profile_output.prof'")
        print("You can visualize the results using snakeviz: snakeviz profile_output.prof")
    else:
        main()
