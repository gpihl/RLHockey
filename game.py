import pygame
import torch
import pygame.gfxdraw
import numpy as np
import time
import math
import globals as g
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
import environment
from paddle import Paddle
from puck import Puck

class Game:
    _instance = None

    def __new__(cls, training):
        if cls._instance is None:
            cls._instance = super(Game, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, training=True):
        if self._initialized:
            return
        self._initialized = True            
        self.training = training
        self.total_steps = 0
        self.prev_t = time.time()
        self.curr_t = time.time()
        self.last_scorer = 2
        self.steps = 0
        self.fps = g.HIGH_FPS if training else g.LOW_FPS
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.total_reward = 0.0
        self.init_pygame()      
        self.create_objects()
        self.reset()
        self.last_ui_input = 0
        self.player_2_model = None
        self.player_2_human = False
        self.sliders = []
        self.create_sliders()
        self.events = []
        self.previous_action = np.array([0,0])
        print("Game initialization done")

    def init_pygame(self):
        pygame.init()
        vsync = 0 if self.training else 1
        self.screen = pygame.display.set_mode((g.WIDTH, g.HEIGHT), flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync = vsync)
        self.clock = pygame.time.Clock()
        self.reward_font = pygame.font.SysFont(g.REWARD_FONT, g.REWARD_FONT_SIZE)
        self.time_font = pygame.font.SysFont(g.TIME_FONT, g.TIME_FONT_SIZE)
        self.steps_left_font = pygame.font.SysFont(g.STEPS_LEFT_FONT, g.STEPS_LEFT_FONT_SIZE)

    def create_objects(self):
        self.paddle1 = Paddle(1)
        self.paddle2 = Paddle(2)
        self.puck = Puck()

    def reset(self):
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.steps = 0
        self.paddle1.reset(self.training)
        self.paddle2.reset(self.training)
        self.puck.reset(self.training, self.last_scorer)

    def create_sliders(self):
        y = int(g.HEIGHT - 400 * g.WIDTH / 2000)
        x = int(g.WIDTH - 270 * g.WIDTH / 2000)
        for key, value in g.REWARD_POLICY.items():
            slider = Slider(self.screen, x, y, int(200 * g.WIDTH / 2000), int(20 * g.WIDTH / 2000), -max(np.abs(value) * 2, 10), max(np.abs(value * 2), 10), value, key)
            self.sliders.append(slider)
            y += int(50 * g.WIDTH / 2000)

    def update_sliders(self):
        for slider in self.sliders:
            g.REWARD_POLICY[slider.label] = slider.get_value()        

    def handle_keyboard_input(self):
        self.events = pygame.event.get()        
        
        if np.abs(time.time() - self.last_ui_input) < 0.5:
            return
        self.last_ui_input = time.time()

        keys = g.get_keys()
        if keys[pygame.K_e]:
            self.fps = g.HIGH_FPS if self.fps == g.LOW_FPS else g.LOW_FPS
            print(f"switching to {self.fps} FPS")
        elif keys[pygame.K_r]:
            g.TRAINING_PARAMS['no_render'] = not g.TRAINING_PARAMS['no_render']
        elif keys[pygame.K_m]:
            g.TRAINING_PARAMS['no_sound'] = not g.TRAINING_PARAMS['no_sound']
        elif keys[pygame.K_t]:
            self.player_2_human = not self.player_2_human

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
    
    def get_reward(self, action):
        acceleration = action['acceleration']
        reward = g.REWARD_POLICY["time_reward"]
        reward += np.linalg.norm(acceleration) * g.REWARD_POLICY["acc_reward"]

        if self.player_1_scored():
            reward += g.REWARD_POLICY["player_1_goal"]
        elif self.player_2_scored():
            reward += g.REWARD_POLICY["player_2_goal"]

        relative_pos = self.puck.pos - self.paddle1.pos
        dist_to_puck = np.linalg.norm(relative_pos)
        side_multiplier = np.dot(relative_pos / dist_to_puck, np.array([1.0, 0.0]))
        proximity_reward = ((g.WIDTH - dist_to_puck) / g.WIDTH) * side_multiplier * g.REWARD_POLICY["ball_proximity"]
        reward += proximity_reward

        dist_to_center = np.abs(self.paddle1.pos[0] - g.WIDTH/2)
        center_reward = ((g.WIDTH / 2 - dist_to_center) / (g.WIDTH / 2)) * g.REWARD_POLICY["center"]
        reward += center_reward

        dist_to_player = np.linalg.norm(self.paddle1.pos - self.paddle2.pos) / g.WIDTH
        reward += dist_to_player * g.REWARD_POLICY['dist_to_player']

        pointless_reward = self.paddle1.pointless_motion(acceleration) * g.REWARD_POLICY["pointless_motion"]
        reward += pointless_reward

        reward += self.puck.collect_shot_reward('vel_2_goal') * g.REWARD_POLICY["ball_vel_2_goal"]
        reward += self.puck.collect_shot_reward('ball_velocity') * g.REWARD_POLICY["ball_velocity"]
        reward /= g.REWARD_POLICY["normalization"]

        self.current_reward = reward
        self.round_reward += reward
        self.total_reward += reward

        return reward
    
    def is_done(self):
        return self.steps > g.TIME_LIMIT or self.player_1_scored() or self.player_2_scored()

    def step(self, action=None):
        player_1_model_action = action
        player_1_action = None
        player_2_action = None

        human_action = g.get_human_action()

        if player_1_model_action is not None:
            player_1_action = g.game_action_from_model_action(player_1_model_action)
        else:
            player_1_action = human_action

        if self.player_2_model is not None:
            player_2_model_action = self.player_2_model.predict(self.get_observation(2))[0]
            player_2_action = g.game_action_from_model_action(player_2_model_action)
        else:
            player_2_action = human_action

        self.prev_t = self.curr_t
        self.curr_t = time.time()
        self.steps += 1
        self.total_steps += 1

        self.handle_keyboard_input()
        

        self.paddle1.update(self.puck, player_1_action)
        self.paddle2.update(self.puck, player_2_action)
        self.paddle1.handle_collision(self.paddle2)
        self.puck.update([self.paddle1, self.paddle2])

        if self.player_1_scored():
            self.last_scorer = 1
        elif self.player_2_scored():
            self.last_scorer = 2

        if not g.TRAINING_PARAMS['no_render']:
            self.render()

        for event in self.events:
            for slider in self.sliders:
                slider.handle_event(event)

        self.clock.tick(self.fps)

        return self.get_observation(1), self.get_reward(player_1_action), self.is_done(), { 'cumulative_reward': self.round_reward }

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

    def scale(self, vec, x_max, y_max):
        return np.array([vec[0] / x_max, vec[1] / y_max])
    
    def mirror_observation(self, arr, pos):
        new_arr = arr.copy()
        new_arr[0] = -new_arr[0]
        if pos:
            new_arr[0] += g.WIDTH

        return new_arr
    
    def render(self):
        self.screen.fill(g.BG_COLOR)
        self.draw_field_lines()
        self.puck.draw(self.screen)        
        self.paddle1.draw(self.screen)
        self.paddle2.draw(self.screen)
        self.draw_goals()        
        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        for slider in self.sliders:
            slider.draw()            
        self.draw_time_left()
        if self.training:
            self.draw_reward()
            self.draw_steps_left()

    def draw_steps_left(self):
        steps_left = str(g.TRAINING_PARAMS['training_steps'] - self.total_steps)
        text = self.steps_left_font.render(steps_left, True, g.STEPS_LEFT_COLOR)
        text_rect = text.get_rect()
        text_rect.center = g.STEPS_LEFT_POS
        self.screen.blit(text, text_rect)

    def draw_reward(self):
        current_reward_label = f"{self.current_reward:.5}"
        text = self.reward_font.render(current_reward_label, True, g.REWARD_COLOR)
        self.screen.blit(text, g.REWARD_POS)

        round_reward_label = f"{self.round_reward:.5}"
        text = self.reward_font.render(round_reward_label, True, g.REWARD_COLOR)
        self.screen.blit(text, (g.REWARD_POS[0], g.REWARD_POS[1] + 30))

    def draw_time_left(self):
        time_left = str(math.ceil((g.TIME_LIMIT - self.steps) / g.LOW_FPS))
        text = self.time_font.render(time_left, True, g.TIME_COLOR)
        text_rect = text.get_rect()
        text_rect.center = g.TIME_POS
        self.screen.blit(text, text_rect)

    def draw_goals(self):
        goal1_color = g.interpolate_color(g.PADDLE_COLOR_1, g.BG_COLOR, 0.7)
        goal2_color = g.interpolate_color(g.PADDLE_COLOR_2, g.BG_COLOR, 0.7)
        goal_width = int(16 * g.WIDTH / 800)
        goal1_pos = (0, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal1_size = (goal_width, g.GOAL_HEIGHT)
        goal2_pos = (g.WIDTH - goal_width, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal2_size = (goal_width, g.GOAL_HEIGHT)
        pygame.draw.rect(self.screen, goal1_color, (*goal1_pos, *goal1_size))
        pygame.draw.rect(self.screen, goal2_color, (*goal2_pos, *goal2_size))

    def draw_field_lines(self):
        color = g.interpolate_color((255,255,255), g.BG_COLOR, 0.9)
        line_thickness = int(18 * g.WIDTH / 800)

        mid_circle_color = g.interpolate_color((255,255,255), g.BG_COLOR, 0.8)
        mid_circle_radius = int(120 * g.WIDTH / 800)
        mid_point_radius = int(40 * g.WIDTH / 800)
        g.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius, color, self.screen, False)
        g.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius - line_thickness, mid_circle_color, self.screen, False)
        g.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_point_radius, color, self.screen, False)

        mid_line_size = (line_thickness, g.HEIGHT)
        mid_line_pos = (g.WIDTH / 2 - mid_line_size[0] / 2, 0)
        pygame.draw.rect(self.screen, color, (*mid_line_pos, *mid_line_size))        

    def close(self):
        pygame.quit()

class Slider:
    def __init__(self, screen, x, y, width, height, min_val, max_val, start_val, label):
        self.screen = screen
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.label = label
        self.handle_rect = pygame.Rect(x + (start_val - min_val) / (max_val - min_val) * width, y, 10, height)
        self.handle_color = g.interpolate_color((255,255,255), g.PADDLE_COLOR_2, 0.5)
        self.text_color = g.TIME_COLOR
        self.bg_color = g.interpolate_color((255,255,255), g.BG_COLOR, 0.5)
        self.font = pygame.font.SysFont('Arial', int(20 * g.WIDTH / 2000))
        self.dragging = False
        self.rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.rect_surface.fill((*self.bg_color, 30))

    def draw(self):
        # Draw background rectangle with transparency
        self.screen.blit(self.rect_surface, (self.rect.x, self.rect.y))

        # Draw handle
        pygame.draw.rect(self.screen, self.handle_color, self.handle_rect)

        # Draw label
        label_surface = self.font.render(f'{self.label}: {self.value:.2f}', True, self.text_color)
        self.screen.blit(label_surface, (self.rect.x, self.rect.y - int(25 * g.WIDTH / 2000)))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x = event.pos[0]
                new_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
                self.handle_rect.x = new_x
                self.value = self.min_val + (new_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
                g.REWARD_POLICY[self.label] = self.value

    def get_value(self):
        return self.value

def standalone_game():
    g.TRAINING_PARAMS['no_sound'] = False
    game = Game(training=False)

    latest_model_path = g.get_latest_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])

    algorithm = None
    if g.TRAINING_PARAMS['algorithm'] == 'PPO':
        algorithm = PPO
    elif g.TRAINING_PARAMS['algorithm'] == 'SAC':
        algorithm = SAC

    if latest_model_path:
        env = make_vec_env(lambda: environment.AirHockeyEnv(False), n_envs=1)
        game.player_2_model = algorithm.load(latest_model_path, env=env)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                print(f"Joystick {event.instance_id} disconnected")
                pygame.joystick.quit()
            elif event.type == pygame.JOYDEVICEADDED:
                print(f"Joystick connected")
                g.init_controls()

        _, _, done, _ = game.step()

        if done:
            game.reset()

    game.close()
    pygame.joystick.quit()
    pygame.quit()

if __name__ == "__main__":
    standalone_game()
