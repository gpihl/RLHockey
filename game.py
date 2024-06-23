import pygame
import torch
import pygame.gfxdraw
import numpy as np
import time
import math
import globals as g
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import environment
from paddle import Paddle
from puck import Puck

class Game:
    def __init__(self, training=True):
        self.training = training
        self.total_steps = 0
        self.steps = 0
        self.no_render = False
        self.fps = g.HIGH_FPS if training else g.LOW_FPS
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.total_reward = 0.0
        self.steps = 0        
        self.init_pygame()
        self.joystick = None
        self.init_controls()
        self.create_objects()
        self.reset()
        self.last_ui_input = 0
        self.player_2_model = None
        self.player_2_human = False
        self.sliders = []
        self.create_sliders()
        self.events = []

    def init_pygame(self):
        pygame.init()
        vsync = 0 if self.training else 1
        self.screen = pygame.display.set_mode((g.WIDTH, g.HEIGHT), flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync = vsync)
        self.clock = pygame.time.Clock()
        self.reward_font = pygame.font.SysFont(g.REWARD_FONT, g.REWARD_FONT_SIZE)
        self.time_font = pygame.font.SysFont(g.TIME_FONT, g.TIME_FONT_SIZE)
        self.steps_left_font = pygame.font.SysFont(g.STEPS_LEFT_FONT, g.STEPS_LEFT_FONT_SIZE)

    def init_controls(self):
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No joystick connected")
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Joystick name: {self.joystick.get_name()}")

    def get_joystick_action(self):
        if self.joystick == None:
            return None
        
        x = self.joystick.get_axis(2)
        y = self.joystick.get_axis(3)

        return [x * g.PADDLE_ACC, y * g.PADDLE_ACC]        

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
        self.puck.reset(self.training)

    def create_sliders(self):
        y = g.HEIGHT - 400
        x = g.WIDTH - 270
        for key, value in g.REWARD_POLICY.items():
            slider = Slider(self.screen, x, y, 200, 20, -max(np.abs(value) * 2, 10), max(np.abs(value * 2), 10), value, key)
            self.sliders.append(slider)
            y += 50

    def update_sliders(self):
        for slider in self.sliders:
            g.REWARD_POLICY[slider.label] = slider.get_value()        

    def handle_ui_input(self, keys):
        self.events = pygame.event.get()
        self.handle_keyboard_input(keys)

    def get_player_action(self, keys):
        action = [0,0]
        joystick_action = self.get_joystick_action()
        if joystick_action != None:
            action = joystick_action

        if keys[pygame.K_w]:
            action[1] = -g.PADDLE_ACC
        if keys[pygame.K_s]:
            action[1] = g.PADDLE_ACC
        if keys[pygame.K_a]:
            action[0] = -g.PADDLE_ACC
        if keys[pygame.K_d]:
            action[0] = g.PADDLE_ACC

        return action

    def handle_keyboard_input(self, keys):
        if keys[pygame.K_e] and np.abs(time.time() - self.last_ui_input) > 0.5:
            self.last_ui_input = time.time()
            self.fps = g.HIGH_FPS if self.fps == g.LOW_FPS else g.LOW_FPS
            print(f"switching to {self.fps} FPS")
        elif keys[pygame.K_r] and np.abs(time.time() - self.last_ui_input) > 0.5:
            self.last_ui_input = time.time()
            self.no_render = not self.no_render
        elif keys[pygame.K_t] and np.abs(time.time() - self.last_ui_input) > 0.5:
            self.last_ui_input = time.time()
            self.player_2_human = True

    def goal_top(self):
        return (g.HEIGHT - g.GOAL_HEIGHT) / 2
    
    def goal_bottom(self):
        return g.GOAL_HEIGHT + (g.HEIGHT - g.GOAL_HEIGHT) / 2

    def player_1_scored(self):
        return self.puck.pos[0] >= g.WIDTH - g.PUCK_RADIUS and self.puck.pos[1] > self.goal_top() and self.puck.pos[1] < self.goal_bottom()
    
    def player_2_scored(self):
        return self.puck.pos[0] <= g.PUCK_RADIUS and self.puck.pos[1] > self.goal_top() and self.puck.pos[1] < self.goal_bottom()
    
    def get_reward(self, action):
        reward = g.REWARD_POLICY["time_reward"]
        reward += np.linalg.norm(np.array(action)) * g.REWARD_POLICY["acc_reward"]

        if self.player_1_scored():
            reward += g.REWARD_POLICY["player_1_goal"]
        elif self.player_2_scored():
            reward += g.REWARD_POLICY["player_2_goal"]

        reward += ((g.WIDTH - np.linalg.norm(self.paddle1.pos - self.puck.pos)) / g.WIDTH) * g.REWARD_POLICY["ball_proximity"]

        reward += self.puck.vel[0] * g.REWARD_POLICY["ball_vel_2_goal"]
        reward += np.linalg.norm(self.puck.vel) * g.REWARD_POLICY["ball_velocity"]
        reward /= g.REWARD_POLICY["normalization"]

        self.current_reward = reward
        self.round_reward += reward
        self.total_reward += reward

        return reward
    
    def is_done(self):
        return self.steps > g.TIME_LIMIT or self.player_1_scored() or self.player_2_scored()

    def step(self, action):
        # print(self.get_observation(2))


        self.steps += 1
        self.total_steps += 1
        keys = pygame.key.get_pressed()
        self.handle_ui_input(keys)

        # print(action)
 
        # self.paddle1.control(action[0], action[1])
        # self.paddle1.update()

        # player_2_action = [0,0]
        # if self.player_2_model:
        #     player_2_action = self.player_2_model.predict(self.get_observation(2))[0]        
        # if self.player_2_human:
        #     player_2_action = self.get_player_action(keys)
        #     self.paddle2.control(player_2_action[0], player_2_action[1])
        # else:
        #     self.paddle2.control(-player_2_action[0], player_2_action[1])
        

        # self.paddle2.update()

        # self.paddle1.check_collision(self.paddle2)
        # self.puck.update([self.paddle1, self.paddle2])

        if not self.no_render:
            self.render()

        for event in self.events:
            for slider in self.sliders:
                slider.handle_event(event)

        self.clock.tick(self.fps)

        return self.get_observation(1), self.get_reward(action), self.is_done(), { 'cumulative_reward': self.round_reward }
    

    
    # def get_observation(self, player):
    #     if player == 1:
    #         obs = {
    #             "paddle_1_pos": self.scale(self.paddle1.pos, g.WIDTH, g.HEIGHT),
    #             "paddle_2_pos": self.scale(self.paddle2.pos, g.WIDTH, g.HEIGHT),
    #             "puck_pos":     self.scale(self.puck.pos, g.WIDTH, g.HEIGHT),
    #             "paddle_1_vel": self.scale(self.paddle1.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
    #             "paddle_2_vel": self.scale(self.paddle2.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
    #             "puck_vel":     self.scale(self.puck.vel, g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED),
    #             "goal_dir":     np.array([0.0, 0.0]),
    #         }

    #     if player == 2:
    #         obs = {
    #             "paddle_1_pos": self.scale(self.mirror_observation(self.paddle2.pos, True), g.WIDTH, g.HEIGHT),
    #             "paddle_2_pos": self.scale(self.mirror_observation(self.paddle1.pos, True), g.WIDTH, g.HEIGHT),
    #             "puck_pos":     self.scale(self.mirror_observation(self.puck.pos, True), g.WIDTH, g.HEIGHT),
    #             "paddle_1_vel": self.scale(self.mirror_observation(self.paddle2.vel, False), g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
    #             "paddle_2_vel": self.scale(self.mirror_observation(self.paddle1.vel, False), g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED),
    #             "puck_vel":     self.scale(self.mirror_observation(self.puck.vel, False), g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED),
    #             "goal_dir":     np.array([0.0, 0.0]),
    #         }            

    #     return obs

    def get_observation(self, player):
        # Set device to CUDA if available
        device = g.device
        
        if player == 1:
            obs = {
                "paddle_1_pos": torch.tensor(self.scale(self.paddle1.pos, g.WIDTH, g.HEIGHT), device=device),
                "paddle_2_pos": torch.tensor(self.scale(self.paddle2.pos, g.WIDTH, g.HEIGHT), device=device),
                "puck_pos":     torch.tensor(self.scale(self.puck.pos, g.WIDTH, g.HEIGHT), device=device),
                "paddle_1_vel": torch.tensor(self.scale(self.paddle1.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED), device=device),
                "paddle_2_vel": torch.tensor(self.scale(self.paddle2.vel, g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED), device=device),
                "puck_vel":     torch.tensor(self.scale(self.puck.vel, g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED), device=device),
                "goal_dir":     torch.tensor([0.0, 0.0], device=device),
            }

        if player == 2:
            obs = {
                "paddle_1_pos": torch.tensor(self.scale(self.mirror_observation(self.paddle2.pos, True), g.WIDTH, g.HEIGHT), device=device),
                "paddle_2_pos": torch.tensor(self.scale(self.mirror_observation(self.paddle1.pos, True), g.WIDTH, g.HEIGHT), device=device),
                "puck_pos":     torch.tensor(self.scale(self.mirror_observation(self.puck.pos, True), g.WIDTH, g.HEIGHT), device=device),
                "paddle_1_vel": torch.tensor(self.scale(self.mirror_observation(self.paddle2.vel, False), g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED), device=device),
                "paddle_2_vel": torch.tensor(self.scale(self.mirror_observation(self.paddle1.vel, False), g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED), device=device),
                "puck_vel":     torch.tensor(self.scale(self.mirror_observation(self.puck.vel, False), g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED), device=device),
                "goal_dir":     torch.tensor([0.0, 0.0], device=device),
            }    

        return {k: v.cpu() for k, v in obs.items()}

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
        goal_width = 16 * int((g.WIDTH) / 800)
        goal1_pos = (0, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal1_size = (goal_width, g.GOAL_HEIGHT)
        goal2_pos = (g.WIDTH - goal_width, (g.HEIGHT - g.GOAL_HEIGHT) / 2)
        goal2_size = (goal_width, g.GOAL_HEIGHT)
        pygame.draw.rect(self.screen, goal1_color, (*goal1_pos, *goal1_size))
        pygame.draw.rect(self.screen, goal2_color, (*goal2_pos, *goal2_size))

    def draw_field_lines(self):
        color = g.interpolate_color((255,255,255), g.BG_COLOR, 0.9)
        line_thickness = 18 * int((g.WIDTH) / 800)

        mid_circle_radius = 100 * int((g.WIDTH) / 800)
        mid_point_radius = 30* int((g.WIDTH) / 800)
        g.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius, color, self.screen, False)
        g.draw_circle([g.WIDTH / 2, g.HEIGHT / 2], mid_circle_radius - line_thickness, g.BG_COLOR, self.screen, False)
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
        self.font = pygame.font.SysFont('Arial', 20)
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
        self.screen.blit(label_surface, (self.rect.x, self.rect.y - 25))
    # def draw(self):
    #     # Draw background
    #     # pygame.draw.rect(self.screen, (*self.bg_color, 100), self.rect)
    #     self.screen.blit(self.rect_surface, (300, 225))
    #     # Draw handle
    #     pygame.draw.rect(self.screen, self.handle_color, self.handle_rect)
    #     # Draw label
    #     label_surface = self.font.render(f'{self.label}: {self.value:.2f}', True, self.text_color)
    #     self.screen.blit(label_surface, (self.rect.x, self.rect.y - 25))

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
                print(g.REWARD_POLICY)

    def get_value(self):
        return self.value


def standalone_game():
    game = Game(training=False)

    latest_model_path = g.get_latest_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])

    if latest_model_path:
        env = make_vec_env(lambda: environment.AirHockeyEnv(False), n_envs=1)
        game.player_2_model = SAC.load(latest_model_path, env=env)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = game.get_player_action(keys)

        _, _, done, _ = game.step(action)

        if done:
            game.reset()

    game.close()

if __name__ == "__main__":
    standalone_game()
