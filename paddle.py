import numpy as np
import random
import globals as g

class Paddle:
    def __init__(self, player):
        self.player = player
        self.color = g.PADDLE_COLOR_1 if self.player == 1 else g.PADDLE_COLOR_2
        self.reset(False)

    def reset(self, training):
        if not training:
            self.pos = self.get_starting_pos_regular()
        else:
            if g.TRAINING_PARAMS['random_starting_locations']:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular()            

        self.vel = np.array([0, 0], dtype=np.float32)

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*g.PADDLE_RADIUS, g.WIDTH - 2*g.PADDLE_RADIUS), 
                                 random.uniform(2*g.PADDLE_RADIUS, g.HEIGHT - 2*g.PADDLE_RADIUS)], 
                                 dtype=np.float32)
        return starting_pos
    
    def get_starting_pos_regular(self):
        if self.player == 1:
            starting_pos = np.array([g.PADDLE_RADIUS*3.0, g.HEIGHT // 2])
        else:
            starting_pos = np.array([g.WIDTH - g.PADDLE_RADIUS*3.0, g.HEIGHT // 2])

        return starting_pos

    def control(self, ax, ay):
        acc = np.array([ax, ay], dtype=np.float32)
        self.vel += acc * g.DELTA_T

    def get_relative_pos_of_paddle_obs(self, paddle):
        relative_pos = paddle.pos - self.pos
        return self.normalize_relative_pos(relative_pos)
    
    def get_relative_pos_of_puck_obs(self, puck):
        relative_pos = puck.pos - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_1_top(self):
        relative_pos = np.array([0, (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)
    
    def get_relative_pos_of_goal_1_bot(self):
        relative_pos = np.array([0, g.HEIGHT - (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)    
    
    def get_relative_pos_of_goal_2_top(self):
        relative_pos = np.array([g.WIDTH, (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_2_bot(self):
        relative_pos = np.array([g.WIDTH, g.HEIGHT - (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def normalize_relative_pos(self, relative_pos):
        relative_pos[0] /= g.WIDTH
        relative_pos[1] /= g.HEIGHT
        return relative_pos
    
    def update(self, training):
        self.vel *= (g.PADDLE_FRICTION ** g.DELTA_T)
        self.vel = np.clip(self.vel, -g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED)
        if training and self.player == 2:
            self.vel += np.random.normal(0, 0.4, 2) * g.DELTA_T

        self.pos += self.vel * g.DELTA_T
        
        if g.TRAINING_PARAMS['field_split']:
            if self.player == 1:
                left_wall = g.PADDLE_RADIUS
                right_wall = g.WIDTH / 2 + g.PADDLE_RADIUS
            else:
                left_wall = g.WIDTH / 2 - g.PADDLE_RADIUS
                right_wall = g.WIDTH - g.PADDLE_RADIUS
        else:
            left_wall = g.PADDLE_RADIUS
            right_wall = g.WIDTH - g.PADDLE_RADIUS

        if self.pos[0] < left_wall:
            self.pos[0] = left_wall
            self.vel[0] = 0
        elif self.pos[0] > right_wall:
            self.pos[0] = right_wall
            self.vel[0] = 0

        if self.pos[1] < g.PADDLE_RADIUS:
            self.pos[1] = g.PADDLE_RADIUS
            self.vel[1] = 0
        elif self.pos[1] > g.HEIGHT - g.PADDLE_RADIUS:
            self.pos[1] = g.HEIGHT - g.PADDLE_RADIUS
            self.vel[1] = 0

    def handle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < g.PADDLE_RADIUS + g.PADDLE_RADIUS:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(2) * velocity_along_normal
            impulse_scalar /= (1 / g.PADDLE_RADIUS + 1 / g.PADDLE_RADIUS)
            impulse = impulse_scalar * normal
            self.vel += (impulse / g.PADDLE_RADIUS)
            paddle.vel -= (impulse / g.PADDLE_RADIUS)
            overlap = g.PADDLE_RADIUS + g.PADDLE_RADIUS - dist
            self.pos += (normal * overlap) / 2
            paddle.pos -= (normal * overlap) / 2

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                sound_vel = np.abs(sound_vel)
                g.sound_handler.play_sound(sound_vel, self.pos[0], 'paddle')

    def draw(self, screen):
        g.draw_circle(self.pos, g.PADDLE_RADIUS, self.color, screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 2), g.interpolate_color(self.color, (0,0,0), 0.3), screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 3), g.interpolate_color(self.color, (0,0,0), 0.1), screen)