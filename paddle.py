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
            # self.pos = self.get_starting_pos_random()
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
        self.vel += acc

    def update(self):
        self.vel *= g.PADDLE_FRICTION
        self.vel = np.clip(self.vel, -g.MAX_PADDLE_SPEED, g.MAX_PADDLE_SPEED)
        self.pos += self.vel

        if self.pos[0] < g.PADDLE_RADIUS:
            self.pos[0] = g.PADDLE_RADIUS
            self.vel[0] = 0
        elif self.pos[0] > g.WIDTH - g.PADDLE_RADIUS:
            self.pos[0] = g.WIDTH - g.PADDLE_RADIUS
            self.vel[0] = 0

        if self.pos[1] < g.PADDLE_RADIUS:
            self.pos[1] = g.PADDLE_RADIUS
            self.vel[1] = 0
        elif self.pos[1] > g.HEIGHT - g.PADDLE_RADIUS:
            self.pos[1] = g.HEIGHT - g.PADDLE_RADIUS
            self.vel[1] = 0

    def check_collision(self, paddle):
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

    def draw(self, screen):
        g.draw_circle(self.pos, g.PADDLE_RADIUS, self.color, screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 2), g.interpolate_color(self.color, (0,0,0), 0.3), screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 3), g.interpolate_color(self.color, (0,0,0), 0.1), screen)