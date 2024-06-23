import numpy as np
import random
import globals as g

class Puck: 
    def __init__(self):
        self.pos = self.get_starting_pos_regular()
        self.prev_puck_start_pos = self.pos
        self.reset(False)

    def reset(self, training):
        if not training:
            self.pos = self.get_starting_pos_regular()
        else:
            # self.pos = self.get_starting_pos_regular()
            # self.pos = self.get_starting_pos_random()
            self.pos = self.get_starting_pos_moved_a_bit()

        self.vel = np.array([0.0,0.0])

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*g.PUCK_RADIUS, g.WIDTH - 2*g.PUCK_RADIUS), 
                                 random.uniform(2*g.PUCK_RADIUS, g.HEIGHT - 2*g.PUCK_RADIUS)], 
                                 dtype=np.float32)
        
        return starting_pos
    
    def get_starting_pos_moved_a_bit(self):
        var = 10
        delta = np.array([random.uniform(-var, var), random.uniform(-var, var)],dtype=np.float32)
        new_starting_pos = self.prev_puck_start_pos + delta
        self.prev_puck_start_pos += delta
        self.prev_puck_start_pos = np.clip(self.prev_puck_start_pos, g.PUCK_RADIUS*2 + 20, g.WIDTH - 2*g.PUCK_RADIUS - 20)
        new_starting_pos = np.clip(new_starting_pos, g.PUCK_RADIUS*2 + 20, g.WIDTH - 2*g.PUCK_RADIUS - 20)
        return new_starting_pos
    
    def get_starting_pos_regular(self):
        return np.array([g.WIDTH / 2, g.HEIGHT / 2])

    def update(self, paddles):
        self.vel *= g.PUCK_FRICTION
        self.vel = np.clip(self.vel, -g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED)
        self.pos += self.vel

        for paddle in paddles:
            self.check_collision(paddle)        

        if self.pos[1] < g.PUCK_RADIUS:
            self.pos[1] = g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]
        elif self.pos[1] > g.HEIGHT - g.PUCK_RADIUS:
            self.pos[1] = g.HEIGHT - g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]

        if self.pos[0] < g.PUCK_RADIUS:
            self.pos[0] = g.PUCK_RADIUS
            self.vel[0] = -self.vel[0]
        elif self.pos[0] > g.WIDTH - g.PUCK_RADIUS:
            self.pos[0] = g.WIDTH - g.PUCK_RADIUS
            self.vel[0] = -self.vel[0]

    def check_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < g.PUCK_RADIUS + g.PADDLE_RADIUS:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(1 + g.PUCK_RESTITUTION) * velocity_along_normal
            impulse_scalar /= (1 / g.PUCK_RADIUS + 1 / g.PADDLE_RADIUS)
            impulse = 1.2 * impulse_scalar * normal
            self.vel += impulse / g.PUCK_RADIUS
            self.vel = np.clip(self.vel, -g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED)
            overlap = g.PUCK_RADIUS + g.PADDLE_RADIUS - dist
            paddle.pos -= normal * (overlap / 2)
            paddle.vel -= 0.2 * impulse / g.PADDLE_RADIUS
            self.pos += normal * (overlap / 2)

    def draw(self, screen):
        red_level = np.linalg.norm(self.vel) / (g.MAX_PUCK_SPEED + 10)
        puck_color = g.interpolate_color(g.PUCK_COLOR, (255, 0, 0), red_level)
        g.draw_circle(self.pos, g.PUCK_RADIUS, puck_color, screen)
        g.draw_circle(self.pos, int(7*g.PUCK_RADIUS / 9), g.interpolate_color(puck_color, (0,0,0), 0.2), screen)
        g.draw_circle(self.pos, int(8*g.PUCK_RADIUS / 9), g.interpolate_color(puck_color, (0,0,0), 0.05), screen)