import numpy as np
import random
import globals as g

class Puck: 
    def __init__(self):
        self.pos = self.get_starting_pos_regular()
        self.prev_puck_start_pos = self.pos
        self.reset()

    def reset(self):
        # self.pos = self.get_starting_pos_random()
        # self.pos = self.get_starting_pos_regular()
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
        return np.array([g.WIDTH / 2 - g.PUCK_RADIUS, g.HEIGHT / 2 - g.PUCK_RADIUS])

    def update(self, paddles):
        self.vel *= g.PUCK_FRICTION
        self.vel = np.clip(self.vel, -g.MAX_PUCK_SPEED, g.MAX_PUCK_SPEED)
        self.pos += self.vel

        if self.pos[1] < g.PUCK_RADIUS:
            self.pos[1] = g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]
        elif self.pos[1] > g.HEIGHT - g.PUCK_RADIUS:
            self.pos[1] = g.HEIGHT - g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]

        # if self.pos[0] < g.PUCK_RADIUS:
        #     self.pos[0] = g.PUCK_RADIUS
        #     self.vel[0] = -self.vel[0]
        # elif self.pos[0] > g.WIDTH - g.PUCK_RADIUS:
        #     self.pos[0] = g.WIDTH - g.PUCK_RADIUS
        #     self.vel[0] = -self.vel[0]

        for paddle in paddles:
            self.check_collision(paddle)

    def check_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < g.PUCK_RADIUS + g.PADDLE_RADIUS:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            restitution = 0.9
            impulse_scalar = -(1 + restitution) * velocity_along_normal
            impulse_scalar /= (1 / g.PUCK_RADIUS + 1 / g.PADDLE_RADIUS)
            impulse = impulse_scalar * normal
            self.vel += impulse / g.PUCK_RADIUS
            overlap = g.PUCK_RADIUS + g.PADDLE_RADIUS - dist
            self.pos += normal * overlap

    def draw(self, screen):
        red_level = np.linalg.norm(self.vel) / g.MAX_PUCK_SPEED
        puck_color = g.interpolate_color(g.PUCK_COLOR, (255, 0, 0), red_level)
        g.draw_circle(self.pos, g.PUCK_RADIUS, puck_color, screen)