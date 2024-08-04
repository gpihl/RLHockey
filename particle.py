import numpy as np
import constants as c
import globals as g
import helpers as h
import random

class Particle:
    def __init__(self, pos, vel, color, radius, decay=0.97):
        self.pos = pos
        self.radius = radius
        self.vel = vel
        self.color = color
        self.decay = decay
        self.pyray_color = g.framework.tuple_to_color(self.color)

    def update(self):
        self.vel *= ((self.decay - 0.02) ** c.settings['delta_t'])
        self.radius *= (self.decay ** c.settings['delta_t'])
        self.pos += self.vel * c.settings['delta_t']

    def is_alive(self):
        return self.radius > 2

    def draw(self):
        g.framework.draw_rectangle_fast(self.pyray_color, (self.pos[0] - self.radius, self.pos[1] - self.radius), (self.radius*2, self.radius*2))

    @staticmethod
    def random_color(base_color, variance=20):
        r = int(min(255, max(0, np.random.normal(base_color[0], variance))))
        g = int(min(255, max(0, np.random.normal(base_color[1], variance))))
        b = int(min(255, max(0, np.random.normal(base_color[2], variance))))
        color = (r,g,b)
        return color

    @staticmethod
    def random_particle(pos, color, radius, vel):
        particle_color = Particle.random_color(color)
        return Particle(pos, vel, particle_color, radius)

    @staticmethod
    def random_particles(pos, n, colors=[(220,220,100)], mean_radius=40):
        speed = 8
        radius = np.random.normal(mean_radius, 4)
        particles = [Particle.random_particle(np.copy(pos), random.choice(colors), radius, np.random.normal([0, 0], [speed, speed])) for i in range(n)]
        return particles

    @staticmethod
    def random_particle_with_direction(pos, dir, color, mean_speed, mean_radius):
        speed = np.random.normal(mean_speed, 7)
        radius = np.random.normal(mean_radius, 30)
        particle_color = Particle.random_color(color, 40)
        decay = 0.91
        particle = Particle(pos, dir * speed, particle_color, radius, decay)
        return particle



