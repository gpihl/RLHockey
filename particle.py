import numpy as np
import constants as c
import globals as g
import helpers as h
import random

class Particle:
    def __init__(self, pos, vel, color, radius):
        self.pos = pos
        self.radius = radius
        self.vel = vel
        self.color = color
        self.pyray_color = g.framework.tuple_to_color(self.color)

    def update(self):
        self.vel *= (0.95 ** c.settings['delta_t'])
        self.radius *= (0.97 ** c.settings['delta_t'])
        self.pos += self.vel * c.settings['delta_t']

    def is_alive(self):
        return self.radius > 2

    def draw(self):
        g.framework.draw_rectangle_fast(self.pyray_color, (self.pos[0] - self.radius, self.pos[1] - self.radius), (self.radius*2, self.radius*2))

    @staticmethod
    def random_particle(pos, color):
        speed = 8
        vel = np.random.normal([0, 0], [speed, speed])
        radius = np.random.normal(40, 4)
        r = int(min(255, max(0, np.random.normal(color[0], 20))))
        g = int(min(255, max(0, np.random.normal(color[1], 20))))
        b = int(min(255, max(0, np.random.normal(color[2], 20))))
        particle_color = (r,g,b)
        return Particle(pos, vel, particle_color, radius)

    @staticmethod
    def random_particles(pos, n, colors=[(220,220,100)]):
        particles = [Particle.random_particle(np.copy(pos), random.choice(colors)) for i in range(n)]
        return particles



