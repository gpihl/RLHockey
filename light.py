import noise
import globals as g
import constants as c
import numpy as np
import helpers as h

class Light:
    def __init__(self, pos, spawn_intensity, amplitude, time_scale, top, color=(255,255,255), light_type="wall"):
        self.pos = pos
        self.spawn_intensity = spawn_intensity
        self.base_intensity = spawn_intensity
        self.intensity = spawn_intensity
        self.amplitude = amplitude
        self.time_scale = time_scale
        self.top = top
        self.width = 40
        self.height = 8
        self.broken = False
        self.color = color
        self.type = light_type

    def reset(self):
        self.broken = False
        self.base_intensity = self.spawn_intensity

    def glow(self, glow_amount):
        self.base_intensity = self.spawn_intensity + glow_amount

    def update(self, puck=None, object=None):
        if self.type == "wall":
            self.handle_puck_collision(puck)
            t = g.current_time
            noise_value = noise.pnoise1(t * self.time_scale)
            self.intensity = self.base_intensity + self.amplitude * noise_value
            self.intensity = max(0.0, min(1.0, self.intensity))
        elif self.type == "paddle":
            self.pos = object.pos
            self.color = h.modify_hsl(object.color, 0, 0, 0.05)
            # if object.charging_dash:
            #     self.intensity = self.base_intensity + object.charging_alpha() * (1.0 - self.base_intensity) * 2
            # else:
            #     self.intensity = self.base_intensity
        elif self.type == "puck":
            self.intensity = self.base_intensity
            self.pos = object.pos
            self.color = h.modify_hsl(object.color, 0, 0, 0.05)

    def draw(self):
        if self.top:
            rect_pos = self.pos + np.array([-self.width/2, 0])
        else:
            rect_pos = self.pos + np.array([-self.width/2, -self.height])

        bright = int(255 * (self.intensity + 0.15))
        bright = max(0, min(255, bright))
        color = (bright, bright, bright)
        g.framework.draw_rectangle(color, rect_pos, (self.width, self.height))

    def handle_puck_collision(self, puck):
        if self.broken:
            return

        puck_to_light = self.pos - puck.pos
        dist = np.linalg.norm(puck_to_light)

        if dist < (self.height + puck.radius) * 1.3 and np.abs(puck.vel[1]) > 20:
            self.broken = True
            self.base_intensity = 0
            g.framework.add_temporary_particles(self.pos, 400, [(100,100,230)])
            g.sound_handler.play_sound(0.2, self.pos[0], "light-broken")





