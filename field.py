import globals as g
import helpers as h
import constants as c
import numpy as np
from light import Light

class Field:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Field, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        self._initialized = True
        self.background_color = g.sound_handler.target_color()
        self.lights = self.create_lights()

    def reset(self):
        for light in self.lights:
            light.reset()

    def create_lights(self):
        light_intensity = 0.75
        lights = [
            Light(np.array([c.settings["field_width"] / 4, 0]), light_intensity, 0.05, 5, top=True),
            Light(np.array([3 * c.settings["field_width"] / 4, 0]), light_intensity, 0.3, 3, top=True),
            Light(np.array([c.settings["field_width"] / 4, h.field_bot()]), light_intensity, 0.15, 7, top=False),
            Light(np.array([3 * c.settings["field_width"] / 4, h.field_bot()]), light_intensity, 0.1, 10, top=False),
        ]

        return lights

    def update(self, puck):
        self.background_color = g.sound_handler.target_color()
        for light in self.lights:
            light.update(puck)

    def draw_bottom_layer(self, puck):
        self.draw_background()
        self.draw_corners()
        self.draw_field_lines(puck)

    def draw_top_layer(self, puck):
        self.draw_goals(puck)
        self.draw_lights()

    def team_1_scored(self, puck):
        if c.settings["blocked_goals"]:
            return False

        return puck.pos[0] >= c.settings["field_width"] - puck.radius and puck.pos[1] > h.goal_top() and puck.pos[1] < h.goal_bottom()

    def team_2_scored(self, puck):
        if c.settings["blocked_goals"]:
            return False

        return puck.pos[0] <= puck.radius and puck.pos[1] > h.goal_top() and puck.pos[1] < h.goal_bottom()

    def draw_field_lines(self, puck):
        color = self.background_color
        line_thickness = 40 * c.settings["field_width"] / 2500

        puck_to_mid_dist = np.abs(puck.pos[0] - c.settings["field_width"] / 2)
        alpha = h.dist_alpha(puck_to_mid_dist) ** 2
        color = h.modify_hsl(color, 0, 0, 0.15 * alpha)

        mid_circle_color = h.modify_hsl(self.background_color, 0.03, 0, -0.04)
        mid_circle_radius = int(6.75 * line_thickness)
        mid_point_radius = int(2.125 * line_thickness)
        g.framework.draw_circle(h.field_mid(), mid_circle_radius, color)
        g.framework.draw_circle(h.field_mid(), mid_circle_radius - line_thickness, mid_circle_color)
        g.framework.draw_circle(h.field_mid(), mid_point_radius, color)

        mid_line_size = (line_thickness, c.settings["field_height"])
        mid_line_pos = (h.field_mid_x() - mid_line_size[0] / 2, 0)
        g.framework.draw_rectangle(color, mid_line_pos, mid_line_size)

    def draw_lights(self):
        for light in self.lights:
            light.draw()

    def draw_goals(self, puck):
        goal1_color = h.modify_hsl(self.background_color, 0.15, 0, 0)
        goal2_color = h.modify_hsl(self.background_color, -0.15, 0, 0)

        puck_to_goal_1_dist = np.linalg.norm(puck.pos - h.goal_pos(1))
        alpha = h.dist_alpha(puck_to_goal_1_dist) ** 2
        goal1_color = h.modify_hsl(goal1_color, 0, 0, 0.45 * alpha)

        puck_to_goal_2_dist = np.linalg.norm(puck.pos - h.goal_pos(2))
        alpha = h.dist_alpha(puck_to_goal_2_dist) ** 2
        goal2_color = h.modify_hsl(goal2_color, 0, 0, 0.45 * alpha)

        goal_width = 50
        goal1_pos = (-goal_width / 2, (h.field_bot() - c.settings["goal_height"]) / 2)
        goal1_size = (goal_width, c.settings["goal_height"])
        goal2_pos = (h.field_right() - goal_width / 2, (h.field_bot() - c.settings["goal_height"]) / 2)
        goal2_size = (goal_width, c.settings["goal_height"])

        g.framework.draw_transparent_rectangle(goal1_color, goal1_pos, goal1_size, 0.7)
        g.framework.draw_transparent_rectangle(goal2_color, goal2_pos, goal2_size, 0.7)

        g.framework.draw_circle(h.goal_top_pos(1), goal_width / 2, goal1_color)
        g.framework.draw_circle(h.goal_bot_pos(1), goal_width / 2, goal1_color)
        g.framework.draw_circle(h.goal_top_pos(2), goal_width / 2, goal2_color)
        g.framework.draw_circle(h.goal_bot_pos(2), goal_width / 2, goal2_color)

    def draw_corners(self):
        corner_radius = c.settings["corner_radius"]
        rect_size = (corner_radius + 4, corner_radius + 4)
        color = h.modify_hsl(self.background_color, 0, 0, -0.2)
        buffer = 10
        g.framework.draw_rectangle(color, np.array([-buffer, -buffer]), rect_size)
        g.framework.draw_rectangle(color, np.array([h.field_right() - corner_radius + buffer, -buffer]), rect_size)
        g.framework.draw_rectangle(color, np.array([-buffer, h.field_bot() - corner_radius + buffer]), rect_size)
        g.framework.draw_rectangle(color, np.array([h.field_right() - corner_radius + buffer, h.field_bot() - corner_radius + buffer]), rect_size)

        g.framework.draw_circle(h.corner_top_left(), corner_radius, self.background_color)
        g.framework.draw_circle(h.corner_top_right(), corner_radius, self.background_color)
        g.framework.draw_circle(h.corner_bot_left(), corner_radius, self.background_color)
        g.framework.draw_circle(h.corner_bot_right(), corner_radius, self.background_color)

    def draw_background(self):
        g.framework.fill_screen(self.background_color, (c.settings["field_width"], c.settings["field_height"]))