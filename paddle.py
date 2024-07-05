import numpy as np
import random
import globals as g
import constants as c
import helpers as h
from collections import deque

class Paddle:
    def __init__(self, team, player):
        self.player = player
        self.team = team
        self.team_mates = 0
        self.color = (0,0,0)
        self.pos = np.zeros(2)
        self.last_pos = np.zeros(2)
        self.friction = 0.86
        self.max_acceleration = 4.0
        self.magnetic_effect_active = False
        self.original_radius = 72
        self.radius = self.original_radius
        self.last_dash_time = 0.0
        self.charging_dash = False
        self.charging_dash_initial = False
        self.charge_start_time = 0.0
        self.dash_charge_power = 0.0
        self.charge_flash_period = 0.2
        self.velocity_history = deque(maxlen=5)
        self.average_velocity = np.zeros(2)
        self.current_reward = 0.0
        self.reset()

    def reset(self):
        if c.settings['is_training']:
            if c.settings['random_starting_locations']:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular()
        else:
            self.pos = self.get_starting_pos_regular()

        self.last_dash_time = 0
        self.charging_dash = False
        self.charging_dash_initial = False
        self.vel = np.zeros(2)

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*self.radius, c.settings['field_width'] - 2*self.radius),
                                 random.uniform(2*self.radius, c.settings['field_height'] - 2*self.radius)],
                                 dtype=np.float32)
        return starting_pos

    def get_starting_pos_regular(self):
        y_spacing = c.settings['field_height'] / (self.team_mates + 1)
        y_coord = round(self.player * y_spacing)

        if self.team == 1:
            starting_pos = np.array([self.radius*3.0, y_coord])
        else:
            starting_pos = np.array([c.settings['field_width'] - self.radius*3.0, y_coord])

        return starting_pos

    def dash(self, puck):
        current_time = g.current_time
        if current_time - self.last_dash_time > c.gameplay['dash_cooldown']:
            self.last_dash_time = current_time
            self.charging_dash = False
            self.charging_dash_initial = False
            self.dash_charge_power = self.charging_alpha()
            dash_direction = self.dash_direction(puck)

            average_speed = np.linalg.norm(self.average_velocity)
            self.vel += dash_direction * self.dash_charge_power * c.gameplay['dash_impulse'] * (average_speed / c.gameplay['max_paddle_speed'])
            self.limit_speed()

            g.sound_handler.play_sound(c.gameplay['max_paddle_speed'] * (self.dash_charge_power ** 2) / 2, self.pos[0], 'dash', pitch_shift=True)

    def dash_direction(self, puck):
        if np.linalg.norm(self.average_velocity) > 0:
            puck_direction = (puck.pos - self.pos) / np.linalg.norm(puck.pos - self.pos)
            velocity_direction = self.average_velocity / np.linalg.norm(self.average_velocity)

            epsilon = 0.3
            dash_direction = epsilon * puck_direction + (1 - epsilon) * velocity_direction
            dash_direction = dash_direction / np.linalg.norm(dash_direction)
        else:
            dash_direction = np.linalg.norm(puck.pos - self.pos)

        return dash_direction

    def velocity_alpha(self):
        return min(1, np.linalg.norm(self.vel) / c.gameplay['max_paddle_speed'])

    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if not self.is_dashing:
            if speed > c.gameplay['max_paddle_speed']:
                self.vel = (self.vel / speed) * c.gameplay['max_paddle_speed']
        else:
            if speed > c.gameplay['max_paddle_speed'] + 15:
                self.vel = (self.vel / speed) * (c.gameplay['max_paddle_speed'] + 15)

    def is_dashing(self):
        return (g.current_time - self.last_dash_time) < c.gameplay['dash_duration'] * self.dash_charge_power

    def is_power_dashing(self):
        return self.is_dashing() and self.charging_alpha() == 1.0

    def full_charge_alpha(self):
        res = min(1.0, max(0.0, (self.charging_time() / c.gameplay['dash_max_charge_time'] - 1)) * 2)
        return res

    def apply_aim_assist(self, puck, max_assist_strength=0.4):
        future_time = 0.3
        predicted_puck_pos = puck.pos + puck.vel * future_time

        to_puck = predicted_puck_pos - self.pos
        distance_to_puck = np.linalg.norm(to_puck)

        ideal_vel = to_puck / future_time

        max_speed = c.gameplay['max_paddle_speed']
        if np.linalg.norm(ideal_vel) > max_speed:
            ideal_vel = ideal_vel / np.linalg.norm(ideal_vel) * max_speed

        relative_velocity = np.linalg.norm(self.vel - puck.vel)
        velocity_factor = min(relative_velocity / c.gameplay['max_paddle_speed'] + c.gameplay['max_puck_speed'], 1.0)
        velocity_factor = velocity_factor ** 2
        proximity_factor = 1 - min(distance_to_puck / (c.settings['field_width'] / 2), 1)
        proximity_factor = proximity_factor ** 4
        assist_strength = max_assist_strength * velocity_factor * proximity_factor

        self.vel = (1 - assist_strength) * self.vel + assist_strength * ideal_vel

    def set_magnetic_effect(self, active):
        self.magnetic_effect_active = active

    def calculate_magnetic_force(self, puck, ambient_force=0.075):
        max_distance = (puck.radius + self.radius) * 1.5
        to_puck = puck.pos - self.pos
        distance = np.linalg.norm(to_puck)
        if distance > max_distance:
            ambient_force = 0.0

        force_strength = ambient_force * (1 - distance / max_distance)
        if self.magnetic_effect_active:
            force_strength += 1.2

        force_direction = -to_puck / distance

        return force_strength * force_direction

    def pointless_motion(self, acceleration, epsilon=0.01):
        distance_moved = np.linalg.norm(self.pos - self.last_pos)
        acceleration_applied = np.any(acceleration != 0)
        if distance_moved < epsilon and acceleration_applied:
            return True
        return False

    def handle_controls(self, puck, action):
        if action == None:
            return

        if 'magnet' in action:
            self.set_magnetic_effect(action['magnet'])

        if action['dash']:
            if not self.charging_dash:
                self.charge_start_time = g.current_time
                self.charging_dash_initial = True
            self.charging_dash = True
        elif self.charging_dash:
            self.charging_dash = False
            self.charging_dash_initial = False
            self.dash(puck)

        if (self.team == 2 and not c.settings['player_2_human']):
            acc = action['acceleration']
            reversed_x_acc = np.array([-acc[0], acc[1]])
            self.apply_force(reversed_x_acc)
        else:
            self.apply_force(action['acceleration'])

    def apply_force(self, acc):
        self.vel += acc * c.settings['delta_t'] * self.max_acceleration

    def update(self, puck, action=None):
        self.handle_controls(puck, action)
        self.last_pos = self.pos.copy()

        charging_alpha = self.charging_alpha()
        if self.charging_dash:
            self.radius = int((1.0 + 0.3 * charging_alpha) * self.original_radius)
            self.apply_force(np.random.normal(0, (self.charging_alpha() ** 4) * 0.7, 2))
        else:
            self.radius = self.original_radius

        if self.charging_dash_initial:
            if charging_alpha >= 1.0:
                self.charging_dash_initial = False
                g.sound_handler.play_sound(30, self.pos[0], 'full-charge')

        g.sound_handler.update_paddle_sound(self)

        if self.is_dashing():
            self.apply_aim_assist(puck)
        else:
            self.vel *= (self.friction ** c.settings['delta_t'])

        self.limit_speed()
        self.update_velocity_history()
        self.pos += self.vel * c.settings['delta_t']

        if c.settings['field_split']:
            if self.team == 1:
                left_wall = self.radius
                right_wall = c.settings['field_width'] / 2 - self.radius
            else:
                left_wall = c.settings['field_width'] / 2 + self.radius
                right_wall = c.settings['field_width'] - self.radius
        else:
            left_wall = self.radius
            right_wall = c.settings['field_width'] - self.radius

        corner_collision = self.handle_corner_collision()
        if not corner_collision:
            if self.pos[0] < left_wall:
                self.pos[0] = left_wall
                self.vel[0] = 0.0
            elif self.pos[0] > right_wall:
                self.pos[0] = right_wall
                self.vel[0] = 0.0

            if self.pos[1] < self.radius:
                self.pos[1] = self.radius
                self.vel[1] = 0.0
            elif self.pos[1] > c.settings['field_height'] - self.radius:
                self.pos[1] = c.settings['field_height'] - self.radius
                self.vel[1] = 0.0

    def handle_corner_collision(self):
        corner_radius = c.settings['corner_radius']
        corner_circle_pos = None

        if self.pos[0] < corner_radius and self.pos[1] < corner_radius:
            corner_circle_pos = h.corner_top_left()
        elif self.pos[0] > h.field_right() - corner_radius and self.pos[1] < corner_radius:
            corner_circle_pos = h.corner_top_right()
        elif self.pos[0] < corner_radius and self.pos[1] > h.field_bot() - corner_radius:
            corner_circle_pos = h.corner_bot_left()
        elif self.pos[0] > h.field_right() - corner_radius and self.pos[1] > h.field_bot() - corner_radius:
            corner_circle_pos = h.corner_bot_right()

        if corner_circle_pos is None:
            return

        corner_circle_pos_dist = np.linalg.norm(corner_circle_pos - self.pos)
        corner_circle_pos_dir = (corner_circle_pos - self.pos) / corner_circle_pos_dist

        overlap = corner_circle_pos_dist + self.radius - corner_radius
        if overlap < 0:
            return

        self.pos += corner_circle_pos_dir * overlap

        projection = np.dot(self.vel, corner_circle_pos_dir) * corner_circle_pos_dir
        self.vel -= projection

    def handle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < self.radius + paddle.radius:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(2.5) * velocity_along_normal
            impulse_scalar /= (1 / self.radius + 1 / paddle.radius)
            impulse = impulse_scalar * normal
            self.vel += (impulse / self.radius)
            paddle.vel -= (impulse / paddle.radius)
            overlap = self.radius + paddle.radius - dist
            self.pos += (normal * overlap) / 2
            paddle.pos -= (normal * overlap) / 2

            sound_vel = np.linalg.norm(relative_velocity) / (2 * c.gameplay['max_paddle_speed'])
            sound_vel = sound_vel ** 2
            if sound_vel != 0:
                g.sound_handler.play_sound(sound_vel * c.gameplay['max_paddle_speed'] / 2, self.pos[0], 'paddle')

    def draw(self, puck, reward_alpha=None, draw_indicator=True):
        theme_color = g.sound_handler.target_color()
        hue_change = 0.2 if self.team == 1 else -0.2
        self.color = h.modify_hsl(theme_color, hue_change, 0.25, 0.2)

        if self.is_power_dashing():
            glow = max(0.0, 1.0 - (g.current_time - self.last_dash_time) / c.gameplay['dash_duration'])
            self.color = h.modify_hsl(self.color, 0, 0, glow*0.5)

        if self.charging_dash:
            charge_color_shift = min(0.9, self.charging_alpha() * 0.5)
            self.color = h.interpolate_color_rgb(self.color, (255,0,0), charge_color_shift)
            self.color = h.modify_hsl(self.color, 0, charge_color_shift * 0.5, charge_color_shift * 0.2)
            self.draw_dash_line(puck)

        if c.settings['is_training']:
            model_name = g.team_1_model_name if self.team == 1 else g.team_2_model_name
            g.framework.draw_text(model_name, 'model_name', (255,255,255), (self.pos[0], self.pos[1] + self.radius * 1.2), 'center')

        self.draw_paddle(self.pos, self.radius, self.color, reward_alpha, draw_indicator)

    def charging_time(self):
        return g.current_time - self.charge_start_time

    def charging_alpha(self):
        charging_time = self.charging_time()
        return max(0.0, min(1.0, charging_time / c.gameplay['dash_max_charge_time'])) ** (1/2)

    def is_overloaded(self):
        return self.charging_dash and self.full_charge_alpha() > 0

    def draw_paddle(self, position, radius, color, reward_alpha=None, draw_indicator=True):
        if self.is_overloaded():
            color = h.modify_hsl(color, 0, 0, 0.1 + 0.1 * np.sin(2 * np.pi * g.current_time / self.charge_flash_period))

        if reward_alpha is not None:
            outer_color = h.interpolate_color_rgb((255,0,0), (0,255,0), reward_alpha)
        else:
            outer_color = h.set_l(color, 0.75)

        self.draw_calls(position, radius, color, outer_color, draw_indicator)

        # if self.is_dashing():
        #     self.draw_calls(position, radius, color, outer_color, g.framework.trail_surface)

    def draw_calls(self, position, radius, color, outer_color, draw_indicator=True):
        g.framework.draw_circle(position, radius, outer_color)
        g.framework.draw_circle(position, int(8*radius / 9), h.interpolate_color_rgb(color, (0,0,0), 0.05))
        g.framework.draw_circle(position, int(radius / 2), h.interpolate_color_rgb(color, (0,0,0), 0.3))
        g.framework.draw_circle(position, int(radius / 3), h.interpolate_color_rgb(color, (0,0,0), 0.1))

        if self.team == 1 and self.player == 1 and draw_indicator:
            g.framework.draw_circle([position[0], position[1] - 1.3 * self.radius], 10, (255,255,255))

    def draw_dash_line(self, puck):
        return
        dash_direction = self.dash_direction(puck)
        angle = h.signed_angle_between(self.dash_direction(puck), np.array([1,0]))
        max_length = self.radius * 2.5
        size_alpha = self.charging_alpha() ** (1/2)
        length = max_length * size_alpha
        position = self.pos
        thickness = 18 * size_alpha
        color = h.modify_hsl(self.color, 0, 0, 0.05)
        g.framework.draw_rotated_line(position, length, -angle, color, thickness)
        arrow_head_pos = self.pos + (dash_direction / np.linalg.norm(dash_direction)) * length
        g.framework.draw_rotated_line(arrow_head_pos, 35, -angle + 180 - 40, color, thickness)
        g.framework.draw_rotated_line(arrow_head_pos, 35, -angle + 180 + 40, color, thickness)

    def update_velocity_history(self):
        self.velocity_history.append(self.vel.copy())
        self.average_velocity = np.mean(self.velocity_history, axis=0)

    def get_relative_pos_of_paddle_obs(self, paddle):
        relative_pos = paddle.pos - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_puck_obs(self, puck):
        relative_pos = puck.pos - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_1_top(self):
        relative_pos = np.array([0, (c.settings['field_height'] - c.settings['goal_height']) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_1_bot(self):
        relative_pos = np.array([0, c.settings['field_height'] - (c.settings['field_height'] - c.settings['goal_height']) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_2_top(self):
        relative_pos = np.array([c.settings['field_width'], (c.settings['field_height'] - c.settings['goal_height']) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_2_bot(self):
        relative_pos = np.array([c.settings['field_width'], c.settings['field_height'] - (c.settings['field_height'] - c.settings['goal_height']) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def normalize_relative_pos(self, relative_pos):
        relative_pos[0] /= c.settings['field_width']
        relative_pos[1] /= c.settings['field_height']
        return relative_pos

    def wall_collision_factor(self, acceleration):
        left_wall_normal = np.array([1, 0])
        right_wall_normal = np.array([-1, 0])
        top_wall_normal = np.array([0, 1])
        bottom_wall_normal = np.array([0, -1])

        dist_to_left = self.pos[0] - self.radius
        dist_to_right = c.settings['field_width'] - self.pos[0] - self.radius
        dist_to_top = self.pos[1] - self.radius
        dist_to_bottom = c.settings['field_height'] - self.pos[1] - self.radius

        total_factor = 0.0

        walls = [
            (left_wall_normal, dist_to_left),
            (right_wall_normal, dist_to_right),
            (top_wall_normal, dist_to_top),
            (bottom_wall_normal, dist_to_bottom)
        ]

        for wall_normal, dist in walls:
            if dist < self.radius * 2:
                dot_product = -np.dot(acceleration, wall_normal)
                factor = max(0, dot_product / np.linalg.norm(acceleration))
                total_factor += factor

        return total_factor

