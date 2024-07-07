import numpy as np
import random
import globals as g
import constants as c
import helpers as h
from light import Light

class Puck:
    def __init__(self):
        self.pos = self.get_starting_pos_regular(2)
        self.prev_puck_start_pos = self.pos
        self.shot_reward = {}
        self.shot_on_goal_reward = {}
        self.radius = 48
        self.wall_elasticity = 0.7
        self.vel = np.zeros(2)
        self.rot_vel = 0.0
        self.rot = 0.0
        self.friction = 0.997
        self.restitution = 0.95
        self.color = (0,0,0)
        self.homing = False
        self.homing_target = 1
        self.last_collider = None
        self.light = Light(self.pos, 0.45, 0, 0, None, self.color, light_type='puck')
        self.reset()

    def reset(self, last_scorer=2):
        if not c.settings['is_training']:
            self.pos = self.get_starting_pos_regular(last_scorer)
        else:
            if c.settings['random_starting_locations']:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular(last_scorer)

            # self.pos = self.get_starting_pos_moved_a_bit()

        self.vel = np.zeros(2)
        self.rot_vel = 10.0
        self.rot = 0.0
        self.homing = False

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*self.radius, c.settings['field_width'] - 2*self.radius),
                                 random.uniform(2*self.radius, c.settings['field_height'] - 2*self.radius)],
                                 dtype=np.float32)

        return starting_pos

    def get_starting_pos_moved_a_bit(self):
        var = 10
        delta = np.array([random.uniform(-var, var), random.uniform(-var, var)],dtype=np.float32)
        new_starting_pos = self.prev_puck_start_pos + delta
        self.prev_puck_start_pos += delta
        self.prev_puck_start_pos = np.clip(self.prev_puck_start_pos, self.radius*2 + 20, c.settings['field_width'] - 2*self.radius - 20)
        new_starting_pos = np.clip(new_starting_pos, self.radius*2 + 20, c.settings['field_width'] - 2*self.radius - 20)
        return new_starting_pos

    def get_starting_pos_regular(self, last_scorer):
        if last_scorer == 2:
            return np.array([c.settings['field_width'] / 4, c.settings['field_height'] / 2])
        elif last_scorer == 1:
            return np.array([c.settings['field_width'] * 3 / 4, c.settings['field_height'] / 2])

    def update(self, paddles):
        self.vel *= (self.friction ** c.settings['delta_t'])
        self.vel += np.random.normal(0, 0.005, 2) * c.settings['delta_t']

        if self.homing:
            self.vel += self.homing_acceleration()

        magnus_coefficient = 0.001
        magnus_force = magnus_coefficient * np.array([-self.vel[1], self.vel[0]]) * self.rot_vel
        self.vel += magnus_force * c.settings['delta_t']

        self.limit_speed()
        self.pos += self.vel * c.settings['delta_t']

        self.rot_vel *= (self.friction ** c.settings['delta_t'])
        self.rot += self.rot_vel * c.settings['delta_t']

        for paddle in paddles:
            self.handle_paddle_collision(paddle)
            magnetic_force = paddle.calculate_magnetic_force(self)
            self.vel += magnetic_force * c.settings['delta_t']

        self.handle_wall_collision()
        self.light.update(object=self)

    def homing_acceleration(self):
        goal_pos = h.goal_pos(self.homing_target)
        target_vel = goal_pos - self.pos
        delta_vel = target_vel - self.vel
        epsilon = 0.0025
        return delta_vel * epsilon

    def handle_wall_collision(self):
        sound_vel, normal = self.handle_corner_collision()

        if sound_vel == 0:
            if self.pos[1] < self.radius:
                self.pos[1] = self.radius
                self.vel[1] = -self.vel[1] * self.wall_elasticity
                sound_vel = self.vel[1]
                normal = np.array([0,1])
            elif self.pos[1] > c.settings['field_height'] - self.radius:
                self.pos[1] = c.settings['field_height'] - self.radius
                self.vel[1] = -self.vel[1] * self.wall_elasticity
                sound_vel = self.vel[1]
                normal = np.array([0,-1])

            if self.pos[0] < self.radius and not (self.pos[1] < h.goal_top() and self.pos[1] > h.goal_bottom()):
                self.pos[0] = self.radius
                self.vel[0] = -self.vel[0] * self.wall_elasticity
                sound_vel = self.vel[0]
                normal = np.array([1,0])
            elif self.pos[0] > c.settings['field_width'] - self.radius and not (self.pos[1] < h.goal_top() and self.pos[1] > h.goal_bottom()):
                self.pos[0] = c.settings['field_width'] - self.radius
                self.vel[0] = -self.vel[0] * self.wall_elasticity
                sound_vel = self.vel[0]
                normal = np.array([-1,0])

        if sound_vel != 0:
            sound_vel = np.abs(sound_vel)
            g.sound_handler.play_sound_velocity_based('table_hit', sound_vel, c.gameplay['max_puck_speed'], 1.5, self.pos[0], pitch_shift=True)
            g.framework.add_temporary_particles(self.pos - self.radius * normal, sound_vel, [self.color])

    def handle_corner_collision(self):
        sound_vel = 0

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
            return sound_vel, None

        corner_circle_pos_dist = np.linalg.norm(corner_circle_pos - self.pos)
        corner_circle_pos_dir = (corner_circle_pos - self.pos) / corner_circle_pos_dist

        overlap = corner_circle_pos_dist + self.radius - corner_radius
        if overlap <= 0:
            return sound_vel, None

        self.pos += corner_circle_pos_dir * np.ceil(overlap)

        projection = np.dot(self.vel, corner_circle_pos_dir) * corner_circle_pos_dir
        self.vel -= projection * (1 + self.wall_elasticity)

        sound_vel = np.linalg.norm(projection)
        return sound_vel, -corner_circle_pos_dir

    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if speed > c.gameplay['max_puck_speed']:
            self.vel = (self.vel / speed) * c.gameplay['max_puck_speed']

    def check_paddle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        return dist < self.radius + paddle.radius

    def collect_shot_reward(self, reward_type, paddle):
        key = f"{paddle.team}_{paddle.player}"
        reward = 0
        if reward_type == 'shot_toward_goal':
            if key in self.shot_on_goal_reward:
                reward = self.shot_on_goal_reward[key]
            self.shot_on_goal_reward[key] = 0
        elif reward_type == 'shot':
            if key in self.shot_reward:
                reward = self.shot_reward[key]
            self.shot_reward[key] = 0

        return reward

    def handle_paddle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if self.check_paddle_collision(paddle):
            self.last_collider = paddle

            if self.homing and self.homing_target == paddle.player:
                self.homing_target = 1 if paddle.player == 2 else 2

            prev_vel = np.array([self.vel[0], self.vel[1]])
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(1 + self.restitution) * velocity_along_normal
            impulse_scalar /= (1 / self.radius + 1 / paddle.radius)
            impulse = 0.87 * impulse_scalar * normal

            self.vel += impulse / self.radius
            self.limit_speed()

            tangent = np.array([-normal[1], normal[0]])
            velocity_along_tangent = np.dot(relative_velocity, tangent)

            rotational_impulse = np.cross(normal, tangent) * velocity_along_tangent
            rotational_impulse = 0 if np.abs(rotational_impulse) < 0.5 else rotational_impulse
            self.rot_vel += rotational_impulse * 0.13

            overlap = self.radius + paddle.radius - dist
            paddle.pos -= normal * (overlap / 2)
            paddle.vel -= 0.8 * impulse / paddle.radius
            paddle.limit_speed()
            self.pos += normal * (overlap / 2)

            # if paddle.is_power_dashing():
            #     g.sound_handler.play_sound(0.5, self.pos[0], 'power')
                # self.homing = True
                # self.homing_target = 2 if paddle.player == 1 else 1

            if c.settings['is_training']:
                key = f"{paddle.team}_{paddle.player}"
                self.shot_reward[key] = np.linalg.norm(relative_velocity)

                goal_pos = h.goal_pos(1 if paddle.team == 2 else 2)
                goal_dir = (goal_pos - self.pos) / np.linalg.norm(goal_pos - self.pos)
                self.shot_on_goal_reward[key] = np.dot(self.vel, goal_dir)

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                g.sound_handler.play_sound_velocity_based('paddle', sound_vel, c.gameplay['max_puck_speed'] + c.gameplay['max_paddle_speed'], 1.0, self.pos[0], exponent=4)
                g.sound_handler.play_sound_velocity_based('table_hit', sound_vel, c.gameplay['max_puck_speed'] + c.gameplay['max_paddle_speed'], 1.5, self.pos[0], pitch_shift=True)

            g.framework.add_temporary_particles(self.pos - self.radius * normal, sound_vel, [self.color, paddle.color])

    def draw(self):
        g.framework.begin_drawing_puck(self)
        intensity = np.linalg.norm(self.vel) * 1.3 / (c.gameplay['max_puck_speed'])
        intensity = max(min(intensity, 1.0), 0.0)
        puck_color = g.sound_handler.target_color()
        puck_color = h.modify_hsl(puck_color, 0.05, 0, 0.3 * intensity + 0.2)
        self.color = puck_color

        if self.homing:
            puck_color = h.set_l(puck_color, 0.9)
            puck_color = h.set_s(puck_color, 1.0)
            color_change_speed = 6
            puck_color = h.modify_hsl(puck_color, 0.5 + 0.5 * np.sin(g.current_time * color_change_speed), 0, 0)

        g.framework.draw_circle(self.pos, self.radius, h.modify_hsl(puck_color, 0, 0, 0.2))
        g.framework.draw_circle(self.pos, int(7*self.radius / 9), h.modify_hsl(puck_color, 0, 0, 0))
        g.framework.draw_circle(self.pos, int(8*self.radius / 9), h.modify_hsl(puck_color, 0, 0, -0.2))
        g.framework.draw_rotated_line_centered(self.pos, self.radius * 1.5, self.rot, puck_color, int(self.radius / 5.0))
        g.framework.end_drawing_puck()
