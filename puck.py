import numpy as np
import random
import globals as g
import constants as c
import helpers as h
from light import Light
from trail import Trail
import random
import time

class Puck:
    def __init__(self):
        self.pos = self.get_starting_pos_regular(2)
        self.prev_puck_start_pos = self.pos
        self.shot_reward = {}
        self.shot_on_goal_reward = {}
        self.base_radius = 48
        self.radius = self.base_radius
        self.wall_elasticity = 0.7
        self.vel = np.zeros(2)
        self.rot_vel = 0.0
        self.rot = 0.0
        self.friction = 0.997
        self.restitution = 0.95
        self.color = (0,0,0)
        self.last_collider = None
        self.light = Light(self.pos, 0.45, 0, 0, None, self.color, light_type="puck")
        self.trail = Trail(0.93, (200,200,200), self.radius)
        self.temperature = 0.0
        self.max_temperature = 130.0
        self.last_explosion = g.current_time
        self.exploding = False
        self.reset()

    def reset(self, last_scorer=2):
        random.seed(time.time())
        self.vel = np.zeros(2)
        self.rot = 0.0
        self.rot_vel = 10.0
        self.temperature = 0.0
        self.radius = self.base_radius
        self.exploding = False

        if c.settings["is_training"]:
            if c.practice is not None:
                self.rot_vel = 0.0
                self.pos = c.practice.get_puck_starting_pos()
                self.vel = c.practice.get_puck_starting_vel()
                self.rot_vel = c.practice.get_puck_starting_rot_vel()
            elif c.settings["random_starting_locations"]:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular(last_scorer)
        else:
            self.pos = self.get_starting_pos_regular(last_scorer)

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*self.radius, c.settings["field_width"] - 2*self.radius),
                                 random.uniform(2*self.radius, c.settings["field_height"] - 2*self.radius)],
                                 dtype=np.float32)

        return starting_pos

    # def get_starting_pos_goalie_practice(self):
    #     starting_pos = np.array([random.uniform(2*self.radius, c.settings["field_width"] * 0.7 - 2 * self.radius),
    #                              random.uniform(2*self.radius, c.settings["field_height"] - 2*self.radius)],
    #                              dtype=np.float32)

    #     return starting_pos

    # def get_starting_pos_scoring_practice(self):
    #     if g.game.scorer == 1 or g.game.scorer == -1:
    #         # starting_pos = np.array([random.uniform(c.settings["field_width"] * 0.7, c.settings["field_width"] * 1.0),
    #         #                         random.uniform(c.settings["field_height"] * 0.0, c.settings["field_height"] * 1.0)],
    #         #                         dtype=np.float32)

    #         max_dist = c.settings["field_width"] * 0.2
    #         min_dist = self.radius * 3.0
    #         angular_range = 30

    #         starting_pos = h.random_vector_within_cone(h.goal_pos(2), np.array([-1.0, 0.0]), min_dist, max_dist, angular_range)
    #         self.last_starting_pos = np.copy(starting_pos)
    #     else:
    #         starting_pos = np.copy(self.last_starting_pos)

    #     return starting_pos

    def get_vel_towards_goal(self):
        goal_top_to_bot = np.array(h.goal_bot_pos(1)) - np.array(h.goal_top_pos(1))
        random_goal_pos = np.array(h.goal_top_pos(1)) + goal_top_to_bot * random.random()
        self_to_goal_pos = random_goal_pos - self.pos
        goal_dir = self_to_goal_pos / np.linalg.norm(self_to_goal_pos)
        vel = goal_dir * (60 + 15 * random.random())
        self.limit_speed()
        return vel

    def get_starting_pos_moved_a_bit(self):
        var = 10
        delta = np.array([random.uniform(-var, var), random.uniform(-var, var)],dtype=np.float32)
        new_starting_pos = self.prev_puck_start_pos + delta
        self.prev_puck_start_pos += delta
        self.prev_puck_start_pos = np.clip(self.prev_puck_start_pos, self.radius*2 + 20, c.settings["field_width"] - 2*self.radius - 20)
        new_starting_pos = np.clip(new_starting_pos, self.radius*2 + 20, c.settings["field_width"] - 2*self.radius - 20)
        return new_starting_pos

    def get_starting_pos_regular(self, last_scorer):
        if last_scorer == 2:
            return np.array([c.settings["field_width"] / 4, c.settings["field_height"] / 2])
        elif last_scorer == 1:
            return np.array([c.settings["field_width"] * 3 / 4, c.settings["field_height"] / 2])

    def get_explosion_dir(self):
        all_paddles = g.game.paddles_1 + g.game.paddles_2
        two_closest_paddles = sorted(all_paddles, key=lambda x: np.linalg.norm(x.pos - self.pos))[:2]
        paddle_to_paddle = two_closest_paddles[1].pos - two_closest_paddles[0].pos
        paddle_to_paddle_dir = paddle_to_paddle / np.linalg.norm(paddle_to_paddle)
        perp_dir = np.array([paddle_to_paddle_dir[1], -paddle_to_paddle_dir[0]])
        puck_to_mid = h.field_mid() - self.pos
        if np.dot(perp_dir, puck_to_mid) < 0:
            perp_dir *= -1

        return perp_dir

    def update(self, paddles):
        self.vel *= (self.friction ** c.settings["delta_t"])
        self.temperature *= (0.987 ** c.settings["delta_t"])
        self.vel += np.random.normal(0, 0.005, 2) * c.settings["delta_t"]

        self.radius = self.base_radius + 10 * self.get_temp_alpha()
        if self.get_temp_alpha() == 1.0 and not self.exploding:
            self.exploding = True
            self.last_explosion = g.current_time
            self.temperature = self.max_temperature * 2.0
            self.rot_vel = np.sign(random.random() - 0.5) * 20
            self.vel += self.get_explosion_dir() * c.gameplay["max_puck_speed"] * 1.7
            g.sound_handler.play_sound(0.3, self.pos[0], "light-broken")
            g.framework.add_explosion_particles(self.pos)

        if g.current_time - self.last_explosion > 3:
            self.exploding = False

        magnus_coefficient = 0.001
        magnus_force = magnus_coefficient * np.array([-self.vel[1], self.vel[0]]) * self.rot_vel
        self.vel += magnus_force * c.settings["delta_t"]

        self.limit_speed()
        self.pos += self.vel * c.settings["delta_t"]

        self.rot_vel *= (self.friction ** c.settings["delta_t"])
        self.rot += self.rot_vel * c.settings["delta_t"]

        for paddle in paddles:
            self.handle_paddle_collision(paddle)
            magnetic_force = paddle.calculate_magnetic_force(self)
            self.vel += magnetic_force * c.settings["delta_t"]

        self.handle_wall_collision()
        self.light.update(object=self)
        self.update_trail()

        # if c.settings["is_training"] and c.goalie_practice:
        #     if g.game.current_step % 80 == 0:
        #         self.vel = self.get_vel_towards_goal()
        #         # self.reset()

        # if c.settings["is_training"] and c.scoring_practice:
        #     if g.game.current_step % 180 == 0:
        #         self.reset()

    def update_trail(self):
        def get_speed_alpha(s):
            return min(1.0, max(0.0, (s / c.gameplay["max_puck_speed"]) ** 3))

        self.trail.update((self.pos.copy(), get_speed_alpha(np.linalg.norm(self.vel))))

    def handle_wall_collision(self):
        sound_vel, normal = self.handle_corner_collision()

        if sound_vel == 0:
            if self.pos[1] < self.radius:
                self.pos[1] = self.radius
                self.vel[1] = -self.vel[1] * self.wall_elasticity
                sound_vel = self.vel[1]
                normal = np.array([0,1])
            elif self.pos[1] > c.settings["field_height"] - self.radius:
                self.pos[1] = c.settings["field_height"] - self.radius
                self.vel[1] = -self.vel[1] * self.wall_elasticity
                sound_vel = self.vel[1]
                normal = np.array([0,-1])

            if self.pos[0] < self.radius and not (self.pos[1] > h.goal_top() and self.pos[1] < h.goal_bottom()):
                self.pos[0] = self.radius
                self.vel[0] = -self.vel[0] * self.wall_elasticity
                sound_vel = self.vel[0]
                normal = np.array([1,0])
            elif self.pos[0] > c.settings["field_width"] - self.radius and not (self.pos[1] > h.goal_top() and self.pos[1] < h.goal_bottom()):
                self.pos[0] = c.settings["field_width"] - self.radius
                self.vel[0] = -self.vel[0] * self.wall_elasticity
                sound_vel = self.vel[0]
                normal = np.array([-1,0])

        if sound_vel != 0:
            sound_vel = np.abs(sound_vel)
            g.sound_handler.play_sound_velocity_based("table_hit", sound_vel, c.gameplay["max_puck_speed"], 1.5, self.pos[0], pitch_shift=True)
            g.framework.add_temporary_particles(self.pos - self.radius * normal, sound_vel, [self.color])

    def handle_corner_collision(self):
        sound_vel = 0

        corner_radius = c.settings["corner_radius"]
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
        if self.exploding:
            return

        speed = np.linalg.norm(self.vel)
        if speed > c.gameplay["max_puck_speed"]:
            self.vel = (self.vel / speed) * c.gameplay["max_puck_speed"]

    def check_paddle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        return dist < self.radius + paddle.radius

    def collect_shot_reward(self, reward_type, paddle):
        key = f"{paddle.team}_{paddle.player}"
        reward = 0
        if reward_type == "shot_toward_goal":
            if key in self.shot_on_goal_reward:
                reward = self.shot_on_goal_reward[key]
            self.shot_on_goal_reward[key] = 0
        elif reward_type == "shot":
            if key in self.shot_reward:
                reward = self.shot_reward[key]
            self.shot_reward[key] = 0

        return reward

    def handle_paddle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if self.check_paddle_collision(paddle):
            self.last_collider = paddle

            prev_vel = np.array([self.vel[0], self.vel[1]])
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(1 + self.restitution) * velocity_along_normal
            impulse_scalar /= (1 / self.radius + 1 / paddle.radius)
            impulse = 0.75 * impulse_scalar * normal

            if not (self.exploding and g.current_time - self.last_explosion < 0.3):
                self.vel += impulse / self.radius

            self.limit_speed()
            self.temperature += (impulse_scalar ** 0.4) * 0.1

            tangent = np.array([-normal[1], normal[0]])
            velocity_along_tangent = np.dot(relative_velocity, tangent)

            rotational_impulse = np.cross(normal, tangent) * velocity_along_tangent
            # rotational_impulse = 0 if np.abs(rotational_impulse) < 0.5 else rotational_impulse
            self.rot_vel += rotational_impulse * 0.10 / c.update_multiplier

            overlap = self.radius + paddle.radius - dist
            paddle.pos -= normal * (overlap / 2)
            paddle.vel -= 1.0 * impulse / paddle.radius
            paddle.limit_speed()
            self.pos += normal * (overlap / 2)

            if c.settings["is_training"]:
                key = f"{paddle.team}_{paddle.player}"
                self.shot_reward[key] = np.linalg.norm(relative_velocity)

                goal_pos = h.goal_pos(1 if paddle.team == 2 else 2)
                goal_dir = (goal_pos - self.pos) / np.linalg.norm(goal_pos - self.pos)
                self.shot_on_goal_reward[key] = np.dot(self.vel, goal_dir)

                if paddle.is_dashing():
                    paddle.add_dash_shot_reward(self)

            if paddle.team == 1 and paddle.player == 1:
                h.report_practice_event("puck_finding")

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                g.sound_handler.play_sound_velocity_based("paddle", sound_vel, c.gameplay["max_puck_speed"] + c.gameplay["max_paddle_speed"], 0.4, self.pos[0], exponent=4)
                g.sound_handler.play_sound_velocity_based("table_hit", sound_vel, c.gameplay["max_puck_speed"] + c.gameplay["max_paddle_speed"], 0.7, self.pos[0], pitch_shift=True)

            g.framework.add_temporary_particles(self.pos - self.radius * normal, sound_vel, [self.color, paddle.color])

    def get_temp_alpha(self):
        temp_alpha = min(self.temperature / self.max_temperature, 1.0)
        return temp_alpha

    def draw(self):
        self.light.glow((self.temperature / self.max_temperature) * 0.7)

        if h.full_visuals():
            self.trail.draw()
            g.framework.begin_drawing_puck(self)

        intensity = np.linalg.norm(self.vel) * 1.3 / (c.gameplay["max_puck_speed"])
        intensity = max(min(intensity, 1.0), 0.0)
        puck_color = g.sound_handler.target_color()
        puck_color = h.modify_hsl(puck_color, 0.05, 0, 0.3 * intensity + 0.2)
        puck_color = h.interpolate_color_rgb(puck_color, (255,150,120), self.get_temp_alpha())
        self.color = puck_color

        g.framework.draw_circle(self.pos, self.radius, h.modify_hsl(puck_color, 0, 0, 0.2))
        g.framework.draw_circle(self.pos, int(7*self.radius / 9), h.modify_hsl(puck_color, 0, 0, 0))
        g.framework.draw_circle(self.pos, int(8*self.radius / 9), h.modify_hsl(puck_color, 0, 0, -0.2))
        g.framework.draw_rotated_line_centered(self.pos, self.radius * 1.5, self.rot, puck_color, int(self.radius / 5.0))

        if h.full_visuals():
            g.framework.end_drawing_puck()



    def draw_puck_path(self):
        g.framework.draw_puck_path(self)

