import numpy as np
import random
import globals as g
import time
from collections import deque

class Paddle:
    def __init__(self, player):
        self.player = player
        self.underlying_color = g.PADDLE_COLOR_1 if self.player == 1 else g.PADDLE_COLOR_2
        self.color = self.underlying_color
        self.pos = np.zeros(2)
        self.last_pos = np.zeros(2)
        self.magnetic_effect_active = False
        self.radius = g.PADDLE_RADIUS        
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
        if g.SETTINGS['is_training']:
            if g.TRAINING_PARAMS['random_starting_locations']:
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
        starting_pos = np.array([random.uniform(2*self.radius, g.WIDTH - 2*self.radius), 
                                 random.uniform(2*self.radius, g.HEIGHT - 2*self.radius)], 
                                 dtype=np.float32)
        return starting_pos
        
    def get_starting_pos_regular(self):
        if self.player == 1:
            starting_pos = np.array([self.radius*3.0, g.HEIGHT // 2])
        else:
            starting_pos = np.array([g.WIDTH - self.radius*3.0, g.HEIGHT // 2])

        return starting_pos
      
    def dash(self, puck):
        if g.SETTINGS['is_training'] and not g.TRAINING_PARAMS['dash_enabled']:
            return

        current_time = g.current_time
        if current_time - self.last_dash_time > g.GAMEPLAY_PARAMS['dash_cooldown']:
            self.last_dash_time = current_time
            self.charging_dash = False
            self.charging_dash_initial = False
            self.dash_charge_power = self.charging_alpha()
            dash_direction = self.dash_direction(puck) 

            average_speed = np.linalg.norm(self.average_velocity)
            self.vel += dash_direction * self.dash_charge_power * g.GAMEPLAY_PARAMS['dash_impulse'] * (average_speed / g.MAX_PADDLE_SPEED)
            self.limit_speed()
            
            g.sound_handler.play_sound(g.MAX_PADDLE_SPEED * self.dash_charge_power / 2, self.pos[0], 'dash', pitch_shift=True)
            
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

    # def dash_direction(self, puck):
    #     alpha = 0.1
    #     N = 4

    #     if np.linalg.norm(self.average_velocity) > 0:
    #         r1 = self.pos
    #         v1 = self.average_velocity
    #         r2 = puck.pos
    #         v2 = np.zeros_like(v1)  # Assuming puck is stationary

    #         # Calculate the line-of-sight vector
    #         d = r2 - r1

    #         # Calculate the relative velocity
    #         v_rel = v2 - v1

    #         # Calculate the angular rate of the line-of-sight
    #         d_norm = np.linalg.norm(d)
    #         if d_norm == 0:
    #             dash_direction = np.zeros_like(d)  # Avoid division by zero
    #         else:
    #             # Angular rate of LOS (using 2D cross product equivalent)
    #             lambda_dot = np.cross(d, v_rel) / d_norm**2

    #             # Compute the proportional navigation acceleration direction
    #             a_direction = N * lambda_dot * np.array([-d[1], d[0]]) / d_norm  # Rotate vector by 90 degrees

    #             # Scale the acceleration to the fixed magnitude
    #             dash_direction = alpha * a_direction / np.linalg.norm(a_direction)
    #     else:
    #         dash_direction = (puck.pos - self.pos) / np.linalg.norm(puck.pos - self.pos)
        
    #     return dash_direction
   
    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if speed > g.MAX_PUCK_SPEED:
            self.vel = (self.vel / speed) * g.MAX_PUCK_SPEED

    def is_dashing(self):
        return (g.current_time - self.last_dash_time) < g.GAMEPLAY_PARAMS['dash_duration'] * self.dash_charge_power
    
    def is_power_dashing(self):
        return self.is_dashing() and self.charging_alpha() == 1.0
    
    def full_charge_alpha(self):
        res = min(1.0, max(0.0, (self.charging_time() / g.GAMEPLAY_PARAMS['dash_max_charge_time'] - 1)) * 2)
        return res

    def apply_aim_assist(self, puck, max_assist_strength=0.4):
        future_time = 0.3  # Look 0.5 seconds ahead
        predicted_puck_pos = puck.pos + puck.vel * future_time

        # Calculate vector from paddle to predicted puck position
        to_puck = predicted_puck_pos - self.pos
        distance_to_puck = np.linalg.norm(to_puck)

        # Calculate the ideal velocity vector
        ideal_vel = to_puck / future_time

        # Limit the magnitude of the ideal velocity
        max_speed = g.MAX_PADDLE_SPEED
        if np.linalg.norm(ideal_vel) > max_speed:
            ideal_vel = ideal_vel / np.linalg.norm(ideal_vel) * max_speed

        # Calculate assist strength based on current velocity and proximity to puck
        relative_velocity = np.linalg.norm(self.vel - puck.vel)
        velocity_factor = min(relative_velocity / g.MAX_PADDLE_SPEED + g.MAX_PUCK_SPEED, 1.0)
        velocity_factor = velocity_factor ** 2
        proximity_factor = 1 - min(distance_to_puck / (g.WIDTH / 2), 1)
        proximity_factor = proximity_factor ** 4
        assist_strength = max_assist_strength * velocity_factor * proximity_factor

        # Blend current velocity with ideal velocity
        self.vel = (1 - assist_strength) * self.vel + assist_strength * ideal_vel

    def set_magnetic_effect(self, active):
        self.magnetic_effect_active = active

    def calculate_magnetic_force(self, puck_pos, ambient_force=0.1):
        max_distance = (g.PUCK_RADIUS + self.radius) * 1.5
        to_puck = puck_pos - self.pos
        distance = np.linalg.norm(to_puck)
        
        if distance > max_distance:
            ambient_force = 0.0

        # Calculate force strength (stronger when closer)
        force_strength = ambient_force * (1 - distance / max_distance)

        if self.magnetic_effect_active:
            force_strength += 1.2
        
        # Calculate force direction (normalized vector towards paddle)
        force_direction = -to_puck / distance
        
        return force_strength * force_direction        

    def pointless_motion(self, acceleration, epsilon=0.01):
        # Calculate the distance moved since last position
        distance_moved = np.linalg.norm(self.pos - self.last_pos)
        
        # Check if acceleration is being applied (non-zero)
        acceleration_applied = np.any(acceleration != 0)
        
        # Check if movement is below epsilon and acceleration is being applied
        if distance_moved < epsilon and acceleration_applied:
            return True
        return False
    
    def handle_controls(self, puck, action):
        if action == None:
            return
        
        self.set_magnetic_effect(action['magnet'])
        
        if action['dash']:
            if not self.charging_dash:
                self.charge_start_time = g.current_time
                self.charging_dash_initial = True
                # g.sound_handler.play_sound(20, self.pos[0], f"charge{self.player}")
            self.charging_dash = True
        elif self.charging_dash:
            self.charging_dash = False
            self.charging_dash_initial = False
            self.dash(puck)
        
        if (self.player == 2 and not g.SETTINGS['player_2_human']):
            acc = action['acceleration']
            reversed_x_acc = np.array([-acc[0], acc[1]])
            self.apply_force(reversed_x_acc)
        else:
            self.apply_force(action['acceleration'])

    def apply_force(self, acc):
        self.vel += acc * g.DELTA_T * g.PADDLE_ACC
  
    def update(self, puck, action=None):

        # if self.player == 2:
        #     print(action)
        self.handle_controls(puck, action)

        self.last_pos = self.pos.copy()

        charging_alpha = self.charging_alpha()
        if self.charging_dash:
            self.radius = int((1.0 + 0.3 * charging_alpha) * g.PADDLE_RADIUS)
            self.apply_force(np.random.normal(0, (self.charging_alpha() ** 4) * 0.7, 2))
        else:
            self.radius = g.PADDLE_RADIUS

        if self.charging_dash_initial:
            if charging_alpha >= 1.0:
                self.charging_dash_initial = False
                g.sound_handler.play_sound(30, self.pos[0], 'full-charge')
        
        g.sound_handler.update_paddle_sound(self)
                
        if self.is_dashing():
            self.apply_aim_assist(puck)
        else:
            self.vel *= (g.PADDLE_FRICTION ** g.DELTA_T)
 
        speed = np.linalg.norm(self.vel)
        if speed > g.MAX_PADDLE_SPEED:
            self.vel = (self.vel / speed) * g.MAX_PADDLE_SPEED

        self.update_velocity_history()      
        self.pos += self.vel * g.DELTA_T
        
        if g.TRAINING_PARAMS['field_split']:
            if self.player == 1:
                left_wall = self.radius
                right_wall = g.WIDTH / 2 - self.radius
            else:
                left_wall = g.WIDTH / 2 + self.radius
                right_wall = g.WIDTH - self.radius
        else:
            left_wall = self.radius
            right_wall = g.WIDTH - self.radius

        if self.pos[0] < left_wall:
            self.pos[0] = left_wall
            self.vel[0] = 0.0
        elif self.pos[0] > right_wall:
            self.pos[0] = right_wall
            self.vel[0] = 0.0

        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] = 0.0
        elif self.pos[1] > g.HEIGHT - self.radius:
            self.pos[1] = g.HEIGHT - self.radius
            self.vel[1] = 0.0

    def handle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < self.radius + paddle.radius:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(3.5) * velocity_along_normal
            impulse_scalar /= (1 / self.radius + 1 / paddle.radius)
            impulse = impulse_scalar * normal
            self.vel += (impulse / self.radius)
            paddle.vel -= (impulse / paddle.radius)
            overlap = self.radius + paddle.radius - dist
            self.pos += (normal * overlap) / 2
            paddle.pos -= (normal * overlap) / 2

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                g.sound_handler.play_sound(sound_vel, self.pos[0], 'paddle')

    def draw(self, puck, reward_alpha=None):        
        theme_color = g.sound_handler.target_color()
        # self.color = g.interpolate_color(self.underlying_color, theme_color, 0.5)
        hue_change = 0.15 if self.player == 1 else -0.15
        self.color = g.modify_hsl(theme_color, hue_change, 0, 0.2)

        if self.is_power_dashing():
            glow = max(0.0, 1.0 - (g.current_time - self.last_dash_time) / g.GAMEPLAY_PARAMS['dash_duration'])
            self.color = g.modify_hsl(self.color, 0, 0, glow*0.5)

        if self.charging_dash:
            charge_color_shift = min(0.9, self.charging_alpha() * 0.5)
            self.color = g.interpolate_color_rgb(self.color, (255,0,0), charge_color_shift)
            self.color = g.modify_hsl(self.color, 0, charge_color_shift * 0.5, charge_color_shift * 0.2)
            self.draw_dash_line(puck)
        
        self.draw_paddle(self.pos, self.radius, self.color, reward_alpha)

    def charging_time(self):
        return g.current_time - self.charge_start_time
    
    def charging_alpha(self):
        charging_time = self.charging_time()
        return min(1.0, charging_time / g.GAMEPLAY_PARAMS['dash_max_charge_time']) ** (1/2)
    
    def is_overloaded(self):
        return self.charging_dash and self.full_charge_alpha() > 0

    def draw_paddle(self, position, radius, color, reward_alpha=None):
        if self.is_overloaded():
            # glow = (1.0 - full_charge_alpha) ** 4
            # color = g.modify_hsl(color, 0, 0, glow * 0.5)
            color = g.modify_hsl(color, 0, 0, 0.1 + 0.1 * np.sin(2 * np.pi * g.current_time / self.charge_flash_period))            
       
        if reward_alpha is not None:

            outer_color = g.interpolate_color_rgb((255,0,0), (0,255,0), reward_alpha)
        else:
            outer_color = g.set_l(color, 0.75)

        g.framework.draw_circle(position, radius, outer_color)
        g.framework.draw_circle(position, int(8*radius / 9), g.interpolate_color_rgb(color, (0,0,0), 0.05))
        g.framework.draw_circle(position, int(radius / 2), g.interpolate_color_rgb(color, (0,0,0), 0.3))
        g.framework.draw_circle(position, int(radius / 3), g.interpolate_color_rgb(color, (0,0,0), 0.1))

    def draw_dash_line(self, puck):
        dash_direction = self.dash_direction(puck)
        angle = g.signed_angle_between(self.dash_direction(puck), np.array([1,0]))
        max_length = self.radius * 2.5
        size_alpha = self.charging_alpha() ** (1/2)
        length = max_length * size_alpha
        position = self.pos
        thickness = 18 * size_alpha
        color = g.modify_hsl(self.color, 0, 0, 0.05)
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
        relative_pos = np.array([0, (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)
    
    def get_relative_pos_of_goal_1_bot(self):
        relative_pos = np.array([0, g.HEIGHT - (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)    
    
    def get_relative_pos_of_goal_2_top(self):
        relative_pos = np.array([g.WIDTH, (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def get_relative_pos_of_goal_2_bot(self):
        relative_pos = np.array([g.WIDTH, g.HEIGHT - (g.HEIGHT - g.GOAL_HEIGHT) / 2]) - self.pos
        return self.normalize_relative_pos(relative_pos)

    def normalize_relative_pos(self, relative_pos):
        relative_pos[0] /= g.WIDTH
        relative_pos[1] /= g.HEIGHT
        return relative_pos

    def wall_collision_factor(self, acceleration):
        left_wall_normal = np.array([1, 0])
        right_wall_normal = np.array([-1, 0])
        top_wall_normal = np.array([0, 1])
        bottom_wall_normal = np.array([0, -1])

        dist_to_left = self.pos[0] - g.PADDLE_RADIUS
        dist_to_right = g.WIDTH - self.pos[0] - g.PADDLE_RADIUS
        dist_to_top = self.pos[1] - g.PADDLE_RADIUS
        dist_to_bottom = g.HEIGHT - self.pos[1] - g.PADDLE_RADIUS

        total_factor = 0.0

        walls = [
            (left_wall_normal, dist_to_left),
            (right_wall_normal, dist_to_right),
            (top_wall_normal, dist_to_top),
            (bottom_wall_normal, dist_to_bottom)
        ]

        for wall_normal, dist in walls:
            if dist < g.PADDLE_RADIUS * 2:
                dot_product = -np.dot(acceleration, wall_normal)
                factor = max(0, dot_product / np.linalg.norm(acceleration))
                total_factor += factor
        
        return total_factor

