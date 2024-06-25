import numpy as np
import random
import globals as g
import time
from collections import deque

class Paddle:
    def __init__(self, player):
        self.player = player
        self.color = g.PADDLE_COLOR_1 if self.player == 1 else g.PADDLE_COLOR_2
        self.pos = np.zeros(2)
        self.last_pos = np.zeros(2)
        self.magnetic_effect_active = False
        self.radius = g.PADDLE_RADIUS        
        self.reset()
        self.last_dash_time = 0.0
        self.charging_dash = False
        self.charge_start_time = 0.0
        self.dash_charge_power = 0.0
        self.velocity_history = deque(maxlen=5)
        self.average_velocity = np.zeros(2)       

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

        current_time = time.time()
        if current_time - self.last_dash_time > g.GAMEPLAY_PARAMS['dash_cooldown']:
            self.last_dash_time = current_time
            self.charging_dash = False
            charge_time = current_time - self.charge_start_time
            self.dash_charge_power = min(1.0, charge_time / g.GAMEPLAY_PARAMS['dash_max_charge_time'])

            # Apply impulse in the direction of current movement
            if np.linalg.norm(self.average_velocity) > 0:
                puck_direction = (puck.pos - self.pos) / np.linalg.norm(puck.pos - self.pos)
                velocity_direction = self.average_velocity / np.linalg.norm(self.average_velocity)

                epsilon = 0.3
                dash_direction = epsilon * puck_direction + (1 - epsilon) * velocity_direction
                dash_direction = dash_direction / np.linalg.norm(dash_direction)
            else:
                # If not moving, dash in the direction the paddle is facing
                dash_direction = np.array([1, 0]) if self.player == 1 else np.array([-1, 0])

            average_speed = np.linalg.norm(self.average_velocity)


            self.vel += dash_direction * self.dash_charge_power * g.GAMEPLAY_PARAMS['dash_impulse'] * (average_speed / g.MAX_PADDLE_SPEED)
            self.limit_speed()
            
            g.sound_handler.play_sound(g.MAX_PADDLE_SPEED * self.dash_charge_power / 2, self.pos[0], 1)
            
    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if speed > g.MAX_PUCK_SPEED:
            self.vel = (self.vel / speed) * g.MAX_PUCK_SPEED

    def is_dashing(self):
        return (time.time() - self.last_dash_time) < g.GAMEPLAY_PARAMS['dash_duration'] * self.dash_charge_power

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
        self.set_magnetic_effect(action['magnet'])

        if action['dash']:
            if not self.charging_dash:
                self.charge_start_time = time.time()
            self.charging_dash = True
        elif self.charging_dash:
            self.charging_dash = False
            self.dash(puck)
        
        self.apply_force(action['acceleration'])

    def apply_force(self, acc):
        self.vel += acc * g.DELTA_T * g.PADDLE_ACC
  
    def update(self, puck, action):
        self.handle_controls(puck, action)

        self.last_pos = self.pos.copy()

        if self.charging_dash:
            charging_time = time.time() - self.charge_start_time
            self.radius = int((1.0 + 0.3 * min(1.0, charging_time / g.GAMEPLAY_PARAMS['dash_max_charge_time'])) * g.PADDLE_RADIUS)
        else:
            self.radius = g.PADDLE_RADIUS

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
        if dist < self.radius + self.radius:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(2) * velocity_along_normal
            impulse_scalar /= (1 / self.radius + 1 / self.radius)
            impulse = impulse_scalar * normal
            self.vel += (impulse / self.radius)
            paddle.vel -= (impulse / self.radius)
            overlap = self.radius + self.radius - dist
            self.pos += (normal * overlap) / 2
            paddle.pos -= (normal * overlap) / 2

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                g.sound_handler.play_sound(sound_vel, self.pos[0], 'paddle')

    def draw(self):
        glow = max(0.0, 1.0 - (time.time() - self.last_dash_time) / g.GAMEPLAY_PARAMS['dash_duration'])
        color = g.interpolate_color(self.color, (255,255,255), glow)

        if self.charging_dash:
            charging_time = time.time() - self.charge_start_time
            charge_color_shift = min(1.0, charging_time / g.GAMEPLAY_PARAMS['dash_max_charge_time'])
            color = g.interpolate_color(color, (255,100,100), charge_color_shift)

        g.framework.draw_circle(self.pos, self.radius, color)
        g.framework.draw_circle(self.pos, int(self.radius / 2), g.interpolate_color(color, (0,0,0), 0.3))
        g.framework.draw_circle(self.pos, int(self.radius / 3), g.interpolate_color(color, (0,0,0), 0.1))

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