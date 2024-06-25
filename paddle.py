import numpy as np
import random
import globals as g
import time
from collections import deque

class Paddle:
    def __init__(self, player):
        self.player = player
        self.color = g.PADDLE_COLOR_1 if self.player == 1 else g.PADDLE_COLOR_2
        self.pos = np.array([0.0,0.0])
        self.last_pos = np.array([0.0,0.0])
        self.magnetic_effect_active = False
        self.reset(training=False)
        self.last_dash_time = 0
        self.velocity_history = deque(maxlen=20)  # Stores the last 10 velocities by default
        self.average_velocity = np.array([0.0, 0.0])        

    def reset(self, training):
        if not training:
            self.pos = self.get_starting_pos_regular()
        else:
            if g.TRAINING_PARAMS['random_starting_locations']:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular()           

        self.vel = np.array([0, 0])

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*g.PADDLE_RADIUS, g.WIDTH - 2*g.PADDLE_RADIUS), 
                                 random.uniform(2*g.PADDLE_RADIUS, g.HEIGHT - 2*g.PADDLE_RADIUS)], 
                                 dtype=np.float32)
        return starting_pos
    
    def get_starting_pos_regular(self):
        if self.player == 1:
            starting_pos = np.array([g.PADDLE_RADIUS*3.0, g.HEIGHT // 2])
        else:
            starting_pos = np.array([g.WIDTH - g.PADDLE_RADIUS*3.0, g.HEIGHT // 2])

        return starting_pos
    
    def update_velocity_history(self):
        """
        Update the velocity history and calculate the average velocity.
        """
        self.velocity_history.append(self.vel.copy())
        self.average_velocity = np.mean(self.velocity_history, axis=0)  
    
    def dash(self, puck):
        current_time = time.time()
        if current_time - self.last_dash_time > g.GAMEPLAY_PARAMS['dash_cooldown']:
            self.last_dash_time = current_time


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
            # Apply the impulse
            self.vel += dash_direction * g.GAMEPLAY_PARAMS['dash_impulse'] * (average_speed / g.MAX_PADDLE_SPEED)

    def is_dashing(self):
        return time.time() - self.last_dash_time < g.GAMEPLAY_PARAMS['dash_duration']

    def apply_aim_assist(self, puck, max_assist_strength=0.4):
        """
        Apply aim assist to the paddle's movement, proportional to velocity and proximity to the puck.
        
        :param puck: The Puck object
        :param max_assist_strength: Maximum strength of the aim assist (0 to 1)
        """
        # Predict puck position in the near future
        future_time = 0.5  # Look 0.5 seconds ahead
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
        velocity_factor = min(np.linalg.norm(self.vel) / max_speed, 1.0)
        velocity_factor = velocity_factor ** 2
        proximity_factor = 1 - min(distance_to_puck / (g.WIDTH / 2), 1)  # Assumes field width as max distance
        proximity_factor = proximity_factor ** 4
        assist_strength = max_assist_strength * velocity_factor * proximity_factor

        # Blend current velocity with ideal velocity
        self.vel = (1 - assist_strength) * self.vel + assist_strength * ideal_vel

    def set_magnetic_effect(self, active):
        self.magnetic_effect_active = active

    def calculate_magnetic_force(self, puck_pos, ambient_force=0.1):
        max_distance = (g.PUCK_RADIUS + g.PADDLE_RADIUS) * 1.5
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

    def control(self, ax, ay):
        acc = np.array([ax, ay])
        self.vel = self.vel.astype(np.float64)
        self.vel += acc * g.DELTA_T * g.PADDLE_ACC / 1.6

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
    
    def update(self, training):
        self.last_pos = self.pos.copy()

        # if not dashing, apply friction
        if not self.is_dashing():
            self.vel *= (g.PADDLE_FRICTION ** g.DELTA_T)
 
        speed = np.linalg.norm(self.vel)
        if speed > g.MAX_PADDLE_SPEED:
            self.vel = (self.vel / speed) * g.MAX_PADDLE_SPEED

        self.update_velocity_history()
        # if training and self.player == 2:
        #     self.vel += np.random.normal(0, 0.4, 2) * g.DELTA_T
        
        self.pos += self.vel * g.DELTA_T
        
        if g.TRAINING_PARAMS['field_split']:
            if self.player == 1:
                left_wall = g.PADDLE_RADIUS
                right_wall = g.WIDTH / 2 - g.PADDLE_RADIUS
            else:
                left_wall = g.WIDTH / 2 + g.PADDLE_RADIUS
                right_wall = g.WIDTH - g.PADDLE_RADIUS
        else:
            left_wall = g.PADDLE_RADIUS
            right_wall = g.WIDTH - g.PADDLE_RADIUS

        if self.pos[0] < left_wall:
            self.pos[0] = left_wall
            self.vel[0] = 0
        elif self.pos[0] > right_wall:
            self.pos[0] = right_wall
            self.vel[0] = 0

        if self.pos[1] < g.PADDLE_RADIUS:
            self.pos[1] = g.PADDLE_RADIUS
            self.vel[1] = 0
        elif self.pos[1] > g.HEIGHT - g.PADDLE_RADIUS:
            self.pos[1] = g.HEIGHT - g.PADDLE_RADIUS
            self.vel[1] = 0

    def handle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        if dist < g.PADDLE_RADIUS + g.PADDLE_RADIUS:
            normal = (self.pos - paddle.pos) / dist
            relative_velocity = self.vel - paddle.vel
            velocity_along_normal = np.dot(relative_velocity, normal)
            if velocity_along_normal > 0:
                return

            impulse_scalar = -(2) * velocity_along_normal
            impulse_scalar /= (1 / g.PADDLE_RADIUS + 1 / g.PADDLE_RADIUS)
            impulse = impulse_scalar * normal
            self.vel += (impulse / g.PADDLE_RADIUS)
            paddle.vel -= (impulse / g.PADDLE_RADIUS)
            overlap = g.PADDLE_RADIUS + g.PADDLE_RADIUS - dist
            self.pos += (normal * overlap) / 2
            paddle.pos -= (normal * overlap) / 2

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                sound_vel = np.abs(sound_vel)
                g.sound_handler.play_sound(sound_vel, self.pos[0], 'paddle')

    def draw(self, screen):
        glow = max(0.0, 1.0 - (time.time() - self.last_dash_time) / g.GAMEPLAY_PARAMS['dash_duration'])
        # glow 
        color = g.interpolate_color(self.color, (255,255,255), glow)
        g.draw_circle(self.pos, g.PADDLE_RADIUS, color, screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 2), g.interpolate_color(color, (0,0,0), 0.3), screen)
        g.draw_circle(self.pos, int(g.PADDLE_RADIUS / 3), g.interpolate_color(color, (0,0,0), 0.1), screen)