import numpy as np
import random
import globals as g

class Puck: 
    def __init__(self):
        self.pos = self.get_starting_pos_regular(2)
        self.prev_puck_start_pos = self.pos
        self.shot_reward = 0
        self.shot_on_goal_reward = 0
        self.vel = np.zeros(2)
        self.rot_vel = 0.0
        self.rot = 0.0
        self.last_collider = None
        self.reset()

    def reset(self, last_scorer=2):
        if not g.SETTINGS['is_training']:
            self.pos = self.get_starting_pos_regular(last_scorer)
        else:
            if g.TRAINING_PARAMS['random_starting_locations']:
                self.pos = self.get_starting_pos_random()
            else:
                self.pos = self.get_starting_pos_regular(last_scorer)
            
            # self.pos = self.get_starting_pos_moved_a_bit()

        self.vel = np.zeros(2)
        self.rot_vel = 10.0
        self.rot = 0.0

    def get_starting_pos_random(self):
        starting_pos = np.array([random.uniform(2*g.PUCK_RADIUS, g.WIDTH - 2*g.PUCK_RADIUS), 
                                 random.uniform(2*g.PUCK_RADIUS, g.HEIGHT - 2*g.PUCK_RADIUS)], 
                                 dtype=np.float32)
        
        return starting_pos
    
    def get_starting_pos_moved_a_bit(self):
        var = 10
        delta = np.array([random.uniform(-var, var), random.uniform(-var, var)],dtype=np.float32)
        new_starting_pos = self.prev_puck_start_pos + delta
        self.prev_puck_start_pos += delta
        self.prev_puck_start_pos = np.clip(self.prev_puck_start_pos, g.PUCK_RADIUS*2 + 20, g.WIDTH - 2*g.PUCK_RADIUS - 20)
        new_starting_pos = np.clip(new_starting_pos, g.PUCK_RADIUS*2 + 20, g.WIDTH - 2*g.PUCK_RADIUS - 20)
        return new_starting_pos
    
    def get_starting_pos_regular(self, last_scorer):
        if last_scorer == 2:
            return np.array([g.WIDTH / 4, g.HEIGHT / 2])
        elif last_scorer == 1:
            return np.array([g.WIDTH * 3 / 4, g.HEIGHT / 2])

    def update(self, paddles):
        self.vel *= (g.PUCK_FRICTION ** g.DELTA_T)
        self.vel += np.random.normal(0, 0.005, 2) * g.DELTA_T

        magnus_coefficient = 0.001
        magnus_force = magnus_coefficient * np.array([-self.vel[1], self.vel[0]]) * self.rot_vel
        self.vel += magnus_force * g.DELTA_T

        self.limit_speed()
        self.pos += self.vel * g.DELTA_T

        self.rot_vel *= (g.PUCK_FRICTION ** g.DELTA_T)
        self.rot += self.rot_vel

        for paddle in paddles:
            self.handle_paddle_collision(paddle)
            magnetic_force = paddle.calculate_magnetic_force(self.pos)
            self.vel += magnetic_force * g.DELTA_T

        self.handle_wall_collision()

    def handle_wall_collision(self):
        sound_vel = 0

        if self.pos[1] < g.PUCK_RADIUS:
            self.pos[1] = g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]
            sound_vel = self.vel[1]
        elif self.pos[1] > g.HEIGHT - g.PUCK_RADIUS:
            self.pos[1] = g.HEIGHT - g.PUCK_RADIUS
            self.vel[1] = -self.vel[1]
            sound_vel = self.vel[1]

        if self.pos[0] < g.PUCK_RADIUS:
            self.pos[0] = g.PUCK_RADIUS
            self.vel[0] = -self.vel[0]
            sound_vel = self.vel[1]
        elif self.pos[0] > g.WIDTH - g.PUCK_RADIUS:
            self.pos[0] = g.WIDTH - g.PUCK_RADIUS
            self.vel[0] = -self.vel[0]
            sound_vel = self.vel[1]

        if sound_vel != 0:
            sound_vel = np.linalg.norm(self.vel)
            g.sound_handler.play_sound(sound_vel, self.pos[0], 'table_hit')

    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if speed > g.MAX_PUCK_SPEED:
            self.vel = (self.vel / speed) * g.MAX_PUCK_SPEED

    def check_paddle_collision(self, paddle):
        dist = np.linalg.norm(self.pos - paddle.pos)
        return dist < g.PUCK_RADIUS + paddle.radius
    
    def collect_shot_reward(self, reward_type):
        if reward_type == 'vel_2_goal':
            reward = self.shot_on_goal_reward
            self.shot_on_goal_reward = 0
        elif reward_type == 'ball_velocity':
            reward = self.shot_reward
            self.shot_reward = 0

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
            
            impulse_scalar = -(1 + g.PUCK_RESTITUTION) * velocity_along_normal
            impulse_scalar /= (1 / g.PUCK_RADIUS + 1 / g.PADDLE_RADIUS)
            impulse = 0.9 * impulse_scalar * normal

            self.vel += impulse / g.PUCK_RADIUS
            self.limit_speed()

            tangent = np.array([-normal[1], normal[0]])
            velocity_along_tangent = np.dot(relative_velocity, tangent)            

            rotational_impulse = np.cross(normal, tangent) * velocity_along_tangent
            rotational_impulse = 0 if np.abs(rotational_impulse) < 0.5 else rotational_impulse
            self.rot_vel += rotational_impulse * 0.1

            overlap = g.PUCK_RADIUS + paddle.radius - dist
            paddle.pos -= normal * (overlap / 2)
            paddle.vel -= 0.8 * impulse / g.PADDLE_RADIUS
            paddle.limit_speed()
            self.pos += normal * (overlap / 2)

            self.shot_on_goal_rewardl = self.vel[0] - prev_vel[0]            
            self.shot_reward = np.linalg.norm(relative_velocity)
            
            if paddle.player == 2:
                self.shot_reward *= -1

            sound_vel = np.linalg.norm(relative_velocity)
            if sound_vel != 0:
                g.sound_handler.play_sound(sound_vel / 4, self.pos[0], 'paddle')
                g.sound_handler.play_sound(sound_vel, self.pos[0], 'table_hit')

    def draw(self):
        themed_color = g.interpolate_color(g.PUCK_COLOR, g.sound_handler.current_color(), 0.3)
        red_level = np.linalg.norm(self.vel) / (g.MAX_PUCK_SPEED + 10)
        red_level = max(min(red_level, 1.0), 0.0)
        puck_color = g.interpolate_color(themed_color, (255, 0, 0), red_level)
        puck_color = g.modify_hsl(puck_color, 0, 0, 0.2)
        g.framework.draw_circle(self.pos, g.PUCK_RADIUS, puck_color)
        g.framework.draw_circle(self.pos, int(7*g.PUCK_RADIUS / 9), g.interpolate_color(puck_color, (0,0,0), 0.05))
        g.framework.draw_circle(self.pos, int(8*g.PUCK_RADIUS / 9), g.interpolate_color(puck_color, (0,0,0), 0.2))
        g.framework.draw_rotated_line(self.pos, g.PUCK_RADIUS * 2.1, -self.rot, puck_color, int(g.PUCK_RADIUS / 2.5))
        # g.framework.draw_text('RL', 'puck', g.PUCK_TEXT_COLOR, self.pos, True, self.rot)