import pygame
import helpers as h
import numpy as np

class Controls:
    def __init__(self):
        self.control_params = {
            'dash': 'x',
            'magnet': 'l1',
        }

        ps5_controller = {
            'x': 0,
            'l1': 9
        }

        self.controller_type = ps5_controller
        self.stick = self.init_controls()

    def init_controls(self):
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No joystick connected")
            return None

        stick = pygame.joystick.Joystick(0)
        stick.init()
        print(f"Joystick name: {stick.get_name()}")
        self.stick = stick
        return stick

    def button_pressed(self, action_name):
        button_idx = self.controller_type[self.control_params[action_name]]
        return self.stick.get_button(button_idx)

    def get_human_action(self):
        action = Controls.empty_action()

        if self.stick != None:
            action = self.get_joystick_action()

        keys = Controls.get_keys()
        if keys[pygame.K_w]:
            action['acceleration'][1] = -1.0
        if keys[pygame.K_s]:
            action['acceleration'][1] = 1.0
        if keys[pygame.K_a]:
            action['acceleration'][0] = -1.0
        if keys[pygame.K_d]:
            action['acceleration'][0] = 1.0
        if keys[pygame.K_LSHIFT] or keys[pygame.K_SPACE] or keys[pygame.K_RSHIFT]:
            action['dash'] = True

        return action

    def get_joystick_action(self):
        if self.stick == None:
            return None
        try:
            input_vector = np.array([self.stick.get_axis(0), self.stick.get_axis(1)])
            input_vector = self.apply_non_linear_response(input_vector)

            return {
                'acceleration': np.array([input_vector[0] * 1.0, input_vector[1] * 1.0]),
                'dash': self.button_pressed('dash'),
                'magnet': self.button_pressed('magnet'),
            }
        except:
            return Controls.empty_action()

    @staticmethod
    def apply_non_linear_response(input_vector, exponent=1.4):
        magnitude = np.linalg.norm(input_vector)
        modified_magnitude = np.power(magnitude, exponent)
        modified_magnitude = np.clip(modified_magnitude, 0, 1)

        if magnitude == 0:
            return np.zeros_like(input_vector)

        return input_vector * (modified_magnitude / magnitude)

    @staticmethod
    def get_keys():
        return pygame.key.get_pressed()

    @staticmethod
    def empty_action():
        return {
            'acceleration': np.array([0.0, 0.0]),
            'dash': False,
        }

    @staticmethod
    def game_action_from_model_action(model_action):
        action = {
            'acceleration': h.clip_vector_length_inplace(np.array([model_action[0], model_action[1]])),
            'dash': model_action[1] > 0.0,
        }
        return action