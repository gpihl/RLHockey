import pyray as pr
import helpers as h
import numpy as np

class Controls:
    def __init__(self):
        self.control_params = {
            "dash": "x",
            "magnet": "l1",
        }
        ps5_controller = {
            "x": 7,
            "l1": 9
        }
        self.controller_type = ps5_controller
        self.stick = None
        self.init_controls()

    def init_controls(self):
        if pr.is_gamepad_available(0):
            print(f"Joystick name: {pr.get_gamepad_name(0)}")
            self.stick = 0

    def button_pressed(self, action_name):
        button_idx = self.controller_type[self.control_params[action_name]]
        return pr.is_gamepad_button_down(0, button_idx)

    def get_human_action(self):
        action = Controls.empty_action()
        if self.stick is not None:
            action = self.get_joystick_action()

        if pr.is_key_down(pr.KEY_W):
            action["acceleration"][1] = -1.0
        if pr.is_key_down(pr.KEY_S):
            action["acceleration"][1] = 1.0
        if pr.is_key_down(pr.KEY_A):
            action["acceleration"][0] = -1.0
        if pr.is_key_down(pr.KEY_D):
            action["acceleration"][0] = 1.0
        if pr.is_key_down(pr.KEY_LEFT_SHIFT) or pr.is_key_down(pr.KEY_SPACE) or pr.is_key_down(pr.KEY_RIGHT_SHIFT):
            action["dash"] = True
        return action

    def get_joystick_action(self):
        if self.stick is None:
            return None
        try:
            input_vector = np.array([
                pr.get_gamepad_axis_movement(0, 0),
                pr.get_gamepad_axis_movement(0, 1)
            ])
            input_vector = self.apply_non_linear_response(input_vector)
            return {
                "acceleration": np.array([input_vector[0] * 1.0, input_vector[1] * 1.0]),
                "dash": self.button_pressed("dash"),
                "magnet": self.button_pressed("magnet"),
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
        return {
            pr.KEY_W: pr.is_key_down(pr.KEY_W),
            pr.KEY_S: pr.is_key_down(pr.KEY_S),
            pr.KEY_A: pr.is_key_down(pr.KEY_A),
            pr.KEY_D: pr.is_key_down(pr.KEY_D),
            pr.KEY_LEFT_SHIFT: pr.is_key_down(pr.KEY_LEFT_SHIFT),
            pr.KEY_SPACE: pr.is_key_down(pr.KEY_SPACE),
            pr.KEY_RIGHT_SHIFT: pr.is_key_down(pr.KEY_RIGHT_SHIFT),
        }

    @staticmethod
    def empty_action():
        return {
            "acceleration": np.array([0.0, 0.0]),
            "dash": False,
        }

    @staticmethod
    def game_action_from_model_action(model_action):
        if model_action is None:
            return None

        action = {
            "acceleration": h.clip_vector_length_inplace(np.array([model_action[0], model_action[1]])),
            "dash": model_action[2] > 0,
        }
        return action