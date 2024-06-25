import pygame.gfxdraw
import os
import torch
import random
import numpy as np
from sound_handler import SoundHandler
from framework import Framework
from ui import UI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training params

SETTINGS = {
    'is_training': False,
    'player_2_human': False,
}

REWARD_POLICY = {   
    'time_reward': -0.05,
    'acc_reward': -0.02,
    'player_1_goal': 20,
    'player_2_goal': -20,
    'ball_proximity': 0.25,
    'ball_velocity': 0.5,
    'ball_vel_2_goal': 0.3,
    'center': -0.01,
    'dist_to_player': 0.1,
    'pointless_motion': -0.4,
    'dash': -10.0,
    'normalization': 1.0,    
}

TRAINING_PARAMS = {
    'training_steps': 15000,
    'learning_rate': 1.0e-4,
    'model_name': 'hej',
    'base_path': 'models',
    'training_iterations': 4000,
    'player_2_active': True,
    'blocked_goals': False,
    'random_starting_locations': True,
    'no_render': False,
    'no_sound': True,
    'field_split': False,
    'device': 'cpu',
    'algorithm': 'PPO',
    'dash_enabled': False,
}

TRAINING_PARAMS['model_name'] += "-" + TRAINING_PARAMS['algorithm']

GAMEPLAY_PARAMS = {
    'dash_cooldown': 0.5,
    'dash_impulse': 700,
    'dash_duration': 0.22,
    'dash_max_charge_time': 0.5,
}

CONTROLS_PARAMS = {
    'dash': 'x',
    'magnet': 'l1',
}

PS5_CONTROLLER = {
    'x': 0,
    'l1': 9
}

controller = PS5_CONTROLLER

# Colors
BG_COLOR = (43, 50, 80)

# Display
WIDTH, HEIGHT = 2000, 1000
HIGH_FPS = 60000
LOW_FPS = 60

# Fonts
REWARD_COLOR = (154, 120, 134)
REWARD_FONT = None
REWARD_FONT_SIZE = 30
REWARD_POS = (20, 10)

TIME_COLOR = (200, 160, 174)
TIME_FONT = None
TIME_FONT_SIZE = 70
TIME_POS = (WIDTH/2, 30)

STEPS_LEFT_COLOR = (154, 120, 134)
STEPS_LEFT_FONT = None
STEPS_LEFT_FONT_SIZE = 30
STEPS_LEFT_POS = (WIDTH - 100, 30)

# Paddle
PADDLE_COLOR_1 = (51, 153, 255)
PADDLE_COLOR_2 = (0, 204, 102)
PADDLE_RADIUS = int(30 * WIDTH / 800)
PADDLE_FRICTION = 0.86
PADDLE_ACC = 2.0
MAX_PADDLE_SPEED = 40

# Puck
PUCK_COLOR = (124, 254, 253)
MAX_PUCK_SPEED = 60
PUCK_START_SPEED = 4
PUCK_RADIUS = int(20 * WIDTH / 800)
PUCK_FRICTION = 0.996
PUCK_RESTITUTION = 0.95

# Goal
GOAL_HEIGHT = int(140 * WIDTH / 800)

# Match
TIME_LIMIT = 60 * LOW_FPS

# Physics
DELTA_T = 0.80 * 120 / LOW_FPS

# Singletons
sound_handler = SoundHandler()
framework = Framework()
ui = UI()

# Helper functions
def save_text_to_file(text, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(text + '\n\n')
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_latest_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(base_path, latest_model)

def get_random_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        return None

    models.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
    random_index = max(0, len(models) - int(np.abs(np.random.normal(0, 0.25, 1)[0]) * len(models)) - 1)
    print(f"random_index: {random_index}")
    random_model = models[random_index]
    return os.path.join(base_path, random_model)

def get_next_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        next_model_number = 1
    else:
        latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_number = int(latest_model.split('_')[-1].split('.')[0])
        next_model_number = latest_number + 1
    return os.path.join(base_path, f"{prefix}_{next_model_number}.zip")

def button_pressed(action_name):
    button_idx = controller[CONTROLS_PARAMS[action_name]]
    return joystick.get_button(button_idx)

def get_human_action():
    action = empty_action()

    if joystick != None:
        action = get_joystick_action()

    keys = get_keys()
    if keys[pygame.K_w]:
        action['acceleration'][1] = -PADDLE_ACC
    if keys[pygame.K_s]:
        action['acceleration'][1] = PADDLE_ACC
    if keys[pygame.K_a]:
        action['acceleration'][0] = -PADDLE_ACC
    if keys[pygame.K_d]:
        action['acceleration'][0] = PADDLE_ACC
    
    return action

def get_joystick_action():
    if joystick == None:
        return None
            
    input_vector = np.array([joystick.get_axis(0), joystick.get_axis(1)])
    input_vector = apply_non_linear_response(input_vector)

    return {
        'acceleration': np.array([input_vector[0] * PADDLE_ACC, input_vector[1] * PADDLE_ACC]),
        'dash': button_pressed('dash'),
        'magnet': button_pressed('magnet'),
    }

def apply_non_linear_response(input_vector, exponent=1.5):
    magnitude = np.linalg.norm(input_vector)
    modified_magnitude = np.power(magnitude, exponent)
    modified_magnitude = np.clip(modified_magnitude, 0, 1)

    if magnitude == 0:
        return np.zeros_like(input_vector)
    
    return input_vector * (modified_magnitude / magnitude)

def get_keys():
    return pygame.key.get_pressed()

def init_controls():
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick connected")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Joystick name: {joystick.get_name()}")
    return joystick

def empty_action():
    return {
        'acceleration': np.array([0.0, 0.0]),
        'dash': False,
        'magnet': False,
    }

def game_action_from_model_action(model_action):
    return {
        'acceleration': np.array([model_action[0], model_action[1]]),
        'dash': model_action[2] > 0.9,
        'magnet': False,
    }

def interpolate_color(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )

joystick = init_controls()