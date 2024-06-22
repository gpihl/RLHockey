import pygame.gfxdraw
import os

# Training params
REWARD_POLICY = {   
    'time_reward': -1.0,
    'acc_reward': -1.0,
    'player_1_goal': 400,
    'player_2_goal': -400,
    'ball_proximity': 1.0,
    'ball_velocity': 2.0,
    'ball_velocity_toward_goal': 4.0,
    'normalization': 100,
}

TRAINING_PARAMS = {
    'training_steps': 5000,
    'learning_rate': 1.0e-4,
    'model_name': 'newester_test',
    'base_path': 'models',
    'training_iterations': 10,
}

# Colors
BG_COLOR = (43, 50, 80)

# Display
WIDTH, HEIGHT = 800, 400
HIGH_FPS = 60000
LOW_FPS = 60

# Fonts
REWARD_COLOR = (154, 120, 134)
REWARD_FONT = None
REWARD_FONT_SIZE = 30
REWARD_POS = (25, 10)

TIME_COLOR = (200, 160, 174)
TIME_FONT = None
TIME_FONT_SIZE = 70
TIME_POS = (WIDTH/2, 30)

STEPS_LEFT_COLOR = (154, 120, 134)
STEPS_LEFT_FONT = None
STEPS_LEFT_FONT_SIZE = 30
STEPS_LEFT_POS = (WIDTH - 70, 30)

# Paddle
PADDLE_COLOR_1 = (51, 153, 255)
PADDLE_COLOR_2 = (0, 204, 102)
PADDLE_RADIUS = 20
PADDLE_FRICTION = 0.92
PADDLE_SPEED = 1.0
MAX_PADDLE_SPEED = 25

# Puck
PUCK_COLOR = (124, 254, 253)
MAX_PUCK_SPEED = 25
PUCK_START_SPEED = 4
PUCK_RADIUS = 15
PUCK_FRICTION = 0.99

# Match
TIME_LIMIT = 15 * 60

# Helper functions
def draw_circle(pos, radius, color, screen, aa=True):
    pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), radius, color)
    if aa:
        pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), radius, (0,0,0))

def interpolate_color(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )    

def save_dict_to_file(data_dict, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(str(data_dict))
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_latest_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(base_path, latest_model)

def get_next_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        next_model_number = 1
    else:
        latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_number = int(latest_model.split('_')[-1].split('.')[0])
        next_model_number = latest_number + 1
    return os.path.join(base_path, f"{prefix}_{next_model_number}.zip")