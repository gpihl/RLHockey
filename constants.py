resolutions = [
    (2560, 1440),
    (1920, 1080),
    (1470, 956),
    (1280, 720),
    (800, 400),
    (200, 100),
]

settings = {
    'is_training': False,
    'player_2_human': False,
    'no_render': False,
    'no_sound': True,
    'field_split': False,
    'blocked_goals': False,
    'random_starting_locations': False,
    'resolution': 4,
    'field_width': 2100,
    'field_height': 1050,
    # 'field_width': 1920,
    # 'field_height': 960,
    'goal_height': 300,
    'fps': 60,
    # 'fps': 165,
    # 'delta_t': 96,
    'delta_t': 92,
}

settings['delta_t'] /= settings['fps']

reward_policy = {
    'time_reward': -0.4,
    'acc_reward': -0.005,
    'goal': 110,    
    'puck_proximity': 0.0,    
    'goal_puck_proximity': 0.8,
    'shot': 0.0,    
    'shot_toward_goal': 0.3,
    'pointless_motion': -0.4,
    'normalization': 1.0,
}

training_params = {
    'training_steps': 8000,
    'learning_rate': 1.0e-4,
    'base_path': 'models',
    'training_iterations': 4000,
    'player_2_active': True,
    'algorithm': 'PPO',
}

gameplay_params = {
    'dash_cooldown': 0.4,
    'dash_impulse': 700,
    'dash_duration': 0.22,
    'dash_max_charge_time': 1.5,
}

MAX_PADDLE_SPEED = 40
MAX_PUCK_SPEED = 60

REWARD_COLOR = (255, 255, 255)
REWARD_FONT = None
REWARD_FONT_SIZE = 30
REWARD_POS = (20, 10)

TIME_COLOR = (255, 255, 255)
TIME_FONT = None
TIME_FONT_SIZE = 120
TIME_POS = (settings['field_width']/2, 55)

STEPS_LEFT_COLOR = (255, 255, 255)
STEPS_LEFT_FONT = None
STEPS_LEFT_FONT_SIZE = 30
STEPS_LEFT_POS = (settings['field_width'] - 100, 30)

SCORE_COLOR = (255, 255, 255)
SCORE_FONT = None
SCORE_FONT_SIZE = 85
SCORE_POS = (180, 45)