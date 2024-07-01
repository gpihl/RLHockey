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
    # 'blocked_goals': True,
    # 'random_starting_locations': True,
    'random_starting_locations': False,
    'resolution': 3,
    'field_width': 2096,
    'field_height': 1048,
    'corner_radius': 248,
    # 'field_width': 1920,
    # 'field_height': 960,
    'goal_height': 300,
    'fps': 60,
    # 'fps': 165,
    # 'delta_t': 96,
    'delta_t': 92,
}

settings['delta_t'] /= settings['fps']

rewards = {
    'time_reward': -0.4,
    'acc_reward': -0.005,
    'goal': 240,
    'puck_proximity': 0.0,
    'goal_puck_proximity': 0.8,
    'shot': 0.0,
    'shot_toward_goal': 0.3,
    'pointless_motion': -0.4,
    'normalization': 1.0,
}

training = {
    'training_steps': 8000,
    'learning_rate': 1.0e-4,
    'base_path': 'models',
    'training_iterations': 4000,
    'player_2_active': True,
    'algorithm': 'PPO',
}

gameplay = {
    'dash_cooldown': 0.4,
    'dash_impulse': 700,
    'dash_duration': 0.22,
    'dash_max_charge_time': 1.5,
    'max_paddle_speed': 40,
    'max_puck_speed': 60,
}

