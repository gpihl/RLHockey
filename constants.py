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
    'paused': False,
    'player_2_human': False,
    'no_render': False,
    'no_sound': True,
    'field_split': False,
    'blocked_goals': False,
    # 'blocked_goals': True,
    'random_starting_locations': True,
    # 'random_starting_locations': False,
    'resolution': 3,
    'field_width': 3000,
    'field_height': 1500,
    # 'field_width': 2097,
    # 'field_height': 1049,
    'corner_radius': 249,
    'fps': 60,
    # 'fps': 165,
    # 'delta_t': 96,
    'delta_t': 92,
}

settings['goal_height'] = 400 * settings['field_width'] / 3000

settings['delta_t'] /= settings['fps']

rewards = {
    'time_reward': -0.4,
    'vel_reward': -0.01,
    # 'vel_reward': -0.1,
    'goal': 400,
    # 'team_mate_proximity': -0.2,
    'team_mate_proximity': -0.8,
    'wrong_side_of_puck': -0.8,
    'puck_proximity': 1.0,
    'goal_puck_proximity': 0.5,
    'shot': 0.8,
    'shot_toward_goal': 1.0,
    'pointless_motion': -0.4,
    'normalization': 1.0,
}

# rewards = {
#     'time_reward': -0,
#     'vel_reward': -0,
#     'goal': 400,
#     'puck_proximity': 0.0,
#     'goal_puck_proximity': 0.0,
#     'shot': 0.0,
#     'shot_toward_goal': 0,
#     'pointless_motion': -0,
#     'normalization': 1.0,
# }

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

