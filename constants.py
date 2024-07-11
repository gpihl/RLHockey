resolutions = [
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 360),
    (320, 180),
    (160, 90),
]

settings = {
    "round_time": 20,
    # "round_time": 60,
    "is_training": False,
    "paused": False,
    "no_render": False,
    "no_sound": True,
    "field_split": False,
    "blocked_goals": False,
    # "blocked_goals": True,
    "random_starting_locations": True,
    # "random_starting_locations": False,
    "resolution": 0,
    "field_width": 3000,
    "field_height": 1500,
    "corner_radius": 249,
    "fps": 60,
    "original_delta_t": 92,
    "delta_t": 0,
    "team_size": 2,
    # "agent_control_training": ["ai", "ai", "ai", "ai"],
    "agent_control_training": ["ai", "off", "off", "off"],
    # "agent_control_regular": ["ai", "ai", "ai", "ai"],
    "agent_control_regular": ["human", "ai", "ai", "ai"],
}
settings["goal_height"] = 400 * settings["field_width"] / 3000
settings["delta_t"] = settings["original_delta_t"] / settings["fps"]

training_regime_probabilities = [
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]

training_regimes = [
    ["ai", "off", "off", "off"],
    ["ai", "ai", "off", "off"],
    ["ai", "off", "ai", "off"],
    ["ai", "ai", "ai", "off"],
    ["ai", "ai", "ai", "ai"],
]

# rewards = {
#     "time_reward": -0.0,
#     "vel_reward": -0.0,
#     "goal": 1000,
#     "team_mate_proximity": -0.3,
#     "wrong_side_of_puck": -0.0,
#     "puck_proximity": 0.0,
#     "puck_vel_toward_goal": 0.0,
#     "goal_puck_proximity": 0.0,
#     "shot": 0.0,
#     "shot_toward_goal": 0.0,
#     "dash": 0.0,
#     "normalization": 1.0
# }

# bra!
# rewards_start = {
#     "time_reward": -0.0,
#     "vel_reward": -0.0,
#     "goal": 1000,
#     "team_mate_proximity": -0.0,
#     "wrong_side_of_puck": -0.01,
#     "puck_proximity": 0.5,
#     "puck_vel_toward_goal": 0.0,
#     "goal_puck_proximity": 0.0,
#     "shot": 5.0,
#     "shot_toward_goal": 2.0,
#     "dash": 0.0,
#     "normalization": 1.0
# }

rewards_start = {'shot': 2.9393580245987914, 'dash': 0.0, 'puck_proximity': 0.29393580245987916, 'goal_puck_proximity': 0.0, 'team_mate_proximity': 0.0, 'wrong_side_of_puck': -0.005878716049197583, 'shot_toward_goal': 1.1757432098395166, 'normalization': 1.0, 'puck_vel_toward_goal': 0.0, 'goal': 1000.0, 'vel_reward': 0.0, 'time_reward': 0.0}

rewards_end = {
    "time_reward": -0.0,
    "vel_reward": -0.0,
    "goal": 1000,
    "team_mate_proximity": -0.0,
    "wrong_side_of_puck": -0.0,
    "puck_proximity": 0.0,
    "puck_vel_toward_goal": 0.0,
    "goal_puck_proximity": 0.0,
    "shot": 0.0,
    "shot_toward_goal": 0.0,
    "dash": 0.0,
    "normalization": 1.0
}

rewards = rewards_start

time_to_reach_end_reward = 60 * 60 * 2 * 10

training = {
    "training_steps": 8000,
    # "learning_rate": 0.2e-4,
    "learning_rate": 1.0e-4,
    # "learning_rate": 2.0e-4,
    # "ent_coef": 0.05,
    "ent_coef": 0.0,
    "model_selection_variance_opponent": 0.35,
    "model_selection_variance_team": 0.05,
    "base_path": "models",
    "training_iterations": 4000,
    "algorithm": "PPO",
    "model_name": "Brum"
}

gameplay = {
    "dash_cooldown": 0.4,
    "dash_impulse": 700,
    "dash_duration": 0.22,
    "dash_max_charge_time": 1.5,
    "max_paddle_speed": 40,
    "max_puck_speed": 60,
}

