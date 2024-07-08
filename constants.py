resolutions = [
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 360),
    (320, 180),
    (160, 90),
]

settings = {
    "is_training": False,
    "paused": False,
    "player_2_human": False,
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
    # "field_width": 2097,
    # "field_height": 1049,
    "corner_radius": 249,
    "fps": 60,
    # "fps": 165,
    # "delta_t": 96,
    "delta_t": 92,
    "team_size": 2,
}
settings["goal_height"] = 400 * settings["field_width"] / 3000
settings["delta_t"] /= settings["fps"]

rewards = {
    "time_reward": -0,
    "vel_reward": -0,
    "goal": 500000,
    "team_mate_proximity": 0,
    "wrong_side_of_puck": 0,
    "puck_proximity": 0.0,
    # "puck_vel_toward_goal": 0.8,
    "puck_vel_toward_goal": 0.0,
    "goal_puck_proximity": 0,
    "shot": 0.0,
    "shot_toward_goal": 0.0,
    "dash": 20000,
    "normalization": 1.0
}

# rewards = {
#     "time_reward": -0.3,
#     "vel_reward": -0.07,
#     "goal": 300,
#     "team_mate_proximity": -0.1,
#     "wrong_side_of_puck": -0.3,
#     "puck_proximity": 0.3,
#     "goal_puck_proximity": 0.2,
#     "shot": 3.0,
#     "shot_toward_goal": 6.0,
#     "normalization": 1.0
# }




training = {
    "training_steps": 8000,
    # "learning_rate": 0.2e-4,
    "learning_rate": 1.0e-4,
    # "learning_rate": 2.0e-4,
    "ent_coef": 0.1,
    # "ent_coef": 0.0,
    "base_path": "models",
    "training_iterations": 4000,
    "team_2_active": False,
    # "team_2_active": True,
    # "alone": True,
    "alone": False,
    "algorithm": "PPO",
    "model_name": "Albert"
}

gameplay = {
    "dash_cooldown": 0.4,
    "dash_impulse": 700,
    "dash_duration": 0.22,
    "dash_max_charge_time": 1.5,
    "max_paddle_speed": 40,
    "max_puck_speed": 60,
}

