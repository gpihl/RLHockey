resolutions = [
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 360),
    (320, 180),
    (160, 90),
]

update_multiplier = 7
updates_per_delta_t = update_multiplier * 15 / 23

settings = {
    "round_time": 10,
    "is_training": False,
    "paused": False,
    "no_render": False,
    "no_sound": True,
    "field_split": False,
    "blocked_goals": False,
    # "blocked_goals": True,
    "random_starting_locations": True,
    # "random_starting_locations": False,
    # "random_starting_locations_probability": 0.5,
    "random_starting_locations_probability": 0.99,
    "resolution": 0,
    "field_width": 3400,
    "field_height": 1700,
    "corner_radius": 249,
    "fps": 60,
    # "original_delta_t": 86,
    "original_delta_t": 76,
    "delta_t": 0,
    "team_size": 2,
    "agent_control_training": ["ai", "ai", "ai", "ai"],
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

fixed_training_regime = None

training = {
    "training_steps": 8000,
    # "learning_rate": 0.3e-3,
    "learning_rate": 1.0e-4,
    # "ent_coef": 0.005,
    "ent_coef": 0.0,
    "model_selection_variance_opponent": 0.30,
    "model_selection_variance_team": 0.0001,
    "base_path": "models",
    "algorithm": "PPO",
}

model_names = ["DefenderNormal", "ShooterNormal", "ShooterNormal", "ShooterNormal"]

gameplay = {
    "dash_cooldown": 0.4,
    "dash_impulse": 65,
    "dash_duration": 0.22,
    "dash_max_charge_time": 1.0,
    "max_paddle_speed": 40,
    "max_puck_speed": 75,
}

