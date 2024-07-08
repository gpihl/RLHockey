import numpy as np
import os
import constants as c
import globals as g
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from collections import deque
import environment

class Model:
    def __init__(self, model, version, algorithm_name, model_name, team_size):
        self.model = model
        self.version = version
        self.algorithm_name = algorithm_name
        self.model_name = model_name
        self.team_size = team_size
        self.action_history = deque(maxlen=10)

    @staticmethod
    def get_algorithm(algorithm_name):
        match algorithm_name:
            case "PPO":
                return PPO
            case "SAC":
                return SAC
            case "TD3":
                return TD3
            case _:
                return None

    def update_action_history(self, action):
        self.action_history.append(action[2])

    def train_model(self, training_steps):
        print("Starting training round.")
        self.model.learn(total_timesteps=training_steps)
        print("Finished training round.")

    def save_model(self):
        search_path = Model.get_search_path(self.model_name, self.algorithm_name, self.team_size)
        save_path = f"{search_path}/{self.version+1}"
        print(f"Saving model: {save_path}")
        self.model.save(save_path)
        print("Model saved!")
        self.version += 1

    def get_action(self, observation):
        model_action = self.model.predict(observation)[0]
        model_action = self.process_action(model_action)
        return model_action

    def process_action(self, model_action):
        self.update_action_history(model_action)
        average_dash_action = np.mean(self.action_history, axis=0)
        model_action[2] = average_dash_action
        return model_action

    def get_name(self):
        return f"{self.algorithm_name}-{self.team_size}-{self.version}-{self.model_name}"

    @staticmethod
    def get_paddle_model(paddle):
        player = paddle.player
        team = paddle.team
        print(f"Getting model for paddle {player} on team {team}")

        model = None
        if c.settings["all_ai"]:
            model = Model.get_latest_model(c.training["model_name"], "PPO", c.settings["team_size"])
        elif team == 2 and not c.training["team_2_active"]:
            model = None
        elif (not (team == 1 and player == 1)) and c.training["alone"]:
            model = None
        elif team == 1 and player == 1:
            if c.settings["is_training"]:
                model = Model.get_latest_model(c.training["model_name"], "PPO", c.settings["team_size"])
        else:
            if c.settings["is_training"]:
                model = Model.get_random_model(c.training["model_name"], "PPO", c.settings["team_size"], paddle)
            else:
                model = Model.get_latest_model(c.training["model_name"], "PPO", c.settings["team_size"])

        if model is None:
            print("No model fetched")

        return model

    @staticmethod
    def get_environment():
        env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)
        return env

    @staticmethod
    def create_new_model(model_name, algorithm_name, team_size):
        print(f"Creating new model: {model_name}, {algorithm_name}, {team_size}")
        training_algorithm = Model.get_algorithm(algorithm_name)
        env = Model.get_environment()
        sb3_model = training_algorithm("MultiInputPolicy", env, learning_rate=c.training["learning_rate"], ent_coef=c.training["ent_coef"], verbose=1, device=g.device)
        version = 0
        model = Model(sb3_model, version, algorithm_name, model_name, team_size)
        print("Model created")
        return model

    @staticmethod
    def get_latest_model(model_name, algorithm_name, team_size):
        print(f"Trying to get latest model: {model_name}, {algorithm_name}, {team_size}")
        search_path = Model.get_search_path(model_name, algorithm_name, team_size)
        Model.ensure_path_exists(search_path)
        models = [f for f in os.listdir(search_path) if f.endswith(".zip")]
        if len(models) == 0:
            print("Couldn't find any matching models")
            return Model.create_new_model(model_name, algorithm_name, team_size)

        file_name = max(models, key=lambda x: int(Model.get_version(x)))
        model_path = os.path.join(search_path, file_name)
        sb3_model = Model.get_model(model_path, algorithm_name)
        version = Model.get_version(file_name)
        model = Model(sb3_model, version, algorithm_name, model_name, team_size)
        print(f"Latest model: {model_path} loaded!")
        return model

    @staticmethod
    def get_search_path(model_name, algorithm_name, team_size):
        search_path = f"{c.training['base_path']}/{algorithm_name}/{team_size}/{model_name}"
        return search_path

    @staticmethod
    def get_random_model(model_name, algorithm_name, team_size, paddle):
        print(f"Trying to get random model: {model_name}, {algorithm_name}, {team_size}")
        search_path = Model.get_search_path(model_name, algorithm_name, team_size)
        Model.ensure_path_exists(search_path)
        models = [f for f in os.listdir(search_path) if f.endswith(".zip")]
        if len(models) == 0:
            print("Couldn't find any matching models")
            return Model.create_new_model(model_name, algorithm_name, team_size)

        models = sorted(models, key=lambda x: int(x.split('.')[0]))

        if paddle.team == 1:
            variance = c.training["model_selection_variance_team"]
        else:
            variance = c.training["model_selection_variance_opponent"]

        random_index = max(0, len(models) - int(np.abs(np.random.normal(0, variance, 1)[0]) * len(models)) - 1)
        file_name = models[random_index]
        model_path = os.path.join(search_path, file_name)
        sb3_model = Model.get_model(model_path, algorithm_name)
        version = Model.get_version(file_name)
        model = Model(sb3_model, version, algorithm_name, model_name, team_size)
        print(f"Random model: {model_path} loaded!")
        return model

    @staticmethod
    def get_version(model_path):
        version = int(model_path.split(".")[0])
        return version

    @staticmethod
    def get_model(model_path, algorithm_name):
        training_algorithm = Model.get_algorithm(algorithm_name)
        env = Model.get_environment()
        custom_objects = { "learning_rate": c.training["learning_rate"], "ent_coef": c.training["ent_coef"] }
        sb3_model = training_algorithm.load(model_path, env, custom_objects=custom_objects)
        return sb3_model

    @staticmethod
    def ensure_path_exists(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

    # def save_model_name():
    #     save_text_to_file(g.current_model_name, "model_name/name.txt")



