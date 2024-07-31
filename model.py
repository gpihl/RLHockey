import numpy as np
import os
import constants as c
import globals as g
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from collections import deque
import environment
import json
from reward import Reward

class Model:
    def __init__(self, model, version, algorithm_name, model_name, team_size, environment):
        self.model = model
        self.environment = environment
        self.version = version
        self.algorithm_name = algorithm_name
        self.model_name = model_name
        self.team_size = team_size
        self.score_ratios = []
        self.action_history = deque(maxlen=5)

    @staticmethod
    def get_algorithm(algorithm_name):
        match algorithm_name:
            case "PPO":
                return PPO
            case "SAC":
                return SAC
            case "TD3":
                return TD3
            case "RecurrentPPO":
                return RecurrentPPO
            case _:
                return None

    def add_score(self, score):
        if score[1] == 0:
            return

        score_ratio = float(score[0]) / float(score[1])
        self.score_ratios.append((self.version, round(score_ratio, 3), score[0], score[1]))
        sorted_ratios = sorted(self.score_ratios, key=lambda x: x[1], reverse=True)

        with open(f"{self.get_search_path_self()}/score_ratios.txt", "w") as file:
            # file.write(text + "\n\n")
            sorted_lines = list(map(lambda x: f"{x[0]}: {x[1]} | ({x[2]} - {x[3]})\n", sorted_ratios))
            file.write("Sorted by score ratio:\n\n")
            file.writelines(sorted_lines)
            lines = list(map(lambda x: f"{x[0]}: {x[1]} | ({x[2]} - {x[3]})\n", self.score_ratios))
            file.write("\n\n\nSorted by version:\n\n")
            file.writelines(lines)

    def update_action_history(self, action):
        self.action_history.append(action[2])

    def train_model(self, training_steps):
        print("Starting training round.")
        self.model.learn(total_timesteps=training_steps)
        print("Finished training round.")

    def save_model(self):
        def simple_clip_range(progress_remaining):
            return 0.2 * (1 - progress_remaining)

        self.model.clip_range = simple_clip_range
        search_path = Model.get_search_path(self.model_name, self.algorithm_name, self.team_size)

        model_save_path = f"{search_path}/{self.version+1}/model"
        print(f"Saving model: {model_save_path}")
        self.model.save(model_save_path)
        print("Model saved!")

        env_save_path = f"{search_path}/{self.version+1}/normalized_env"
        print(f"Saving environment: {env_save_path}")
        self.environment.save(env_save_path)
        print("Environment saved!")

        reward_save_path = f"{search_path}/{self.version+1}/rewards.txt"
        print(f"Saving reward: {reward_save_path}")
        self.save_reward_structure(reward_save_path)
        print(f"Reward saved!")

        self.version += 1

    def save_reward_structure(self, path):
        with open(path, "w") as file:
            json.dump(Reward.rewards, file)

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

        if paddle.agent_control == "human" or paddle.agent_control == "off":
            return None

        if c.settings["is_training"]:
            if team == 1 and player == 1:
                model = Model.get_latest_model(c.training["model_name"], c.training["algorithm"], c.settings["team_size"])
            else:
                model = Model.get_random_model(c.training["model_name"], c.training["algorithm"], c.settings["team_size"], paddle)
        else:
            model = Model.get_latest_model(c.training["model_name"], c.training["algorithm"], c.settings["team_size"])

        if model is None:
            print("No model fetched")

        return model

    @staticmethod
    def get_environment_norm():
        env = Model.get_environment()
        env = VecNormalize(env, norm_obs=False, clip_obs=False, norm_reward=True, clip_reward=10.)
        return env

    @staticmethod
    def get_environment():
        env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)
        return env

    @staticmethod
    def create_new_model(model_name, algorithm_name, team_size):
        print(f"Creating new model: {model_name}, {algorithm_name}, {team_size}")
        training_algorithm = Model.get_algorithm(algorithm_name)
        env = Model.get_environment_norm()
        policy_kwargs = dict(net_arch=dict(pi=[64, 128, 64], vf=[64, 128, 64]))
        sb3_model = training_algorithm("MultiInputPolicy", env, learning_rate=c.training["learning_rate"], ent_coef=c.training["ent_coef"], verbose=1, device=g.device, policy_kwargs=policy_kwargs)
        version = 0
        model = Model(sb3_model, version, algorithm_name, model_name, team_size, env)
        print("Model created")
        return model

    @staticmethod
    def get_latest_model(model_name, algorithm_name, team_size):
        print(f"Trying to get latest model: {model_name}, {algorithm_name}, {team_size}")
        search_path = Model.get_search_path(model_name, algorithm_name, team_size)
        Model.ensure_path_exists(search_path)
        models = os.listdir(search_path)
        if len(models) == 0:
            print("Couldn't find any matching models")
            return Model.create_new_model(model_name, algorithm_name, team_size)

        models = [model for model in models if model.isdigit()]
        version = max(models, key=lambda x: int(x))
        model_path = os.path.join(search_path, version)
        sb3_model, env = Model.get_model(model_path, algorithm_name)
        model = Model(sb3_model, int(version), algorithm_name, model_name, team_size, env)
        print(f"Latest model: {model_path} loaded!")
        return model

    def get_search_path_self(self):
        return Model.get_search_path(self.model_name, self.algorithm_name, self.team_size)

    @staticmethod
    def get_search_path(model_name, algorithm_name, team_size):
        search_path = f"{c.training['base_path']}/{algorithm_name}/{team_size}/{model_name}"
        return search_path

    @staticmethod
    def get_random_model(model_name, algorithm_name, team_size, paddle):
        print(f"Trying to get random model: {model_name}, {algorithm_name}, {team_size}")
        search_path = Model.get_search_path(model_name, algorithm_name, team_size)
        Model.ensure_path_exists(search_path)
        models = os.listdir(search_path)
        if len(models) == 0:
            print("Couldn't find any matching models")
            return Model.create_new_model(model_name, algorithm_name, team_size)

        models = [model for model in models if model.isdigit()]
        models = sorted(models, key=lambda x: int(x))

        if paddle.team == 1:
            variance = c.training["model_selection_variance_team"]
        else:
            variance = c.training["model_selection_variance_opponent"]

        random_index = max(0, len(models) - int(np.abs(np.random.normal(0, variance, 1)[0]) * len(models)) - 1)
        version = models[random_index]
        model_path = os.path.join(search_path, version)
        sb3_model, env = Model.get_model(model_path, algorithm_name)
        model = Model(sb3_model, int(version), algorithm_name, model_name, team_size, env)
        print(f"Random model: {model_path} loaded!")
        return model

    @staticmethod
    def get_model(path, algorithm_name):
        training_algorithm = Model.get_algorithm(algorithm_name)
        model_path = f"{path}/model"
        env_path = f"{path}/normalized_env"
        print(f"loading environment from: {env_path}")
        env = VecNormalize.load(env_path, Model.get_environment())
        custom_objects = { "learning_rate": c.training["learning_rate"], "ent_coef": c.training["ent_coef"] }
        print(f"loading model from: {model_path}")
        sb3_model = training_algorithm.load(model_path, env, custom_objects=custom_objects, device=g.device)
        return sb3_model, env

    @staticmethod
    def ensure_path_exists(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

    # def save_model_name():
    #     save_text_to_file(g.current_model_name, "model_name/name.txt")



