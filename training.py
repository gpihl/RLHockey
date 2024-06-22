import os
import environment
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import globals as g

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

base_path = "./models"
model_prefix = "newester_test"
latest_model_path = get_latest_model_path(base_path, model_prefix)
next_model_path = get_next_model_path(base_path, model_prefix)

env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)

if latest_model_path:
    print(f"Loading model {latest_model_path}")
    model1 = SAC.load(latest_model_path, env=env)
    model2 = SAC.load(latest_model_path, env=env)
else:
    print(f"Creating new model {latest_model_path}")
    model1 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1)
    model2 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1)

env.envs[0].get_wrapper_attr('game').player_2_model = model2

model1.learn(total_timesteps=g.TRAINING_PARAMS['training_steps'])

print("Finished training.")

print(f"Saving model {next_model_path}")
model1.save(next_model_path)
save_dict_to_file(g.REWARD_POLICY, next_model_path + '.txt')
save_dict_to_file(g.TRAINING_PARAMS, next_model_path + '.txt')
