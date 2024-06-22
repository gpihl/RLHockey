import environment
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import globals as g

for i in range(g.TRAINING_PARAMS['training_iterations']):
    latest_model_path = g.get_latest_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])
    next_model_path = g.get_next_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])

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
    g.save_dict_to_file(g.REWARD_POLICY, next_model_path + '.txt')
    g.save_dict_to_file(g.TRAINING_PARAMS, next_model_path + '.txt')
