import environment
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
import globals as g
from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    for i in range(g.TRAINING_PARAMS['training_iterations']):
        latest_model_path = g.get_latest_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])
        next_model_path = g.get_next_model_path(g.TRAINING_PARAMS['base_path'], g.TRAINING_PARAMS['model_name'])

        env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)
        # env = SubprocVecEnv([lambda: environment.AirHockeyEnv(training=True) for _ in range(4)])

        if latest_model_path:
            print(f"Loading model {latest_model_path}")
            model1 = SAC.load(latest_model_path, env=env, device=g.device)
            model2 = SAC.load(latest_model_path, env=env, device=g.device)
        else:
            print(f"Creating new model {latest_model_path}")
            model1 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device=g.device)
            model2 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device=g.device)
            # model1 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device=g.device, batch_size=4096, buffer_size=1000000)
            # model2 = SAC("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device=g.device, batch_size=4096, buffer_size=1000000)

        game = env.envs[0].get_wrapper_attr('game')
        game.player_2_model = model2

        model1.learn(total_timesteps=g.TRAINING_PARAMS['training_steps'])

        print("Finished training.")

        print(f"Saving model {next_model_path}")
        model1.save(next_model_path)
        g.save_text_to_file(str(g.REWARD_POLICY), next_model_path + '.txt')
        g.save_text_to_file(str(g.TRAINING_PARAMS), next_model_path + '.txt')
        g.save_text_to_file(f"Total reward: {game.total_reward}", next_model_path + '.txt')
        
if __name__ == '__main__':
    main()