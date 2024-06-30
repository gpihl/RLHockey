import environment
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
import globals as g
from framework import Framework
import sys
import torch
sys.setrecursionlimit(1000000)

def main():
    g.SETTINGS['is_training'] = True
    g.framework = Framework()
    for i in range(g.TRAINING_PARAMS['training_iterations']):
        latest_model_path, training_algorithm = g.get_latest_model_path_with_algorithm(g.TRAINING_PARAMS['base_path'])
        player_2_model_path, opponent_algorithm = g.get_random_model_with_algorithm()

        next_model_path = g.get_next_model_path(g.TRAINING_PARAMS['base_path'], training_algorithm)
        env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)

        batch_size = 2048

        model2 = None
        if training_algorithm == 'PPO':
            training_algorithm = PPO
        elif training_algorithm == 'SAC':
            training_algorithm = SAC
        elif training_algorithm == 'TD3':
            training_algorithm = TD3

        if opponent_algorithm == 'PPO':
            opponent_algorithm = PPO
        elif opponent_algorithm == 'SAC':
            opponent_algorithm = SAC
        elif opponent_algorithm == 'TD3':
            opponent_algorithm = TD3

        print(f"training_algorithm {training_algorithm}")
        print(f"opponent_algorithm {opponent_algorithm}")           

        if latest_model_path:
            print(f"Loading model for player 1: {latest_model_path}")
            model1 = training_algorithm.load(latest_model_path, env=env, device=g.device, batch_size=batch_size)

            if g.TRAINING_PARAMS['player_2_active']:
                print(f"Loading model for player 2: {player_2_model_path}")
                model2 = opponent_algorithm.load(player_2_model_path, env=env, device='cpu', batch_size=batch_size)
        else:
            print(f"Creating new model {latest_model_path}")
            model1 = training_algorithm("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device=g.device, batch_size=batch_size)
            if g.TRAINING_PARAMS['player_2_active']:            
                model2 = training_algorithm("MultiInputPolicy", env, learning_rate=g.TRAINING_PARAMS['learning_rate'], verbose=1, device='cpu', batch_size=batch_size)

        game = env.envs[0].get_wrapper_attr('game')
        game.player_2_model = model2

        model1.learn(total_timesteps=g.TRAINING_PARAMS['training_steps'])

        print("Finished training.")

        print(f"Saving model {next_model_path}")
        def simple_clip_range(progress_remaining):
            return 0.2 * (1 - progress_remaining)

        model1.clip_range = simple_clip_range

        model1.save(next_model_path)
        g.save_text_to_file(str(g.REWARD_POLICY), next_model_path + '.txt')
        g.save_text_to_file(str(g.TRAINING_PARAMS), next_model_path + '.txt')
        g.save_text_to_file(f"Total reward: {game.total_reward}", next_model_path + '.txt')
        
if __name__ == '__main__':
    main()