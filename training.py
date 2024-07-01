import environment
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
import globals as g
import helpers as h
import constants as c

def main():
    g.initialize()
    c.settings['is_training'] = True
    for i in range(c.training_params['training_iterations']):
        latest_model_path, training_algorithm = h.get_latest_model_path_with_algorithm(c.training_params['base_path'])
        player_2_model_path, opponent_algorithm = h.get_random_model_with_algorithm()

        next_model_path = h.get_next_model_path(c.training_params['base_path'], training_algorithm)
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

            if c.training_params['player_2_active']:
                print(f"Loading model for player 2: {player_2_model_path}")
                model2 = opponent_algorithm.load(player_2_model_path, env=env, device='cpu', batch_size=batch_size)
        else:
            print(f"Creating new model {latest_model_path}")
            model1 = training_algorithm("MultiInputPolicy", env, learning_rate=c.training_params['learning_rate'], verbose=1, device=g.device, batch_size=batch_size)
            if c.training_params['player_2_active']:            
                model2 = training_algorithm("MultiInputPolicy", env, learning_rate=c.training_params['learning_rate'], verbose=1, device='cpu', batch_size=batch_size)

        game = env.envs[0].get_wrapper_attr('game')
        game.player_2_model = model2

        model1.learn(total_timesteps=c.training_params['training_steps'])

        print("Finished training.")

        print(f"Saving model {next_model_path}")
        def simple_clip_range(progress_remaining):
            return 0.2 * (1 - progress_remaining)

        model1.clip_range = simple_clip_range

        model1.save(next_model_path)
        g.save_text_to_file(str(c.reward_policy), next_model_path + '.txt')
        g.save_text_to_file(str(c.training_params), next_model_path + '.txt')
        g.save_text_to_file(f"Total reward: {game.total_reward}", next_model_path + '.txt')
        
if __name__ == '__main__':
    main()