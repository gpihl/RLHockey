import environment
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
import globals as g
import helpers as h
import constants as c

def get_algorithm(algorithm_name):
    match algorithm_name:
        case 'PPO':
            return PPO
        case 'SAC':
            return SAC
        case 'TD3':
            return TD3
        case _:
            return None

def simple_clip_range(progress_remaining):
    return 0.2 * (1 - progress_remaining)

def main():
    c.settings['is_training'] = True
    g.initialize()
    for i in range(c.training['training_iterations']):
        latest_model_path, training_algorithm = h.get_latest_model_path_with_algorithm(c.training['base_path'])
        player_2_model_path, opponent_algorithm = h.get_random_model_with_algorithm()

        next_model_path = h.get_next_model_path(c.training['base_path'], training_algorithm)
        env = make_vec_env(lambda: environment.AirHockeyEnv(), n_envs=1)

        training_algorithm = get_algorithm(training_algorithm)
        opponent_algorithm = get_algorithm(opponent_algorithm)

        print(f"Training algorithm: {training_algorithm}")
        print(f"Opponent algorithm: {opponent_algorithm}")

        model2 = None
        if latest_model_path:
            print(f"Loading model for player 1: {latest_model_path}")
            model1 = training_algorithm.load(latest_model_path, env=env, device=g.device)

            if c.training['player_2_active']:
                print(f"Loading model for player 2: {player_2_model_path}")
                model2 = opponent_algorithm.load(player_2_model_path, env=env, device='cpu')
        else:
            print(f"Creating new model {latest_model_path}")
            model1 = training_algorithm("MultiInputPolicy", env, learning_rate=c.training['learning_rate'], verbose=1, device=g.device)
            if c.training['player_2_active']:
                model2 = training_algorithm("MultiInputPolicy", env, learning_rate=c.training['learning_rate'], verbose=1, device='cpu')

        game = env.envs[0].get_wrapper_attr('game')
        game.player_2_model = model2

        print("Starting training round.")
        model1.learn(total_timesteps=c.training['training_steps'])
        print("Finished training round.")

        print(f"Saving model: {next_model_path}")
        model1.clip_range = simple_clip_range
        model1.save(next_model_path)

        h.save_text_to_file(str(c.rewards), next_model_path + '.txt')
        h.save_text_to_file(str(c.training), next_model_path + '.txt')
        h.save_text_to_file(f"Total reward: {game.total_reward}", next_model_path + '.txt')

if __name__ == '__main__':
    main()