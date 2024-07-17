import random
import numpy as np
import math
import globals as g
import constants as c
import helpers as h
import argparse
from paddle import Paddle
from puck import Puck
import cProfile
from reward import Reward
from functools import reduce

class Game:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Game, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ai=False):
        if self._initialized:
            self.total_steps = 0
            return

        self._initialized = True
        self.total_steps = 0
        self.prev_t = g.current_time
        self.curr_t = g.current_time
        self.last_scorer = 2
        self.paddles_1 = []
        self.paddles_2 = []
        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.start_time = 0
        self.total_reward = 0.0
        self.score = [0, 0]
        if c.settings["is_training"]:
            self.match_steps = c.settings["round_time"] * c.settings["fps"]
        else:
            self.match_steps = 60 * c.settings["fps"]
        self.create_objects()
        # g.controls.stick = g.controls.init_controls()
        self.reset()
        self.max_paddle_speed = 0
        self.max_puck_speed = 0
        self.max_puck_spin = 0
        self.player_1_observation = {}
        self.player_2_observation = {}
        g.clock.unpause()
        print("Game initialization done")

    def create_objects(self):
        self.puck = Puck()

        for i in range(c.settings["team_size"]):
            paddle = Paddle(1, i+1)
            paddle.team_mates = c.settings["team_size"]
            self.paddles_1.append(paddle)

        for i in range(c.settings["team_size"]):
            paddle = Paddle(2, i+1)
            paddle.team_mates = c.settings["team_size"]
            self.paddles_2.append(paddle)

        for paddle in self.paddles_1:
            team_mates = list(filter(lambda x: x.player != paddle.player, self.paddles_1))
            paddle.reward = Reward(paddle, self.puck, team_mates)

        for paddle in self.paddles_2:
            team_mates = list(filter(lambda x: x.player != paddle.player, self.paddles_2))
            paddle.reward = Reward(paddle, self.puck, team_mates)

        g.paddles = self.paddles_1 + self.paddles_2

    def reset(self):
        if c.settings["is_training"]:
            c.settings["random_starting_locations"] = maybe_random_starting_locations()

        self.current_step = 0
        self.current_reward = 0.0
        self.round_reward = 0.0
        self.start_time = g.current_time

        for paddle in self.paddles_1 + self.paddles_2:
            paddle.reset()
            paddle.reward.reset()

        self.puck.reset(self.last_scorer)
        g.sound_handler.reset()
        g.field.reset()
        # g.sound_handler.play_sound(0.4, h.field_mid()[0], "light-broken")

    def non_player_1_team_1_paddles(self):
        return list(filter(lambda x: not (x.player == 1 and x.team == 1), self.paddles_1 + self.paddles_2))

    def step_training(self, player_1_model_action):
        running, _ = g.framework.handle_events()
        if not running:
            exit()

        Reward.update()

        player_1_model_action = self.paddles_1[0].model.process_action(player_1_model_action)
        player_1_action = g.controls.game_action_from_model_action(player_1_model_action)
        self.player_1_observation = self.get_observation(1, 1)
        self.player_2_observation = self.get_observation(1, 2)

        team_1_actions = [player_1_action]
        for paddle in list(filter(lambda x: not (x.player == 1 and x.team == 1), self.paddles_1)):
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_1_actions.append(action)

        team_2_actions = []
        for paddle in self.paddles_2:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_2_actions.append(action)

        scorer = self.update(team_1_actions, team_2_actions)
        reward = self.handle_rewards(team_1_actions, team_2_actions, scorer)

        return self.player_1_observation, reward, self.is_done(scorer), {}

    def step(self):
        self.curr_t = g.current_time
        delta_t = self.curr_t - self.prev_t
        c.settings["delta_t"] = min(c.settings["original_delta_t"] * delta_t, 3)
        self.prev_t = self.curr_t

        team_1_actions = []
        for paddle in self.paddles_1:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_1_actions.append(action)

        team_2_actions = []
        for paddle in self.paddles_2:
            observation = self.get_observation(paddle.team, paddle.player)
            action = paddle.get_action(observation)
            team_2_actions.append(action)

        scorer = self.update(team_1_actions, team_2_actions)

        if scorer != 0:
            self.goal_scored_sequence(scorer)

        return self.is_done(scorer)

    def update(self, team_1_actions, team_2_actions):
        self.handle_game_paused()
        self.current_step += 1
        self.total_steps += 1
        delta_t_temp = c.settings["delta_t"]
        num_updates = max(1, round(c.updates_per_delta_t * c.settings["delta_t"]))
        c.settings["delta_t"] /= num_updates
        scorer = 0

        for i in range(num_updates):
            for idx, action in enumerate(team_1_actions):
                self.paddles_1[idx].update(self.puck, action)

            for idx, action in enumerate(team_2_actions):
                self.paddles_2[idx].update(self.puck, action)

            all_paddles = self.paddles_1 + self.paddles_2
            for i in range(len(all_paddles)):
                for j in range(i+1, len(all_paddles)):
                    all_paddles[i].handle_collision(all_paddles[j])

            self.puck.update(all_paddles)

            if g.field.team_1_scored(self.puck):
                scorer = 1
                self.last_scorer = 1
                self.score[0] += 1
                break
            elif g.field.team_2_scored(self.puck):
                scorer = 2
                self.last_scorer = 2
                self.score[1] += 1
                break

        c.settings["delta_t"] = delta_t_temp

        g.sound_handler.update(scorer)
        g.field.update(self.puck)
        if not c.settings["is_training"]:
            g.framework.update_particles()
            g.framework.update_paddle_data(all_paddles)

            g.framework.update_object_data(self.paddles_1 + self.paddles_2 + [self.puck])
            paddle_lights = list(map(lambda x: x.light, all_paddles))
            g.framework.update_light_data(g.field.lights + paddle_lights + [self.puck.light])

        self.handle_rendering()
        g.framework.tick()
        # self.print_max_speeds()

        return scorer

    def handle_rewards(self, team_1_actions, team_2_actions, scorer):
        for idx, paddle in enumerate(self.paddles_1):
            paddle.current_reward = paddle.reward.calculate_total_reward(team_1_actions[idx], scorer)

        for idx, paddle in enumerate(self.paddles_2):
            paddle.current_reward = paddle.reward.calculate_total_reward(team_2_actions[idx], scorer)

        team1_reward = sum([paddle.current_reward for paddle in self.paddles_1])
        team2_reward = sum([paddle.current_reward for paddle in self.paddles_2])
        reward = team1_reward - team2_reward

        self.current_reward = reward
        self.round_reward += reward
        self.total_reward += reward
        return reward

    def handle_rendering(self):
        if not c.settings["no_render"]:
            if c.settings["is_training"] and (not g.framework.fps_locked):
                if self.current_step % 10 == 0:
                    self.render()
            else:
                self.render()
        else:
            if self.current_step % 60 == 0:
                g.framework.rendering_off_message()

    def handle_game_paused(self):
        i = 0
        while c.settings["paused"]:
            g.clock.pause()
            i += 1
            running, new_presses = g.framework.handle_events()
            if not running:
                exit()

            if i % 60 == 0:
                g.framework.paused()

            if g.framework.take_paused_step(new_presses):
                break



        g.clock.unpause()

    def print_max_speeds(self):
        self.max_puck_speed = max(self.max_puck_speed, np.linalg.norm(self.puck.vel))
        for paddle in self.paddles_1 + self.paddles_2:
            self.max_paddle_speed = max(self.max_paddle_speed, np.linalg.norm(paddle.vel))

        self.max_puck_spin = max(self.max_puck_spin, np.abs(self.puck.rot_vel))

        if self.current_step % 60 == 0:
            print(f"puck_speed: {self.max_puck_speed}")
            print(f"puck_spin: {self.max_puck_spin}")
            print(f"paddle_speed: {self.max_paddle_speed}")


    def goal_scored_sequence(self, scorer):
        if not c.settings["is_training"]:
            goal_time = g.current_time
            scorer = self.paddles_1[0] if scorer == 1 else self.paddles_2[0]
            position = h.field_mid()
            radius = c.settings["field_height"] / 5.4
            g.framework.begin_drawing()
            scorer.draw_paddle(position, radius, scorer.color, draw_indicator=False)
            g.framework.end_drawing()
            prev_time = None
            while g.current_time - goal_time < 0.9:
                if g.current_time - goal_time > 0.65:
                    if prev_time is None:
                        prev_time = g.current_time
                    delta_t = g.current_time - prev_time
                    prev_time = g.current_time
                    opacity = int(1400 * delta_t)
                    g.framework.fill_screen_semiopaque_black(opacity)

                g.framework.tick()

    def scale_absolute_position(self, pos):
        centered_origin_pos = pos - h.field_mid()
        scaled_pos = np.array([centered_origin_pos[0] / (c.settings["field_width"] / 2), centered_origin_pos[1] / (c.settings["field_height"] / 2)])
        return scaled_pos

    # Absolute positions
    def get_observation(self, team, player):
        player = player - 1
        max_puck_speed = 60
        max_paddle_speed = 110
        max_puck_spin = 130

        if team == 1:
            charging_alpha = -1 if not self.paddles_1[player].charging_dash else self.paddles_1[player].charging_alpha() * 2 - 1
            charging_alpha = np.array([charging_alpha])

            obs = {
                "self_pos":         self.scale_absolute_position(self.paddles_1[player].pos),
                "puck_pos":         self.scale_absolute_position(self.puck.pos),
                "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
                "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
                "charging_alpha":   charging_alpha,
                "goal_1_top_pos":   self.scale_absolute_position(h.goal_top_pos(1)),
                "goal_1_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(1)),
                "goal_2_top_pos":   self.scale_absolute_position(h.goal_top_pos(2)),
                "goal_2_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(2)),
            }

            other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_1))
            all_paddles_on_team = [self.paddles_1[player]] + other_paddles_on_team

            players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(other_paddles_on_team) }
            players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(self.paddles_2) }
            players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
            players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_2) }

            obs |= {
                **players_positions_team_1,
                **players_positions_team_2,
                **players_velocities_team_1,
                **players_velocities_team_2
            }

        elif team == 2:
            charging_alpha = -1 if not self.paddles_2[player].charging_dash else self.paddles_2[player].charging_alpha() * 2 - 1
            charging_alpha = np.array([charging_alpha])

            obs = {
                "self_pos":         self.scale_absolute_position(self.paddles_2[player].pos),
                "puck_pos":         self.scale_absolute_position(self.puck.pos),
                "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
                "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
                "charging_alpha":   charging_alpha,
                "goal_1_top_pos":   self.scale_absolute_position(h.goal_top_pos(2)),
                "goal_1_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(2)),
                "goal_2_top_pos":   self.scale_absolute_position(h.goal_top_pos(1)),
                "goal_2_bot_pos":   self.scale_absolute_position(h.goal_bot_pos(1)),
            }

            other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_2))
            all_paddles_on_team = [self.paddles_2[player]] + other_paddles_on_team

            players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(other_paddles_on_team) }
            players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.scale_absolute_position(paddle.pos) for i, paddle in enumerate(self.paddles_1) }
            players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
            players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_1) }

            obs |= {
                **players_positions_team_1,
                **players_positions_team_2,
                **players_velocities_team_1,
                **players_velocities_team_2
            }

            obs = { k: v if v.size == 1 else np.array([-v[0], v[1]]) for k, v in obs.items() }

        return obs

    # Relative positions
    # def get_observation(self, team, player):
    #     player = player - 1
    #     max_puck_speed = 60
    #     max_paddle_speed = 110
    #     max_puck_spin = 130

    #     if team == 1:
    #         charging_alpha = -1 if not self.paddles_1[player].charging_dash else self.paddles_1[player].charging_alpha() * 2 - 1
    #         charging_alpha = np.array([charging_alpha])

    #         obs = {
    #             "puck_pos":         self.paddles_1[player].get_relative_pos_of_puck_obs(self.puck),
    #             "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
    #             "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
    #             "charging_alpha":   charging_alpha,
    #             "goal_1_top_pos":   self.paddles_1[player].get_relative_pos_of_goal_1_top(),
    #             "goal_1_bot_pos":   self.paddles_1[player].get_relative_pos_of_goal_1_bot(),
    #             "goal_2_top_pos":   self.paddles_1[player].get_relative_pos_of_goal_2_top(),
    #             "goal_2_bot_pos":   self.paddles_1[player].get_relative_pos_of_goal_2_bot(),
    #         }

    #         other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_1))
    #         all_paddles_on_team = [self.paddles_1[player]] + other_paddles_on_team

    #         players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.paddles_1[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(other_paddles_on_team) }
    #         players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.paddles_1[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(self.paddles_2) }
    #         players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
    #         players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_2) }

    #         obs |= {
    #             **players_positions_team_1,
    #             **players_positions_team_2,
    #             **players_velocities_team_1,
    #             **players_velocities_team_2
    #         }

    #     elif team == 2:
    #         charging_alpha = -1 if not self.paddles_2[player].charging_dash else self.paddles_2[player].charging_alpha() * 2 - 1
    #         charging_alpha = np.array([charging_alpha])

    #         obs = {
    #             "puck_pos":         self.paddles_2[player].get_relative_pos_of_puck_obs(self.puck),
    #             "puck_vel":         h.scale_v(self.puck.vel, max_puck_speed, max_puck_speed),
    #             "puck_rot_vel":     h.scale(self.puck.rot_vel, max_puck_spin),
    #             "charging_alpha":   charging_alpha,
    #             "goal_1_top_pos":   self.paddles_2[player].get_relative_pos_of_goal_2_top(),
    #             "goal_1_bot_pos":   self.paddles_2[player].get_relative_pos_of_goal_2_bot(),
    #             "goal_2_top_pos":   self.paddles_2[player].get_relative_pos_of_goal_1_top(),
    #             "goal_2_bot_pos":   self.paddles_2[player].get_relative_pos_of_goal_1_bot(),
    #         }

    #         other_paddles_on_team = list(filter(lambda x: x.player != player + 1, self.paddles_2))
    #         all_paddles_on_team = [self.paddles_2[player]] + other_paddles_on_team

    #         players_positions_team_1 = { f"paddle_{1}_{i+2}_pos": self.paddles_2[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(other_paddles_on_team) }
    #         players_positions_team_2 = { f"paddle_{2}_{i+1}_pos": self.paddles_2[player].get_relative_pos_of_paddle_obs(paddle) for i, paddle in enumerate(self.paddles_1) }
    #         players_velocities_team_1 = { f"paddle_{1}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(all_paddles_on_team) }
    #         players_velocities_team_2 = { f"paddle_{2}_{i+1}_vel": h.scale_v(paddle.vel, max_paddle_speed, max_paddle_speed) for i, paddle in enumerate(self.paddles_1) }

    #         obs |= {
    #             **players_positions_team_1,
    #             **players_positions_team_2,
    #             **players_velocities_team_1,
    #             **players_velocities_team_2
    #         }

    #         obs = { k: v if v.size == 1 else np.array([-v[0], v[1]]) for k, v in obs.items() }

    #     return obs

    def render(self):
        g.framework.begin_drawing()
        g.field.draw_bottom_layer(self.puck)
        self.puck.draw()
        for paddle in self.paddles_1 + self.paddles_2:
            paddle.draw()
        g.field.draw_top_layer(self.puck)
        if not c.settings["is_training"]:
            g.framework.draw_particles()
        self.draw_ui()
        g.framework.end_drawing()

    def draw_ui(self):
        g.ui.draw_time_left(self.seconds_left())
        g.ui.draw_score(self.score, self.paddles_1[0], self.paddles_2[0])
        # g.framework.draw_fps(0,0)
        if c.settings["is_training"]:
            g.ui.draw_steps_left(str(self.total_training_steps_left()))
            team_1_breakdown = reduce(lambda x, y: h.add_dicts(x.reward.reward_breakdown, y.reward.reward_breakdown), self.paddles_1)
            team_2_breakdown = reduce(lambda x, y: h.add_dicts(x.reward.reward_breakdown, y.reward.reward_breakdown), self.paddles_2)
            g.ui.draw_reward_breakdown(team_1_breakdown, team_2_breakdown)
            # g.ui.draw_observation(self.player_1_observation)

    def total_training_steps_left(self):
        return c.training["training_steps"] - self.total_steps

    def seconds_left(self):
        if c.settings["is_training"]:
            seconds_left = math.ceil((self.match_steps - self.current_step) / c.settings["fps"])
        else:
            # seconds_left = math.ceil((self.match_steps - self.current_step) / c.settings["fps"])
            seconds_left = math.ceil((self.match_steps / c.settings["fps"]) - (g.current_time - self.start_time))
        return seconds_left

    def is_done(self, scorer):
        if c.settings["is_training"]:
            return self.current_step > self.match_steps or scorer != 0
        else:
            return self.seconds_left() < 0 or scorer != 0

    def close(self):
        g.framework.close()

def pick_training_regime():
    if c.fixed_training_regime is not None:
        return c.fixed_training_regime

    random_roll = random.random()

    for i, probability in enumerate(c.training_regime_probabilities):
        if random_roll < probability:
            selection = i
            break

    training_regime = c.training_regimes[selection]
    print(training_regime)
    return training_regime

def maybe_random_starting_locations():
    random_roll = random.random()
    return random_roll < c.settings["random_starting_locations_probability"]

def main():
    c.settings["agent_control_training"] = pick_training_regime()
    g.initialize()

    if c.settings["is_training"]:
        training_model = g.game.paddles_1[0].model
        non_training_paddles = g.game.non_player_1_team_1_paddles()
        while True:
            total_training_steps = c.training["training_steps"]
            c.settings["agent_control_training"] = pick_training_regime()
            training_model.train_model(total_training_steps)
            training_model.add_score(g.game.score)
            training_model.save_model()
            print(Reward.rewards)
            for paddle in non_training_paddles:
                paddle.load_new_model()

            g.game.total_steps = 0
    else:
        running = True
        while running:
            running, _ = g.framework.handle_events()
            done = g.game.step()

            if done:
                g.game.reset()

        g.game.close()

def validate_string(value):
    if len(value) != 4 or not all(char in 'hao' for char in value):
        raise argparse.ArgumentTypeError("Input must be a 4-character string containing only 'h', 'a', or 'o'")
    return value

def map_control(code):
    mapping = {'h': 'human', 'a': 'ai', 'o': 'off'}
    return [mapping[char] for char in code]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the game with optional profiling.")
    parser.add_argument("-p", "--profile", action="store_true", help="Run the game with profiling enabled")
    parser.add_argument("-t", "--training", action="store_true", help="Perform training of RL agents")
    parser.add_argument("-n", "--number", type=int, default=2, help="Number of players per team")
    parser.add_argument('-c', '--control', type=validate_string, required=False, help="A 4-character string containing only 'h', 'a', or 'o'")
    args = parser.parse_args()
    c.settings["team_size"] = args.number
    c.settings["is_training"] = args.training
    c.settings["no_sound"] = args.training
    if args.control is not None:
        if c.settings["is_training"]:
            c.fixed_training_regime = map_control(args.control)
        else:
            c.settings["agent_control_regular"] = map_control(args.control)

    if args.profile:
        cProfile.run("main()", "profile_output.prof")
    else:
        main()
