import constants as c
import globals as g

class UI():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UI, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True

    def draw_steps_left(self, steps_left):
        text = f"steps left: {steps_left}"
        g.framework.draw_text(text, 'steps_left', c.STEPS_LEFT_COLOR, c.STEPS_LEFT_POS, centered=True)

    def draw_score(self, score, paddle1, paddle2):
        text1 = str(score[0])
        text2 = str(score[1])
        pos1 = (c.settings['field_width'] / 2 - c.SCORE_POS[0], c.SCORE_POS[1])
        pos2 = (c.settings['field_width'] / 2 + c.SCORE_POS[0], c.SCORE_POS[1])
        g.framework.draw_text(text1, 'score', c.SCORE_COLOR, pos1, centered=True)
        g.framework.draw_text(text2, 'score', c.SCORE_COLOR, pos2, centered=True)
        g.framework.draw_circle((pos1[0] - 60, pos1[1] - 5), 20, paddle1.color)
        g.framework.draw_circle((pos2[0] + 60, pos2[1] - 5), 20, paddle2.color)

    def draw_reward(self, current_reward, round_reward):
        current_reward = f"curr reward: {current_reward:.5}"
        g.framework.draw_text(current_reward, 'reward', c.REWARD_COLOR, c.REWARD_POS)

        round_reward = f"tot reward: {round_reward:.5}"
        g.framework.draw_text(round_reward, 'reward', c.REWARD_COLOR, (c.REWARD_POS[0], c.REWARD_POS[1] + 30))

    def draw_time_left(self, time_left):
        g.framework.draw_text(str(time_left), 'time_left', c.TIME_COLOR, c.TIME_POS, centered=True)
