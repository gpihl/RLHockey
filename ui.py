import constants as c
import globals as g

REWARD_POS = (20, 10)
SCORE_POS = (170, 20)
TIME_POS = (c.settings['field_width']/2, 10)
STEPS_LEFT_POS = (c.settings['field_width'] - 150, 30)
TEXT_COLOR = (255, 255, 255)

REWARD_FONT_SIZE = 40
TIME_FONT_SIZE = 120
STEPS_LEFT_FONT_SIZE = 40
SCORE_FONT_SIZE = 85
MODEL_NAME_FONT_SIZE = 40
REWARD_BREAKDOWN_FONT_SIZE = 40

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
        g.framework.draw_text(text, 'steps_left', TEXT_COLOR, STEPS_LEFT_POS, alignment='center', font_size=STEPS_LEFT_FONT_SIZE)

    def draw_score(self, score, paddle1, paddle2):
        text1 = str(score[0])
        text2 = str(score[1])
        pos1 = (c.settings['field_width'] / 2 - SCORE_POS[0], SCORE_POS[1])
        pos2 = (c.settings['field_width'] / 2 + SCORE_POS[0], SCORE_POS[1])
        g.framework.draw_text(text1, 'score', TEXT_COLOR, pos1, alignment='right', font_size=SCORE_FONT_SIZE)
        g.framework.draw_text(text2, 'score', TEXT_COLOR, pos2, alignment='left', font_size=SCORE_FONT_SIZE)
        g.framework.draw_circle((pos1[0] + 40, pos1[1] + 42), 20, paddle1.color)
        g.framework.draw_circle((pos2[0] - 40, pos2[1] + 42), 20, paddle2.color)

    def draw_time_left(self, time_left):
        g.framework.draw_text(str(time_left), 'time_left', TEXT_COLOR, TIME_POS, alignment='center', font_size=TIME_FONT_SIZE)

    def draw_reward_breakdown(self, reward_breakdown_1, reward_breakdown_2):
        g.framework.draw_dict(reward_breakdown_1, 'reward_breakdown', (440, c.settings['field_height'] - 50), REWARD_BREAKDOWN_FONT_SIZE)
        g.framework.draw_dict(reward_breakdown_2, 'reward_breakdown', (c.settings['field_width'] - 100, c.settings['field_height'] - 50), REWARD_BREAKDOWN_FONT_SIZE)
