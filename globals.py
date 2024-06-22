import pygame.gfxdraw

# Training params
REWARD_POLICY = {   
    'time_reward': -1.0,
    'acc_reward': -1.0,
    'player_1_goal': 400,
    'player_2_goal': -400,
    'ball_proximity': 1.0,
    'ball_velocity': 2.0,
    'ball_velocity_toward_goal': 4.0,
    'normalization': 100,
}

TRAINING_PARAMS = {
    'training_steps': 50000,
    'learning_rate': 1.0e-4,
}

# Colors
BG_COLOR = (43, 50, 80)

# Display
WIDTH, HEIGHT = 800, 400
HIGH_FPS = 6000
LOW_FPS = 60

# Fonts
REWARD_COLOR = (154, 120, 134)
REWARD_FONT = None
REWARD_FONT_SIZE = 30
REWARD_POS = (10, 10)

TIME_COLOR = (154, 120, 134)
TIME_FONT = None
TIME_FONT_SIZE = 55
TIME_POS = (WIDTH/2, 30)

STEPS_LEFT_COLOR = (154, 120, 134)
STEPS_LEFT_FONT = None
STEPS_LEFT_FONT_SIZE = 30
STEPS_LEFT_POS = (WIDTH - 70, 30)

# Paddle
PADDLE_COLOR_1 = (51, 153, 255)
PADDLE_COLOR_2 = (0, 204, 102)
PADDLE_RADIUS = 20
PADDLE_FRICTION = 0.92
PADDLE_SPEED = 1.0
MAX_PADDLE_SPEED = 25

# Puck
PUCK_COLOR = (124, 254, 253)
MAX_PUCK_SPEED = 25
PUCK_START_SPEED = 4
PUCK_RADIUS = 15
PUCK_FRICTION = 0.99

# Match
TIME_LIMIT = 15 * 60

# Helper functions
def draw_circle(pos, radius, color, screen):
    pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), radius, color)    
    pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), radius, (0,0,0))

def interpolate_color(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )    