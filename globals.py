import torch
from sound_handler import SoundHandler
from framework import Framework
from ui import UI
from controls import Controls
from field import Field
from game import Game
from clock import Clock

current_time = 0
device = None
sound_handler = None
ui = None
joystick = None
field = None
clock = None
paddles = []

def initialize():
    global field, sound_handler, \
        ui, framework, controls, device, \
        current_time, current_model_name, \
        team_1_model_name, team_2_model_name, \
        game, clock

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Creating global singletons")
    clock = Clock()
    ui = UI()
    framework = Framework()
    sound_handler = SoundHandler()
    controls = Controls()
    field = Field()
    game = Game()
    print("Done creating global singletons")
    current_model_name = ""
    team_1_model_name = ""
    team_2_model_name = ""
