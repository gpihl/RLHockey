import torch
from sound_handler import SoundHandler
from framework import Framework
from ui import UI
from controls import Controls
from field import Field

current_time = None
device = None
sound_handler = None
ui = None
joystick = None
field = None
paddles = []

def initialize():
    global field, sound_handler, ui, framework, controls, device, current_time, current_model_name, team_1_model_name, team_2_model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ui = UI()
    framework = Framework()
    sound_handler = SoundHandler()
    controls = Controls()
    field = Field()
    current_model_name = ""
    team_1_model_name = ""
    team_2_model_name = ""
