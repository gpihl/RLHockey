import torch
import time
from sound_handler import SoundHandler
from framework import Framework
from ui import UI
from controls import Controls

current_time = time.time()
current_time = None
device = None
sound_handler = None
ui = None
joystick = None

def initialize():
    global sound_handler, ui, framework, controls, device, current_time, controller_type, current_model_name, team_1_model_name, team_2_model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = time.time()
    sound_handler = SoundHandler()
    ui = UI()
    framework = Framework()
    controls = Controls()
    current_model_name = ""
    team_1_model_name = ""
    team_2_model_name = ""
