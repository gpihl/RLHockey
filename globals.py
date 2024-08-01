import torch
from sound_handler import SoundHandler
from framework import Framework
from ui import UI
from controls import Controls
from field import Field
from game import Game
from clock import Clock
import time

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
        current_time, game, clock, seed


    seed = time.time()
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

