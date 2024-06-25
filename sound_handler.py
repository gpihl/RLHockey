import pygame
import os
import numpy as np
from scipy import signal
import globals as g
import threading
import time
from collections import deque

class SoundHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SoundHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        pygame.mixer.pre_init(44100, -16, 2, 256)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(36)  # Increase number of channels

        self.sounds = {}
        self.active_sounds = deque(maxlen=36)
        self.sound_lock = threading.Lock()
        self.load_sounds()
        self.create_octave_up_sounds()
        self.scales = {}
        self.create_scales()
        self.current_scale = 'maj9'
        self.scale_change_period = 5

        print('Sound handler init done')

    def update(self):
        time_segment = int(time.time() / self.scale_change_period)
        scale_name = list(self.scales.keys())[time_segment % len(self.scales.keys())]
        self.current_scale = scale_name

    def create_scales(self):
        self.scales = {
            'm_pentatonic': [1, 4, 6, 8, 11],
            'M_pentatonic': [1, 5, 6, 8, 10],
            'minor': [1,3,4,6,8,9,11],
            'dorian': [1,3,4,6,8,10,11],
            'mixolydian': [1,3,5,6,8,10,11],
            'lydian': [1,3,5,7,8,10,12],
            'major': [1,3,5,6,8,10,12],
            'maj9': [1,3,5,8,12],
            'phrygian': [1,2,4,6,8,9,11],            
            'chromatic': [1,2,3,4,5,6,7,8,9,10,11],
        }

        for k, _ in self.scales.items():
            self.scales[k] += [n+12 for n in self.scales[k]]

    def map_to_scale(self, number, scale):
        return self.scales[scale][(number - 1) % len(self.scales[scale])]

    def load_sounds(self):
        filenames = [i for i in range(1, 13)] + ['table_hit']
        
        for i in range(len(filenames)):
            path = f"sounds/{filenames[i]}.wav"
            if os.path.exists(path):
                self.sounds[filenames[i]] = pygame.mixer.Sound(path)
            else:
                print(f"Warning: Sound file {path} not found.")

    def velocity_to_sound_index(self, velocity):
        velocity = max(0, min(velocity, g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED))
        index = int((1 - (velocity / (g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED))) * len(self.scales[self.current_scale])) + 1
        scale_index = self.map_to_scale(index, self.current_scale)
        return scale_index

    def create_octave_up_sounds(self):
        for octave in [2]:
            for i in range(1, 13):
                if i in self.sounds:
                    original_sound = self.sounds[i]
                    transposed_sound = self.pitch_shift(original_sound, octave)
                    self.sounds[i + 12 * (octave - 1)] = transposed_sound
                else:
                    print(f"Warning: Original sound {i} not found, skipping transposition")

    def pitch_shift(self, sound, pitch_shift):
        array = pygame.sndarray.array(sound)
        resampled = signal.resample(array, int(len(array) / pitch_shift))
        return pygame.sndarray.make_sound(resampled.astype(np.int16))
    
    def play_goal_sound(self, x):
        for sound_name in self.scales[self.current_scale]:
            self.play_sound(g.MAX_PUCK_SPEED / 4, x, sound_name)

    def play_sound(self, velocity, x_coord, sound_name):
        if g.TRAINING_PARAMS['no_sound']:
            return
        
        if sound_name == 'paddle':
            sound_name = self.velocity_to_sound_index(velocity)

        if sound_name in self.sounds:
            volume = velocity / ((g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED) if sound_name == 'paddle' else (g.MAX_PUCK_SPEED))
            volume = min(0.8, volume)
            left_vol = volume * (g.WIDTH - x_coord) / g.WIDTH
            right_vol = volume * x_coord / g.WIDTH

            sound = self.sounds[sound_name]
            if sound_name == 'table_hit':
                sound = self.pitch_shift(sound, 1 + volume)

            channel = pygame.mixer.find_channel()
            if channel:
                channel.set_volume(left_vol, right_vol)
                channel.play(sound)
            else:
                pass
                # print("No free channel available to play sound")
        else:
            print(f"Warning: No sound file for name {sound_name}")

    def stop_all_sounds(self):
        pygame.mixer.stop()

    def __del__(self):
        pygame.mixer.quit()
        print("SoundHandler destroyed")