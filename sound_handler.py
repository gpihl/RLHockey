import pygame
import os
from scipy import signal
import numpy as np
import globals as g
import threading
import time
from queue import Queue

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

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.set_num_channels(24)
        self.sounds = {}
        self.channels = [pygame.mixer.Channel(i) for i in range(24)]
        self.sound_queue = Queue()
        self.thread = threading.Thread(target=self._sound_thread, daemon=True)
        self.last_played = 0
        self.thread.start()        
        self.load_sounds()
        self.create_octave_up_sounds()
        self.scales = {}
        self.create_scales()

        print('sound handler init done')


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
            self.scales[k] += [n+12 for n in self.scales[k]] + [n+24 for n in self.scales[k]]

    def map_to_scale(self, number, scale):
        return self.scales[scale][(number - 1) % len(scale)]

    def load_sounds(self):
        filenames = [i for i in range(1, 13)] + ['table_hit']
        
        for i in range(len(filenames)):
            path = f"sounds/{filenames[i]}.wav"
            if os.path.exists(path):
                self.sounds[filenames[i]] = pygame.mixer.Sound(path)
                self.sounds[filenames[i]].set_volume(0.1)
            else:
                print(f"Warning: Sound file {path} not found.")

        print(self.sounds)

        

    def velocity_to_sound_index(self, velocity, scale='maj9'):
        velocity = max(0, min(velocity, g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED))
        index = int((velocity * 2 / (g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED)) * 23) + 1
        scale_index = self.map_to_scale(index, scale)
        return scale_index
    

    def create_octave_up_sounds(self):
        for octave in [2, 3]:
            for i in range(1, 13):
                if i in self.sounds:
                    original_sound = self.sounds[i]
                    transposed_sound = self.pitch_shift(original_sound, octave)
                    self.sounds[i + 12] = transposed_sound
                else:
                    print(f"Warning: Original sound {i} not found, skipping transposition")

    def pitch_shift(self, sound, pitch_shift):
        # Convert pygame sound to numpy array
        array = pygame.sndarray.array(sound)
        
        # Resample the audio
        resampled = signal.resample(array, int(len(array) / pitch_shift))
        
        # Convert back to pygame sound
        return pygame.sndarray.make_sound(resampled.astype(np.int16))


    def play_sound(self, velocity, x_coord, sound_name):
        if g.TRAINING_PARAMS['no_sound']:
            return
        
        if (time.time() - self.last_played > 0.1):
            self.sound_queue.put((velocity, x_coord, sound_name))

    def play_channel(self, channel, sound, vol, x_coord):
        channel.set_volume(vol * (g.WIDTH - x_coord) / g.WIDTH, vol * (x_coord) / g.WIDTH)
        channel.play(sound)

    def _sound_thread(self):
        while True:
            velocity, x_coord, sound_name = self.sound_queue.get()
            if sound_name == 'paddle':
                sound_name = self.velocity_to_sound_index(velocity)

            if sound_name in self.sounds:
                channel = self._get_channel()

                pitch_shift = 0.8
                sound = self.sounds[sound_name]
                volume = velocity / ((g.MAX_PUCK_SPEED + g.MAX_PADDLE_SPEED) if sound_name == 'paddle' else (g.MAX_PUCK_SPEED))
                if sound_name == 'table_hit':
                    pitch_shift += (velocity / g.MAX_PUCK_SPEED) * 0.5

                sound = self.pitch_shift(sound, pitch_shift)
                if channel:
                    self.play_channel(channel, sound, volume, x_coord)
                else:
                    print("Warning: No free channel available, stopping all sounds")
                    self.stop_all_sounds()
                    channel = self._get_channel()
                    if channel:
                        self.play_channel(channel, sound, volume, x_coord)
                self.last_played = time.time()
            else:
                print(f"Warning: No sound file for name {sound_name}")
            self.sound_queue.task_done()

    def _get_channel(self):
        # Try to find a free channel
        channel = next((c for c in self.channels if not c.get_busy()), None)
        if channel:
            return channel
        
        # If no free channel, find the oldest playing sound and use that channel
        oldest_channel = min(self.channels, key=lambda c: c.get_endevent() if c.get_endevent() else 0)
        oldest_channel.stop()
        return oldest_channel

    def stop_all_sounds(self):
        for channel in self.channels:
            channel.stop()