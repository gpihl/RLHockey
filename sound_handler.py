import pygame
import os
import numpy as np
from scipy import signal
import globals as g
import constants as c
import helpers as h
import threading
import random
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
        self.reserved_channels = 8
        self.total_channels = 64
        pygame.mixer.set_num_channels(self.total_channels)
        self.scales = {}
        self.chords = {}
        self.colors = {}
        self.scale_order = []
        self.create_scales()

        self.sounds = {}
        self.active_sounds = deque(maxlen=64)
        self.sound_lock = threading.Lock()

        self.active_channels = {
            'ambience': None,
        }

        self.active_volumes = {
            'ambience': None,
        }

        self.load_sounds()
        self.create_octave_up_sounds()

        self.pitch_buckets_per_octave = 50
        self.scale_change_period = 20
        self.bg_music_period = 15
        self.bg_music_last_played = -100
        self.pitch_octaves = 3

        self.current_scale = ''

        self.last_played = dict.fromkeys(self.sounds, 0)
        self.play_ambience()

        print('Sound handler init done')

    def pitch_hash(self, shift_amount):
        idx = int(self.pitch_buckets_per_octave * shift_amount)
        return idx

    def reset(self):
        pass

    def update(self):
        id = self.current_scale_id()
        scale_name = self.scale_id_to_name(id)
        self.current_scale = scale_name
        if g.current_time - self.bg_music_last_played > self.bg_music_period:
            self.bg_music_last_played = g.current_time
            self.play_bg_music()

    def scale_id_to_name(self, id):
        scale_name = self.scale_order[id]
        return scale_name

    def current_scale_id(self):
        current_scale_id = int(g.current_time / self.scale_change_period) % len(self.scales.keys())
        return current_scale_id

    def current_color(self):
        id = self.current_color_id()
        color = self.colors[self.scale_id_to_name(id)]
        return color

    def current_color_id(self):
        id = self.current_scale_id()
        id = round(self.current_scale_id() + (g.current_time % self.scale_change_period) / self.scale_change_period) - 1
        return id % len(self.colors.keys())

    def next_color(self):
        id = (self.current_color_id() + 1) % len(self.colors.keys())
        color = self.colors[self.scale_id_to_name(id)]
        return color

    def theme_alpha(self):
        alpha = (g.current_time % self.scale_change_period) / self.scale_change_period
        alpha = (alpha + 0.5) % 1
        theme_alpha = h.smoothstep(alpha)
        return theme_alpha

    def target_color(self):
        target_color = h.interpolate_color(self.current_color(), self.next_color(), self.theme_alpha())
        return target_color

    def create_scales(self):
        self.scale_order = [
            'maj9',
            'major',
            'lydian',
            'mixolydian',
            'dorian',
            'phrygian',
            'chromatic',
            'minor',
            'minor-pentatonic',
            'major-pentatonic',
            ]

        self.scales = {
            'minor-pentatonic': [1, 4, 6, 8, 11],
            'major-pentatonic': [1, 5, 6, 8, 10],
            'minor': [1,3,4,6,8,9,11],
            'dorian': [1,3,4,6,8,10,11],
            'mixolydian': [1,3,5,6,8,10,11],
            'lydian': [1,3,5,7,8,10,12],
            'major': [1,3,5,6,8,10,12],
            'maj9': [1,3,5,8,12],
            'phrygian': [1,2,4,6,8,9,11],
            'chromatic': [1,2,3,4,5,6,7,8,9,10,11],
        }

        self.chords = {
            'minor-pentatonic': [1, 4, 6, 8, 11, 16],
            'major-pentatonic': [1, 6, 8, 10, 17],
            'minor': [1,8,14,23],
            'dorian': [1,4,8,10,11],
            'mixolydian': [1,8,17,23],
            'lydian': [1,7,8,17,24],
            'major': [1,8,17,24],
            'maj9': [1,3,5,8,12],
            'phrygian': [1,8,16,23],
            'chromatic': [1,7,16,22],
        }

        self.colors = {
            'minor-pentatonic': (59, 120, 63),
            'major-pentatonic': (171, 171, 51),
            'minor': (80, 50, 168),
            'dorian': (12, 35, 168),
            'mixolydian': (204, 121, 27),
            'lydian': (27, 198, 204),
            'major': (97, 242, 109),
            'maj9': (230, 90, 90),
            'phrygian': (50, 47, 138),
            'chromatic': (7, 6, 43),
        }

        for k, _ in self.colors.items():
            self.colors[k] = h.clamp_s(self.colors[k], 0.1, 0.5)
            self.colors[k] = h.clamp_l(self.colors[k], 0.2, 0.5)

        for k, _ in self.scales.items():
            self.scales[k] += [n+12 for n in self.scales[k]]

    def play_bg_music(self):
        self.bg_music_last_played = g.current_time
        scale = self.current_scale
        clip_names = self.get_scale_clip_names(scale)

        if len(clip_names) == 0:
            print(f"Tried to play a {scale} clip but found none.")
            return

        clip_name = random.choice(clip_names)
        x_coord = int(np.clip(np.random.normal(c.settings['field_width'] / 2, c.settings['field_width'] / 4, 1), 0, c.settings['field_width']))
        self.play_sound(25, x_coord, clip_name, priority=True)

    def get_scale_clip_names(self, scale):
        clips_names = list(filter(lambda name: name.rsplit('-', 1)[0] == scale, self.sounds))
        return clips_names

    def map_to_scale(self, number, scale):
        return self.scales[scale][(number - 1) % len(self.scales[scale])]

    def get_wav_files(self, folder_path):
        return [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.wav')]

    def play_ambience(self):
        self.play_sound(40, c.settings['field_width'] / 2, 'ambience', -1, priority=True)
        self.active_channels['ambience'] = None
        self.active_volumes['ambience'] = None

    def load_sounds(self):
        filenames = self.get_wav_files("sounds/")

        for filename in filenames:
            path = f"sounds/{filename}.wav"
            if os.path.exists(path):
                self.sounds[filename] = pygame.mixer.Sound(path)
            else:
                print(f"Warning: Sound file {path} not found.")

        for name in ['overload', 'charge']:
            for i in range(16):
                self.sounds[f"{name}{i}"] = self.sounds[name]
                self.active_channels[f"{name}{i}"] = None
                self.active_volumes[f"{name}{i}"] = None

    def velocity_to_sound_index(self, velocity):
        alpha = 1 - min(1, max(0, velocity / (c.gameplay['max_puck_speed'] + c.gameplay['max_paddle_speed'])))
        alpha = alpha ** 2
        index = int(alpha * len(self.scales[self.current_scale])) + 1
        scale_index = self.map_to_scale(index, self.current_scale)
        return str(scale_index)

    def create_octave_up_sounds(self):
        for octave in [2]:
            for i in range(1, 13):
                if str(i) in self.sounds:
                    original_sound = self.sounds[str(i)]
                    transposed_sound = self.pitch_shift(original_sound, octave)
                    self.sounds[str(i + 12 * (octave - 1))] = transposed_sound
                else:
                    print(f"Warning: Original sound {i} not found, skipping transposition")

    def pitch_shift(self, sound, shift_amount):
        array = pygame.sndarray.array(sound)
        resampled = signal.resample(array, int(len(array) / shift_amount))
        pitched_sound = pygame.sndarray.make_sound(resampled.astype(np.int16))

        return pitched_sound

    def pitch_shift_hashed(self, sound, sound_name, shift_amount):
        hashed_name = f"{sound_name}-{self.pitch_hash(shift_amount)}"
        if hashed_name in self.sounds:
            pitched_sound = self.sounds[hashed_name]
        else:
            pitched_sound = self.pitch_shift(sound, shift_amount)
            self.sounds[hashed_name] = pitched_sound

        return pitched_sound

    def set_pan(self, channel, volume, x_coord):
        left_vol = volume * (c.settings['field_width'] - x_coord) / c.settings['field_width']
        right_vol = volume * x_coord / c.settings['field_width']
        channel.set_volume(left_vol, right_vol)

    def change_pan(self, channel, x_coord, volume=None):
        if volume is not None:
            self.set_pan(channel, volume, x_coord)
        else:
            volume = channel.get_volume()

        self.set_pan(channel, volume, x_coord)

    def update_paddle_sound(self, paddle):
        sound_name = f"overload{(paddle.team - 1) * 2 + paddle.player}"
        channel = self.active_channels[sound_name]
        volume = self.active_volumes[sound_name]
        if paddle.is_overloaded():
            if channel is not None:
                self.change_pan(channel, paddle.pos[0], volume)
            else:
                self.play_sound(15, paddle.pos[0], sound_name, active=True)
        elif channel is not None:
            self.stop_sound(sound_name)
            self.active_channels[sound_name] = None

        sound_name = f"charge{(paddle.team - 1) * 2 + paddle.player}"
        channel = self.active_channels[sound_name]
        volume = self.active_volumes[sound_name]
        if paddle.charging_dash_initial and paddle.charging_alpha() > 0.2:
            if channel is not None:
                self.change_pan(channel, paddle.pos[0], volume)
            else:
                self.play_sound(10, paddle.pos[0], sound_name, active=True)
        elif channel is not None:
            self.stop_sound(sound_name)
            self.active_channels[sound_name] = None

    def play_goal_sound(self, x):
        selected_notes = random.choices(self.scales[self.current_scale][0:8], k=4)
        for sound_name in self.active_channels.keys():
            self.stop_sound(sound_name)

        for sound_name in selected_notes:
            self.play_sound(c.gameplay['max_puck_speed'] / 4, x, str(sound_name), priority=True)

    def stop_sound(self, sound_name):
        channel = self.active_channels[sound_name]
        if channel is not None:
            self.active_channels[sound_name] = None
            channel.fadeout(300)

    def play_sound(self, velocity, x_coord, sound_name, loops=0, pitch_shift=False, active=False, priority=False):
        # print(f"playing: {velocity}, {x_coord}, {sound_name}")
        if c.settings['is_training'] and c.settings['no_sound']:
            return

        if sound_name == 'paddle':
            sound_name = self.velocity_to_sound_index(velocity)

        if sound_name in self.sounds:
            if g.current_time - self.last_played[sound_name] < 0.05 and priority == False:
                return

            volume = velocity / ((c.gameplay['max_puck_speed'] + c.gameplay['max_paddle_speed']) if sound_name == 'paddle' else (c.gameplay['max_puck_speed']))
            volume = min(0.8, volume)

            if volume < 0.01:
                return

            sound = self.sounds[sound_name]
            if pitch_shift:
                sound = self.pitch_shift_hashed(sound, sound_name, 1 + volume)

            channel = self.find_channel(priority=priority)

            if channel:
                self.set_pan(channel, volume, x_coord)
                if active:
                    self.active_channels[sound_name] = channel
                    self.active_volumes[sound_name] = volume
                channel.set_volume(volume)
                channel.play(sound, loops=loops)


                self.last_played[sound_name] = g.current_time
        else:
            print(f"Warning: No sound file for name {sound_name}")

    def find_channel(self, force=False, priority=False):
        start = 0 if priority else self.reserved_channels
        end = self.reserved_channels if priority else self.total_channels

        for i in range(start, end):
            channel = pygame.mixer.Channel(i)
            if not channel.get_busy():
                return channel

        if force:
            return pygame.mixer.Channel(start)

        return None

    def stop_all_sounds(self):
        pygame.mixer.stop()

    def __del__(self):
        pygame.mixer.quit()
        print("SoundHandler destroyed")