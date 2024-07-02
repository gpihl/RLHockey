import pygame
import pygame.gfxdraw
import math
import numpy as np
import time
import constants as c
import globals as g

REWARD_FONT_SIZE = 30
TIME_FONT_SIZE = 120
STEPS_LEFT_FONT_SIZE = 30
SCORE_FONT_SIZE = 85

class Framework():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Framework, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        pygame.init()
        self.vsync = 0 if c.settings['is_training'] else 1
        self.last_ui_input = 0
        self.current_resolution_idx = c.settings['resolution']
        self.fullscreen = False
        self.screen = None
        self.reset()
        self.clock = pygame.time.Clock()

        self.fps = 6000 if c.settings['is_training'] else c.settings['fps']

    def reset(self):
        self.trail_surface = pygame.Surface(c.resolutions[self.current_resolution_idx], pygame.SRCALPHA)
        self.screen = self.create_screen()
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()
        self.fonts = {
            'reward': pygame.font.SysFont(None, self.world_to_screen_length(REWARD_FONT_SIZE)),
            'time_left': pygame.font.SysFont(None, self.world_to_screen_length(TIME_FONT_SIZE)),
            'steps_left': pygame.font.SysFont(None, self.world_to_screen_length(STEPS_LEFT_FONT_SIZE)),
            'score': pygame.font.SysFont(None, self.world_to_screen_length(SCORE_FONT_SIZE)),
        }

    def get_resolution(self):
        return c.resolutions[self.current_resolution_idx]

    def calculate_scaling_and_shift(self):
        x_stretch = (self.get_resolution()[0]) / c.settings['field_width']
        y_stretch = (self.get_resolution()[1]) / c.settings['field_height']
        scaling = min(x_stretch, y_stretch)

        if x_stretch > y_stretch:
            shift = ((self.get_resolution()[0] - c.settings['field_width'] * scaling) / 2, 0)
        elif y_stretch > x_stretch:
            shift = (0, (self.get_resolution()[1] - c.settings['field_height'] * scaling) / 2)
        else:
            shift = (0, 0)

        return (scaling, shift)

    def handle_keyboard_input(self, events):
        if np.abs(g.current_time - self.last_ui_input) < 0.3:
            return

        if events is None:
            return

        key_pressed = False
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    self.fps = 6000 if self.fps == c.settings['fps'] else c.settings['fps']
                    key_pressed = True
                    print(f"Switching to {self.fps} FPS")
                elif event.key == pygame.K_r:
                    c.settings['no_render'] = not c.settings['no_render']
                    key_pressed = True
                    print(f"Setting rendering to {not c.settings['no_render']}")
                elif event.key == pygame.K_m:
                    c.settings['no_sound'] = not c.settings['no_sound']
                    key_pressed = True
                    print(f"Setting sound to {not c.settings['no_sound']}")
                elif event.key == pygame.K_t:
                    c.settings['player_2_human'] = not c.settings['player_2_human']
                    key_pressed = True
                    print(f"Setting player 2 human to {c.settings['player_2_human']}")
                elif event.key == pygame.K_y:
                    self.toggle_fullscreen()
                    key_pressed = True
                    print(f"Toggling fullscreen")
                elif event.key == pygame.K_u:
                    self.change_resolution()
                    key_pressed = True
                    print(f"Changing resolution")

        if key_pressed:
            self.last_ui_input = g.current_time

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        pygame.display.toggle_fullscreen()

    def change_resolution(self):
        self.current_resolution_idx = (self.current_resolution_idx + 1) % len(c.resolutions)
        pygame.display.quit()
        pygame.display.init()
        self.reset()

    def create_screen(self):
        flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        if self.fullscreen:
            flags |= pygame.FULLSCREEN

        screen = pygame.display.set_mode(c.resolutions[self.current_resolution_idx], flags = flags, vsync = self.vsync)
        return screen

    def get_events(self):
        events = None
        try:
            events = pygame.event.get()
        except:
            pass

        return events

    def handle_events(self):
        running = True
        try:
            events = self.get_events()
            self.handle_keyboard_input(events)
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print(f"Joystick {event.instance_id} disconnected")
                    pygame.joystick.quit()
                elif event.type == pygame.JOYDEVICEADDED:
                    print(f"Joystick connected")
                    g.controls.init_controls()
        except:
            pass

        return running

    def fade_surface(self, surface, amount=20):
        dark = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dark.fill((0, 0, 0, amount))
        surface.blit(dark, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    def tick(self):
        g.current_time = time.time()
        self.clock.tick(self.fps)

    def render(self):
        # self.fade_surface(self.trail_surface)
        # self.screen.blit(self.trail_surface, (0, 0))
        pygame.display.flip()

    def fill_screen(self, color, dimensions):
        self.screen.fill((0,0,0))
        self.draw_rectangle(color, (0,0), dimensions)

    def fill_screen_semiopaque_black(self, opacity=10):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, opacity))
        self.screen.blit(overlay, (0, 0))

    def draw_rectangle(self, color, pos, size):
        pos = self.world_to_screen_coord(pos)
        size = (self.world_to_screen_length(size[0]), self.world_to_screen_length(size[1]))
        pygame.draw.rect(self.screen, color, (*pos, *size))

    def draw_transparent_rectangle(self, color, pos, size, opacity):
        pos = self.world_to_screen_coord(pos)
        size = (self.world_to_screen_length(size[0]), self.world_to_screen_length(size[1]))
        rect_surface = pygame.Surface(size, pygame.SRCALPHA)
        alpha = int(opacity * 255)
        transparent_color = (*color, alpha)
        rect_surface.fill(transparent_color)
        self.screen.blit(rect_surface, pos)

    def draw_circle(self, pos, radius, color, surface=None):
        surface = surface if surface is not None else self.screen
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def draw_transparent_circle(self, pos, radius, color, opacity):
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        circle_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        alpha = int(opacity * 255)
        transparent_color = (*color, alpha)
        pygame.draw.circle(circle_surface, transparent_color, (radius, radius), radius)
        self.screen.blit(circle_surface, (pos[0] - radius, pos[1] - radius))

    def world_to_screen_coord(self, coord):
        x = round(coord[0] * self.scaling_factor + self.shift[0])
        y = round(coord[1] * self.scaling_factor + self.shift[1])
        return (x, y)

    def world_to_screen_length(self, length):
        return round(length * self.scaling_factor)

    def draw_text(self, text, font_name, color, position, alignment='left', rotation=0.0):
        position = self.world_to_screen_coord(position)
        surface = self.fonts[font_name].render(text, True, color)
        rect = surface.get_rect()

        if rotation != 0.0:
            surface = pygame.transform.rotate(surface, -rotation)
            rect = surface.get_rect()

        match alignment:
            case 'left':
                self.screen.blit(surface, position)
            case 'center':
                rect.center = (position[0], position[1] + rect.height / 2)
                self.screen.blit(surface, rect)
            case 'right':
                rect.center = (position[0] - rect.width / 2, position[1] + rect.height / 2)
                self.screen.blit(surface, rect)

    def draw_rotated_line_centered(self, pos, length, angle, color, width=1):
        pos = self.world_to_screen_coord(pos)
        length = self.world_to_screen_length(length)
        width = self.world_to_screen_length(width)
        line_surface = pygame.Surface((length, width), pygame.SRCALPHA)
        line_surface.fill(color)

        rotated_surface = pygame.transform.rotate(line_surface, angle)

        rect = rotated_surface.get_rect()
        rect.center = pos

        self.screen.blit(rotated_surface, rect)

    def draw_rotated_line(self, pos, length, angle, color, width=1):
        pos = self.world_to_screen_coord(pos)
        length = self.world_to_screen_length(length)
        width = self.world_to_screen_length(width)

        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        start_x = pos[0]
        start_y = pos[1]

        end_x = start_x + length * cos_angle
        end_y = start_y + length * sin_angle

        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), round(width))

    def close(self):
        pygame.joystick.quit()
        pygame.quit()

