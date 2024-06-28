import pygame
import math
import numpy as np
import time
import globals as g

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
        self.vsync = 0 if g.SETTINGS['is_training'] else 1
        print(g.SETTINGS['is_training'])
        self.last_ui_input = 0
        self.screen = pygame.display.set_mode((g.RESOLUTION_W, g.RESOLUTION_H), flags = pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync = self.vsync)
        # self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()

        self.fonts = {
            'reward': pygame.font.SysFont(g.REWARD_FONT, self.world_to_screen_length(g.REWARD_FONT_SIZE)),
            'time_left': pygame.font.SysFont(g.TIME_FONT, self.world_to_screen_length(g.TIME_FONT_SIZE)),
            'steps_left': pygame.font.SysFont(g.STEPS_LEFT_FONT, self.world_to_screen_length(g.STEPS_LEFT_FONT_SIZE)),
            'score': pygame.font.SysFont(g.SCORE_FONT, self.world_to_screen_length(g.SCORE_FONT_SIZE)),
            'puck': pygame.font.SysFont(g.PUCK_TEXT_FONT, self.world_to_screen_length(g.PUCK_TEXT_FONT_SIZE)),
        }

        self.fps = g.HIGH_FPS if g.SETTINGS['is_training'] else g.LOW_FPS

    def calculate_scaling_and_shift(self):
        x_stretch = g.RESOLUTION_W / g.WIDTH
        y_stretch = g.RESOLUTION_H / g.HEIGHT
        scaling = min(x_stretch, y_stretch)

        if x_stretch > y_stretch:
            shift = ((g.RESOLUTION_W - g.WIDTH * scaling) / 2, 0)
        elif y_stretch > x_stretch:
            shift = (0, (g.RESOLUTION_H - g.HEIGHT * scaling) / 2)
        else:
            shift = (0, 0)

        return (scaling, shift)

    def handle_keyboard_input(self):
        events = pygame.event.get()

        if np.abs(g.current_time - self.last_ui_input) < 0.5:
            return
        
        self.last_ui_input = g.current_time

        keys = g.get_keys()
        if keys[pygame.K_e]:
            self.fps = g.HIGH_FPS if self.fps == g.LOW_FPS else g.LOW_FPS
            print(f"Switching to {self.fps} FPS")
        elif keys[pygame.K_r]:
            g.TRAINING_PARAMS['no_render'] = not g.TRAINING_PARAMS['no_render']
            print(f"Setting rendering to {g.TRAINING_PARAMS['no_render']}")
        elif keys[pygame.K_m]:
            g.TRAINING_PARAMS['no_sound'] = not g.TRAINING_PARAMS['no_sound']
            print(f"Setting sound to {not g.TRAINING_PARAMS['no_sound']}")
        elif keys[pygame.K_t]:
            g.SETTINGS['player_2_human'] = not g.SETTINGS['player_2_human']
            print(f"Setting player 2 human to {not g.SETTINGS['player_2_human']}")
        elif keys[pygame.K_y]:
            self.toggle_fullscreen()
            print(f"Toggling to fullscreen")

    def toggle_fullscreen(self):
        pygame.display.toggle_fullscreen()
        # current_flags = self.screen.get_flags()
        # if current_flags & pygame.FULLSCREEN:
        #     # Switch to windowed mode
        #     self.screen = pygame.display.set_mode((g.RESOLUTION_W, g.RESOLUTION_H), 
        #                                         flags=pygame.SCALED | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 
        #                                         vsync=self.vsync)
        # else:
        #     # Switch to fullscreen mode
        #     self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def handle_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYDEVICEREMOVED:
                print(f"Joystick {event.instance_id} disconnected")
                pygame.joystick.quit()
            elif event.type == pygame.JOYDEVICEADDED:
                print(f"Joystick connected")
                g.init_controls()

        return running

    def tick(self):
        g.current_time = time.time()
        self.clock.tick(self.fps)

    def render(self):
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

    def draw_circle(self, pos, radius, color):
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color)

    def world_to_screen_coord(self, coord):
        x = int(coord[0] * self.scaling_factor) + self.shift[0]
        y = int(coord[1] * self.scaling_factor) + self.shift[1]
        return (x, y)
    
    def world_to_screen_length(self, length):
        return int(length * self.scaling_factor)

    def draw_text(self, text, font_name, color, position, centered=False, rotation=0.0):
        position = self.world_to_screen_coord(position)
        surface = self.fonts[font_name].render(text, True, color)
        rect = surface.get_rect()

        if rotation != 0.0:
            surface = pygame.transform.rotate(surface, -rotation)
            rect = surface.get_rect()

        if centered:
            rect.center = position
            self.screen.blit(surface, rect)
        else:
            self.screen.blit(surface, position)

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

    # def draw_rotated_line(self, pos, length, angle, color, centered=True, width=1):
    #     pos = self.world_to_screen_coord(pos)
    #     length = self.world_to_screen_length(length)
    #     width = self.world_to_screen_length(width)
    #     line_surface = pygame.Surface((length, width), pygame.SRCALPHA)
    #     line_surface.fill(color)
        
    #     rotated_surface = pygame.transform.rotate(line_surface, angle)
        
    #     rect = rotated_surface.get_rect()
    #     if centered:
    #         rect.center = pos
    #     else:
    #         # Calculate the position for non-centered line
    #         angle_rad = math.radians(angle)
    #         offset_x = rect.height * 0.5 * math.cos(angle_rad)
    #         offset_y = rect.height * 0.5 * math.sin(angle_rad)
    #         rect.centerx = pos[0] + offset_x
    #         rect.centery = pos[1] - offset_y
        
    #     self.screen.blit(rotated_surface, rect)

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
        
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), int(width))        

    def close(self):
        pygame.joystick.quit()        
        pygame.quit()

