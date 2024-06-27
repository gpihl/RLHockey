import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import globals as g
import ctypes

class Framework:
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
        self.screen = pygame.display.set_mode((g.RESOLUTION_W, g.RESOLUTION_H), 
                                              OPENGL | DOUBLEBUF | RESIZABLE)
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, self.vsync)
        self.clock = pygame.time.Clock()
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()

        self.setup_opengl()
        self.setup_shaders()
        self.setup_fonts()

        self.fps = g.HIGH_FPS if g.SETTINGS['is_training'] else g.LOW_FPS

        # Create a Pygame surface to draw on
        self.surface = pygame.Surface((g.WIDTH, g.HEIGHT))

    def setup_opengl(self):
        glViewport(0, 0, g.RESOLUTION_W, g.RESOLUTION_H)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, g.RESOLUTION_W, g.RESOLUTION_H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)

        # Create persistent texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g.WIDTH, g.HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def setup_shaders(self):
        vertex_shader = """
        #version 120
        attribute vec2 position;
        attribute vec2 in_texcoord;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
        """

        fragment_shader = """
        #version 120
        varying vec2 v_texcoord;
        uniform sampler2D tex;
        void main() {
            gl_FragColor = texture2D(tex, v_texcoord);
        }
        """

        self.shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

        # Create a fullscreen quad
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.quad_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_buffer)
        quad_data = np.array([
            -1, -1, 0, 1,
             1, -1, 1, 1,
             1,  1, 1, 0,
            -1,  1, 0, 0
        ], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL_STATIC_DRAW)

        position_loc = glGetAttribLocation(self.shader, "position")
        glEnableVertexAttribArray(position_loc)
        glVertexAttribPointer(position_loc, 2, GL_FLOAT, False, 16, ctypes.c_void_p(0))

        texcoord_loc = glGetAttribLocation(self.shader, "in_texcoord")
        glEnableVertexAttribArray(texcoord_loc)
        glVertexAttribPointer(texcoord_loc, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))

        glBindVertexArray(0)

    def setup_fonts(self):
        self.fonts = {
            'reward': pygame.font.Font(None, self.world_to_screen_length(g.REWARD_FONT_SIZE)),
            'time_left': pygame.font.Font(None, self.world_to_screen_length(g.TIME_FONT_SIZE)),
            'steps_left': pygame.font.Font(None, self.world_to_screen_length(g.STEPS_LEFT_FONT_SIZE)),
            'score': pygame.font.Font(None, self.world_to_screen_length(g.SCORE_FONT_SIZE)),
            'puck': pygame.font.Font(None, self.world_to_screen_length(g.PUCK_TEXT_FONT_SIZE)),
        }


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
            g.SETTINGS['player_2_human'] = not g.TRAINING_PARAMS['player_2_human']
            print(f"Setting player 2 human to {not g.TRAINING_PARAMS['player_2_human']}")
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
        # Lock the surface for faster pixel access
        self.surface.lock()
        
        # Convert Pygame surface to OpenGL texture
        surface_data = pygame.image.tostring(self.surface, "RGBA", True)
        
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.WIDTH, g.HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, surface_data)

        # Render fullscreen quad with shader
        glUseProgram(self.shader)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_QUADS, 0, 4)
        glBindVertexArray(0)

        pygame.display.flip()

        # Unlock the surface
        self.surface.unlock()

        # Clear the surface for the next frame
        self.surface.fill((0, 0, 0))

    def fill_screen(self, color, dimensions):
        self.surface.fill(color)

    def fill_screen_semiopaque_black(self, opacity=10):
        overlay = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, opacity))
        self.surface.blit(overlay, (0, 0))

    def draw_rectangle(self, color, pos, size):
        pygame.draw.rect(self.surface, color, (*pos, *size))

    def draw_circle(self, pos, radius, color):
        pygame.draw.circle(self.surface, color, (int(pos[0]), int(pos[1])), int(radius))

    def world_to_screen_coord(self, coord):
        x = int(coord[0] * self.scaling_factor) + self.shift[0]
        y = int(coord[1] * self.scaling_factor) + self.shift[1]
        return (x, y)
    
    def world_to_screen_length(self, length):
        return int(length * self.scaling_factor)

    def draw_text(self, text, font_name, color, position, centered=False, rotation=0.0):
        font = self.fonts[font_name]
        text_surface = font.render(text, True, color)
        
        if rotation != 0.0:
            text_surface = pygame.transform.rotate(text_surface, rotation)

        if centered:
            rect = text_surface.get_rect(center=position)
        else:
            rect = text_surface.get_rect(topleft=position)

        self.surface.blit(text_surface, rect)

    def draw_rotated_line(self, center_pos, length, angle, color, width=1):
        end_pos = (center_pos[0] + length * np.cos(np.radians(angle)),
                   center_pos[1] + length * np.sin(np.radians(angle)))
        pygame.draw.line(self.surface, color, center_pos, end_pos, width)

    def close(self):
        glDeleteTextures([self.texture])
        glDeleteBuffers(1, [self.quad_buffer])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.shader)
        pygame.joystick.quit()        
        pygame.quit()