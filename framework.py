import pyray as pr
import math
import constants as c
import globals as g
import helpers as h
import noise
from particle import Particle
import numpy as np

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

        self.flags = pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_RESIZABLE
        if not c.settings['is_training']:
            self.flags |= pr.FLAG_VSYNC_HINT

        self.current_resolution_idx = c.settings['resolution']
        self.fullscreen = False
        self.fps = c.settings['fps']

        self.render_texture = None

        self.tick()
        self.reset()

        self.keys_pressed = set()
        self.fps_locked = False

        self.light_intensities = []
        self.update_light_intensities()

        if c.settings['is_training']:
            pr.set_target_fps(c.settings['fps'])
            self.fps_locked = True

        font_bold = pr.load_font_ex("fonts/Roboto-Bold.ttf", 800, None, 0)
        font_regular = pr.load_font_ex("fonts/Roboto-Regular.ttf", 800, None, 0)
        self.fxaa_shader = pr.load_shader("shaders/default.vs", "shaders/fxaa.fs")
        self.scanlines_shader = pr.load_shader("shaders/scanlines.vs", "shaders/scanlines.fs")
        self.paddle_shader = pr.load_shader("shaders/default.vs", "shaders/paddle.fs")
        self.shader = self.fxaa_shader

        self.y_extremes_loc = pr.get_shader_location(self.fxaa_shader, "yExtremes")
        resolution = pr.ffi.new('float[2]', [float(self.get_resolution()[0]), float(self.get_resolution()[1])])
        self.resolution_loc = pr.get_shader_location(self.fxaa_shader, "resolution")
        pr.set_shader_value(self.fxaa_shader, self.resolution_loc, resolution, pr.SHADER_UNIFORM_VEC2)
        resolution_loc_paddle = pr.get_shader_location(self.paddle_shader, "resolution")
        pr.set_shader_value(self.paddle_shader, resolution_loc_paddle, resolution, pr.SHADER_UNIFORM_VEC2)
        y_extremes = pr.ffi.new('float[2]', [self.world_to_screen_coord((0, 0))[1], self.world_to_screen_coord((0, c.settings['field_height']))[1]])
        pr.set_shader_value(self.fxaa_shader, self.y_extremes_loc, y_extremes, pr.SHADER_UNIFORM_VEC2)

        self.particle_shader = pr.load_shader("shaders/default.vs", "shaders/particle.fs")

        self.max_paddles = c.settings['team_size'] * 2
        self.paddle_buffer = pr.ffi.new("float[]", self.max_paddles * 9)  # 2 for position, 3 for color, 1 for radius
        self.paddle_count = pr.ffi.new("int *", 0)

        self.light_data = pr.ffi.new("float[]", 4 * 3)

        self.create_light_data()



        self.temporary_particles = []

        self.fonts = {
            'reward': font_bold,
            'time_left': font_bold,
            'steps_left': font_regular,
            'score': font_bold,
            'model_name': font_bold,
            'reward_breakdown': font_regular,
        }

        # In your Framework __init__ method:
        self.shape_texture = self.set_shapes_texture()

        pr.set_shapes_texture(self.shape_texture, pr.Rectangle(0.0, 0.0, 1.0, 1.0))

    # def update_paddle_shader_light_dirs(self, paddle):
    #     id = f"{paddle.team}{paddle.player}"
    #     positions = [self.get_light_pos(i) for i in range(4)]
    #     array = self.paddle_light_dir_dict[id]
    #     for i, position in enumerate(positions):
    #         dir = (paddle.pos - position) / np.linalg.norm(paddle.pos - position)
    #         array[i*2] = dir[0]
    #         array[i*2 + 1] = dir[1]

    #     location = pr.get_shader_location(self.paddle_shader, "LightDir")
    #     pr.set_shader_value_v(self.paddle_shader, location, array, pr.SHADER_UNIFORM_VEC2, len(positions) * 2)

    def set_shapes_texture(self):
        # Create a 1x1 white image
        image = pr.gen_image_color(1, 1, pr.WHITE)

        # Load the image as a texture
        texture = pr.load_texture_from_image(image)

        # Use the texture for shape drawing
        pr.set_texture_filter(texture, pr.TEXTURE_FILTER_POINT)

        # Unload the image (texture is still valid)
        pr.unload_image(image)

        return texture

    def begin_drawing_paddle(self, paddle):
        paddle_pos_loc_paddle = pr.get_shader_location(self.paddle_shader, "paddlePos")
        paddle_pos = pr.ffi.new('float[2]', [self.world_to_screen_coord(paddle.pos)[0], self.world_to_screen_coord(paddle.pos)[1]])
        pr.set_shader_value(self.paddle_shader, paddle_pos_loc_paddle, paddle_pos, pr.SHADER_UNIFORM_VEC2)
        pr.begin_shader_mode(self.paddle_shader)

    def end_drawing_paddle(self):
        pr.end_shader_mode()

    def reset(self):
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()
        pr.set_config_flags(self.flags)
        pr.init_window(1280, 720, "Game")
        monitor = pr.get_current_monitor()
        self.monitor_width = pr.get_monitor_width(monitor)
        self.monitor_height = pr.get_monitor_height(monitor)
        print(f"{self.monitor_width}, {self.monitor_height}")
        self.render_texture = pr.load_render_texture(*self.get_resolution())
        # pr.set_texture_filter(self.render_texture.texture, pr.TEXTURE_FILTER_ANISOTROPIC_16X)

    def update_particles(self):
        # self.particles = list(filter(lambda x: x.is_alive(), self.particles))
        self.temporary_particles = list(filter(lambda x: x.is_alive(), self.temporary_particles))
        for particle in self.temporary_particles:
            particle.update()

        # while len(self.particles) < 20:
        #     self.particles.append(Particle.random_particle(h.field_mid()))

    def draw_particles(self):
        pr.begin_shader_mode(self.particle_shader)
        for particle in self.temporary_particles:
            particle.draw()

        pr.end_shader_mode()

    def add_temporary_particles(self, pos, vel, colors):
        n = int(vel / 10)
        particles = Particle.random_particles(pos, n, colors)
        self.temporary_particles += particles

    def update_light_intensities(self):
        intensity = 0.75
        intensities = [intensity] * 4
        scale = 5
        t = g.current_time
        noise_value = noise.pnoise1(t * scale)
        amplitude = 0.05
        light_intensity = intensity + amplitude * noise_value
        light_intensity = max(0.0, min(1.0, light_intensity))
        intensities[0] = light_intensity

        scale = 10
        t = g.current_time
        noise_value = noise.pnoise1(t * scale)
        amplitude = 0.1
        light_intensity = intensity + amplitude * noise_value
        light_intensity = max(0.0, min(1.0, light_intensity))
        intensities[3] = light_intensity

        self.light_intensities = intensities

    def get_light_pos(self, idx):
        y_offset = 0
        match idx:
            case 0:
                return (c.settings['field_width'] / 4, y_offset)
            case 1:
                return (3 * c.settings['field_width'] / 4, y_offset)
            case 2:
                return (c.settings['field_width'] / 4, c.settings['field_height'] - y_offset)
            case 3:
                return (3 * c.settings['field_width'] / 4, c.settings['field_height'] - y_offset)

    def create_light_data(self):
        positions = [self.get_light_pos(i) for i in range(4)]
        self.update_light_intensities()

        positions = list(map(self.world_to_screen_coord, positions))
        for i, position in enumerate(positions):
            self.light_data[3*i] = position[0]
            self.light_data[3*i + 1] = position[1]
            self.light_data[3*i + 2] = self.light_intensities[i]

        location = pr.get_shader_location(self.fxaa_shader, "LightBuffer")
        pr.set_shader_value_v(self.fxaa_shader, location, self.light_data, pr.SHADER_UNIFORM_VEC3, len(positions) * 3)

        location = pr.get_shader_location(self.paddle_shader, "LightBuffer")
        pr.set_shader_value_v(self.paddle_shader, location, self.light_data, pr.SHADER_UNIFORM_VEC3, len(positions) * 3)


    def get_resolution(self):
        return c.resolutions[self.current_resolution_idx]

    def draw_fps(self, x, y):
        self.draw_text(f"FPS: {pr.get_fps()}", 'steps_left', (255,255,255), (x, y), 'left', 0, 40)

    def update_paddle_data(self, paddles):
        self.paddle_count[0] = len(paddles)
        for i, paddle in enumerate(paddles):
            screen_pos = self.world_to_screen_coord(paddle.pos)
            base_index = i * 9
            self.paddle_buffer[base_index] = paddle.color[0] / 255.0
            self.paddle_buffer[base_index + 1] = paddle.color[1] / 255.0
            self.paddle_buffer[base_index + 2] = paddle.color[2] / 255.0
            self.paddle_buffer[base_index + 3] = screen_pos[0]
            self.paddle_buffer[base_index + 4] = screen_pos[1]
            self.paddle_buffer[base_index + 5] = self.world_to_screen_length(paddle.radius)
            self.paddle_buffer[base_index + 6] = paddle.charging_alpha() if paddle.charging_dash else 0

        location = pr.get_shader_location(self.fxaa_shader, "PaddleBuffer")
        pr.set_shader_value_v(self.fxaa_shader, location, self.paddle_buffer, pr.SHADER_UNIFORM_VEC3, len(paddles) * 3)
        location = pr.get_shader_location(self.fxaa_shader, "paddleCount")
        pr.set_shader_value(self.fxaa_shader, location, self.paddle_count, pr.SHADER_UNIFORM_INT)

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

    def handle_input(self):
        if g.controls.stick == None:
            g.controls.init_controls()

        new_presses = set()
        for key in range(pr.KEY_A, pr.KEY_Z + 1):
            if pr.is_key_pressed(key):
                if key not in self.keys_pressed:
                    new_presses.add(key)

                self.keys_pressed.add(key)

        for key in list(self.keys_pressed):
            if pr.is_key_released(key):
                self.keys_pressed.remove(key)

        return new_presses

    def handle_keyboard_input(self):
        new_presses = self.handle_input()
        if pr.KEY_E in new_presses:
            if c.settings['is_training']:
                self.fps_locked = not self.fps_locked
                if self.fps_locked:
                    pr.set_target_fps(c.settings['fps'])
                else:
                    pr.set_target_fps(0)
                print(f"Setting FPS locked: {self.fps_locked}")
        elif pr.KEY_R in new_presses:
            c.settings['no_render'] = not c.settings['no_render']
            print(f"Setting rendering to {not c.settings['no_render']}")
        elif pr.KEY_M in new_presses:
            c.settings['no_sound'] = not c.settings['no_sound']
            print(f"Setting sound to {not c.settings['no_sound']}")
        elif pr.KEY_T in new_presses:
            c.settings['player_2_human'] = not c.settings['player_2_human']
            print(f"Setting player 2 human to {c.settings['player_2_human']}")
        elif pr.KEY_F in new_presses:
            self.toggle_fullscreen()
            print(f"Toggling fullscreen")
        elif pr.KEY_U in new_presses:
            self.change_resolution()
            print(f"Changing resolution")
        elif pr.KEY_P in new_presses:
            c.settings['paused'] = not c.settings['paused']
            print(f"Paused game")
        elif pr.KEY_V in new_presses:
            h.save_model_name()
            print(f"Saving name of current model")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            pr.set_window_size(self.monitor_width, self.monitor_height)
            pr.toggle_fullscreen()
            pr.set_window_size(self.monitor_width, self.monitor_height)
        else:
            pr.set_window_size(1280, 720)
            pr.toggle_fullscreen()
            pr.set_window_size(1280, 720)

    def rendering_off_message(self):
        self.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.draw_text("Rendering disabled", 10, 10, 20, pr.WHITE)
        self.end_drawing()

    def paused(self):
        self.begin_drawing()
        self.end_drawing()

    def change_resolution(self):
        self.current_resolution_idx = (self.current_resolution_idx + 1) % len(c.resolutions)
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()
        self.render_texture = pr.load_render_texture(*self.get_resolution())
        resolution = pr.ffi.new('float[2]', [float(self.get_resolution()[0]), float(self.get_resolution()[1])])
        pr.set_shader_value(self.fxaa_shader, self.resolution_loc, resolution, pr.SHADER_UNIFORM_VEC2)
        y_extremes = pr.ffi.new('float[2]', [self.world_to_screen_coord((0, 0))[1], self.world_to_screen_coord((0, c.settings['field_height']))[1]])
        pr.set_shader_value(self.fxaa_shader, self.y_extremes_loc, y_extremes, pr.SHADER_UNIFORM_VEC2)
        resolution_loc_paddle = pr.get_shader_location(self.paddle_shader, "resolution")
        pr.set_shader_value(self.paddle_shader, resolution_loc_paddle, resolution, pr.SHADER_UNIFORM_VEC2)


        if self.get_resolution()[0] > 600:
            # self.shader = None
            self.shader = self.fxaa_shader
        else:
            # self.shader = self.scanlines_shader
            self.shader = None

        # pr.set_window_size(*self.get_resolution())

    def handle_events(self):
        self.handle_keyboard_input()
        return not pr.window_should_close()

    def begin_drawing(self):
        pr.begin_texture_mode(self.render_texture)

    def end_drawing(self, paddles=None, shader=True):
        pr.end_texture_mode()
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        if not c.settings['is_training'] and self.shader is not None and shader:
            pr.begin_shader_mode(self.shader)

        if self.fullscreen:
            monitor = pr.get_current_monitor()
            maybe_width = pr.get_monitor_width(monitor)
            maybe_height = pr.get_monitor_height(monitor)
            width = self.monitor_width if self.monitor_width >= maybe_width else maybe_width
            height = self.monitor_height if self.monitor_height >= maybe_height else maybe_height
        else:
            width = pr.get_screen_width()
            height = pr.get_screen_height()

        scale = min(width / self.get_resolution()[0],
                    height / self.get_resolution()[1])
        dest_width = round(self.get_resolution()[0] * scale)
        dest_height = round(self.get_resolution()[1] * scale)
        dest_x = (width - dest_width) // 2
        dest_y = (height - dest_height) // 2

        pr.draw_texture_pro(
            self.render_texture.texture,
            pr.Rectangle(0, 0, self.get_resolution()[0], -self.get_resolution()[1]),
            pr.Rectangle(dest_x, dest_y, dest_width, dest_height),
            pr.Vector2(0, 0),
            0,
            pr.WHITE
        )

        if not c.settings['is_training'] and self.shader is not None and shader:
            pr.end_shader_mode()

        pr.end_drawing()

    def tick(self):
        g.current_time = pr.get_time() + 40

    def tuple_to_color(self, color_tuple):
        if len(color_tuple) == 3:
            return pr.Color(color_tuple[0], color_tuple[1], color_tuple[2], 255)
        elif len(color_tuple) == 4:
            return pr.Color(color_tuple[0], color_tuple[1], color_tuple[2], color_tuple[3])
        else:
            raise ValueError("Color tuple should have 3 or 4 elements")

    def fill_screen(self, color, dimensions):
        pr.clear_background(pr.BLACK)
        self.draw_rectangle(color, (0,0), dimensions)

    def fill_screen_semiopaque_black(self, opacity=10):
        self.begin_drawing()
        pr.draw_rectangle(0, 0, self.get_resolution()[0], self.get_resolution()[1], pr.Color(0, 0, 0, opacity))
        self.end_drawing(shader=False)

    def draw_rectangle(self, color, pos, size):
        color = self.tuple_to_color(color)
        pos = self.world_to_screen_coord(pos)
        size = (self.world_to_screen_length(size[0]), self.world_to_screen_length(size[1]))
        pr.draw_rectangle(int(pos[0]), int(pos[1]), int(size[0]), int(size[1]), color)

    def draw_transparent_rectangle(self, color, pos, size, opacity):
        pos = self.world_to_screen_coord(pos)
        size = (self.world_to_screen_length(size[0]), self.world_to_screen_length(size[1]))
        color_with_alpha = pr.Color(color[0], color[1], color[2], int(opacity * 255))
        pr.draw_rectangle(int(pos[0]), int(pos[1]), int(size[0]), int(size[1]), color_with_alpha)

    def draw_circle(self, pos, radius, color):
        color = self.tuple_to_color(color)
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        pr.draw_circle_sector(pr.Vector2(*pos), radius, 0, 360, max(30, int(radius/2)), color)

    def draw_circle_simple(self, pos, radius, color):
        color = self.tuple_to_color(color)
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        pr.draw_circle_sector(pr.Vector2(*pos), radius, 0, 360, 10, color)

    def draw_transparent_circle(self, pos, radius, color, opacity):
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        color_with_alpha = pr.Color(color[0], color[1], color[2], int(opacity * 255))
        pr.draw_circle(int(pos[0]), int(pos[1]), radius, color_with_alpha)

    def world_to_screen_coord(self, coord):
        x = round(coord[0] * self.scaling_factor + self.shift[0])
        y = round(coord[1] * self.scaling_factor + self.shift[1])
        return (x, y)

    def world_to_screen_length(self, length):
        return round(length * self.scaling_factor)

    def draw_text(self, text, font_name, color, position, alignment='left', rotation=0.0, font_size=20):
        color = self.tuple_to_color(color)
        position = self.world_to_screen_coord(position)
        font_size = self.world_to_screen_length(font_size)
        font = self.fonts[font_name]
        text_width = pr.measure_text_ex(font, text, font_size, 0).x


        if alignment == 'center':
            position = (position[0] - text_width // 2, position[1])
        elif alignment == 'right':
            position = (position[0] - text_width, position[1])

        pr.draw_text_pro(font, text, pr.Vector2(position[0], position[1]), pr.Vector2(0, 0), rotation, font_size, 0, color)

    def draw_dict(self, dictionary, font_name, pos, font_size=20):
        x, y = pos
        label_value_gap = 440
        line_height = 30
        items = list(dictionary.items())
        total_height = len(items) * line_height
        y -= total_height

        max_int_width = 0
        max_frac_width = 0
        for value in dictionary.values():
            value = round(value)
            int_part, _, frac_part = f"{value}".partition('.')
            max_int_width = max(max_int_width, pr.measure_text(int_part, font_size))
            max_frac_width = max(max_frac_width, pr.measure_text(frac_part, font_size))

        for key, value in reversed(items):
            value = round(value)
            label_text = f"{key}:"
            self.draw_text(label_text, font_name, (255,255,255), (x - label_value_gap - pr.measure_text('.', font_size), y), 'left', 0, font_size)

            int_part, _, frac_part = f"{value}".partition('.')
            int_x = x - max_frac_width - pr.measure_text('.', font_size)
            self.draw_text(int_part, font_name, (255,255,255), (int_x, y), 'right', 0, font_size)
            # self.draw_text('.', font_name, (255,255,255), (int_x + pr.measure_text(int_part, font_size), y), 'left', 0, font_size)
            # self.draw_text(frac_part, font_name, (255,255,255), (int_x + pr.measure_text(int_part + '.', font_size), y), 'left', 0, font_size)

            y += line_height

    def draw_rotated_line_centered(self, pos, length, angle, color, width=1):
        color = self.tuple_to_color(color)
        pos = self.world_to_screen_coord(pos)
        length = self.world_to_screen_length(length)
        width = self.world_to_screen_length(width)

        start_x = pos[0] - length * math.cos(math.radians(angle)) / 2
        start_y = pos[1] - length * math.sin(math.radians(angle)) / 2

        end_x = pos[0] + length * math.cos(math.radians(angle)) / 2
        end_y = pos[1] + length * math.sin(math.radians(angle)) / 2

        pr.draw_line_ex(pr.Vector2(start_x, start_y), pr.Vector2(end_x, end_y), width, color)

    def draw_rotated_line(self, pos, length, angle, color, width=1):
        color = self.tuple_to_color(color)
        pos = self.world_to_screen_coord(pos)
        length = self.world_to_screen_length(length)
        width = self.world_to_screen_length(width)

        angle_rad = math.radians(angle)
        end_x = pos[0] + length * math.cos(angle_rad)
        end_y = pos[1] + length * math.sin(angle_rad)

        pr.draw_line_ex(pr.Vector2(pos[0], pos[1]), pr.Vector2(end_x, end_y), width, color)

    def close(self):
        for font in self.fonts.values():
            pr.unload_font(font)
        pr.close_window()