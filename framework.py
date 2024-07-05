import pyray as pr
import math
import constants as c
import globals as g
import helpers as h
import ctypes

# class PaddleData(ctypes.Structure):
#     _fields_ = [
#         ("position", ctypes.c_float * 2),
#         ("color", ctypes.c_float * 3),
#         ("radius", ctypes.c_float)
#     ]

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

        # self.max_paddles = 16
        # PaddleDataArray = PaddleData * self.max_paddles
        # self.paddle_buffer = PaddleDataArray()
        # self.paddle_count = ctypes.c_int(0)

        self.max_paddles = c.settings['team_size'] * 2
        self.paddle_buffer = pr.ffi.new("float[]", self.max_paddles * 9)  # 2 for position, 3 for color, 1 for radius
        self.paddle_count = pr.ffi.new("int *", 0)

        self.tick()
        self.reset()

        self.keys_pressed = set()
        self.fps_locked = False

        if c.settings['is_training']:
            pr.set_target_fps(c.settings['fps'])
            self.fps_locked = True

        font_bold = pr.load_font_ex("fonts/Roboto-Bold.ttf", 800, None, 0)
        font_regular = pr.load_font_ex("fonts/Roboto-Regular.ttf", 800, None, 0)
        self.fxaa_shader = pr.load_shader("shaders/fxaa.vs", "shaders/fxaa.fs")
        self.scanlines_shader = pr.load_shader("shaders/scanlines.vs", "shaders/scanlines.fs")
        self.shader = self.fxaa_shader
        self.resolution_loc = pr.get_shader_location(self.fxaa_shader, "resolution")
        resolution = pr.ffi.new('float[2]', [float(self.get_resolution()[0]), float(self.get_resolution()[1])])
        pr.set_shader_value(self.fxaa_shader, self.resolution_loc, resolution, pr.SHADER_UNIFORM_VEC2)

        self.fonts = {
            'reward': font_bold,
            'time_left': font_bold,
            'steps_left': font_regular,
            'score': font_bold,
            'model_name': font_bold,
            'reward_breakdown': font_regular,
        }

    def reset(self):
        self.scaling_factor, self.shift = self.calculate_scaling_and_shift()
        pr.set_config_flags(self.flags)
        pr.init_window(1280, 720, "Game")
        self.render_texture = pr.load_render_texture(*self.get_resolution())
        # pr.set_texture_filter(self.render_texture.texture, pr.TEXTURE_FILTER_ANISOTROPIC_16X)

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
            self.paddle_buffer[base_index + 6] = paddle.velocity_alpha()

            # print(f"Paddle {i}: Pos=({screen_pos[0]}, {screen_pos[1]}), Color=({self.paddle_buffer[base_index+2]}, {self.paddle_buffer[base_index+3]}, {self.paddle_buffer[base_index+4]}), Radius={self.paddle_buffer[base_index+5]}")


        # Debug print
        # print(f"First paddle data: {self.paddle_buffer[0:6]}")

        # Update the shader uniform
        location = pr.get_shader_location(self.fxaa_shader, "PaddleBuffer")
        if location != -1:
            pr.set_shader_value_v(self.fxaa_shader, location, self.paddle_buffer, pr.SHADER_UNIFORM_VEC3, self.paddle_count[0] * 3)
        else:
            print("Failed to find PaddleBuffer in shader")

        location = pr.get_shader_location(self.fxaa_shader, "paddleCount")
        if location != -1:
            pr.set_shader_value(self.fxaa_shader, location, self.paddle_count, pr.SHADER_UNIFORM_INT)
        else:
            print("Failed to find paddleCount in shader")


        # # Update the shader uniform
        # pr.set_shader_value(self.fxaa_shader,
        #                     pr.get_shader_location(self.fxaa_shader, "PaddleBuffer"),
        #                     self.paddle_buffer,
        #                     pr.SHADER_UNIFORM_VEC3)
        # pr.set_shader_value(self.fxaa_shader,
        #                     pr.get_shader_location(self.fxaa_shader, "paddleCount"),
        #                     self.paddle_count,
        #                     pr.SHADER_UNIFORM_INT)




        # paddle_count = ctypes.c_int(0)
        # for i, paddle in enumerate(paddles):
        #     screen_pos = self.world_to_screen_coord(paddle.pos)
        #     self.paddle_buffer[i].position[0] = screen_pos[0]
        #     self.paddle_buffer[i].position[1] = screen_pos[1]
        #     self.paddle_buffer[i].color[0] = paddle.color[0] / 255.0
        #     self.paddle_buffer[i].color[1] = paddle.color[1] / 255.0
        #     self.paddle_buffer[i].color[2] = paddle.color[2] / 255.0
        #     self.paddle_buffer[i].radius = paddle.radius

        # # Update the shader uniform
        # pr.set_shader_value(self.fxaa_shader,
        #                     pr.get_shader_location(self.fxaa_shader, "PaddleBuffer"),
        #                     self.paddle_buffer,
        #                     pr.SHADER_UNIFORM_FLOAT)
        # pr.set_shader_value(self.fxaa_shader,
        #                     pr.get_shader_location(self.fxaa_shader, "paddleCount"),
        #                     self.paddle_count,
        #                     pr.SHADER_UNIFORM_INT)

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
        pr.toggle_fullscreen()

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
        if self.get_resolution()[0] > 600:
            self.shader = self.fxaa_shader
        else:
            self.shader = self.scanlines_shader

        # pr.set_window_size(*self.get_resolution())

    def handle_events(self):
        self.handle_keyboard_input()
        return not pr.window_should_close()

    def begin_drawing(self):
        pr.begin_texture_mode(self.render_texture)

    def end_drawing(self, paddles=None):
        pr.end_texture_mode()

        # if paddles is not None:
        #     pr.begin_texture_mode(self.paddle_data_texture)
        #     for i, paddle in enumerate(paddles):
        #         screen_coords = self.world_to_screen_coord(paddle.pos)
        #         pr.draw_pixel(i * 2, 0, pr.Color(
        #             screen_coords[0],
        #             screen_coords[1],
        #             int(paddle.radius),
        #             0,
        #         ))
        #         pr.draw_pixel(i * 2 + 1, 0, pr.Color(
        #             *paddle.color,
        #             int(paddle.velocity_alpha() * 255)  # Assuming glow_intensity is 0-1
        #         ))
        #     pr.end_texture_mode()

        #     pr.set_shader_value(self.fxaa_shader, pr.get_shader_location(self.fxaa_shader, "paddleData"), self.paddle_data_texture.texture, pr.SHADER_UNIFORM_SAMPLER2D)
        #     pr.set_shader_value(self.fxaa_shader, pr.get_shader_location(self.fxaa_shader, "paddleCount"), pr.ffi.new("int *", len(2 * c.settings['team_size'])), pr.SHADER_UNIFORM_INT)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.begin_shader_mode(self.shader)

        if self.fullscreen:
            monitor = pr.get_current_monitor()
            width = pr.get_monitor_width(monitor)
            height = pr.get_monitor_height(monitor)
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
        self.end_drawing()

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
        pr.draw_circle(int(pos[0]), int(pos[1]), radius, color)

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
        label_value_gap = 400
        line_height = 30
        items = list(dictionary.items())
        total_height = len(items) * line_height
        y -= total_height

        max_int_width = 0
        max_frac_width = 0
        for value in dictionary.values():
            int_part, _, frac_part = f"{value:.2f}".partition('.')
            max_int_width = max(max_int_width, pr.measure_text(int_part, font_size))
            max_frac_width = max(max_frac_width, pr.measure_text(frac_part, font_size))

        for key, value in reversed(items):
            label_text = f"{key}:"
            self.draw_text(label_text, font_name, (255,255,255), (x - label_value_gap - pr.measure_text('.', font_size), y), 'left', 0, font_size)

            int_part, _, frac_part = f"{value:.2f}".partition('.')
            int_x = x - max_frac_width - pr.measure_text('.', font_size)
            self.draw_text(int_part, font_name, (255,255,255), (int_x, y), 'left', 0, font_size)
            self.draw_text('.', font_name, (255,255,255), (int_x + pr.measure_text(int_part, font_size), y), 'left', 0, font_size)
            self.draw_text(frac_part, font_name, (255,255,255), (int_x + pr.measure_text(int_part + '.', font_size), y), 'left', 0, font_size)

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