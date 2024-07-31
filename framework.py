import pyray as pr
import math
import numpy as np
import constants as c
import globals as g
import helpers as h
from particle import Particle

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
        if not c.settings["is_training"]:
            self.flags |= pr.FLAG_VSYNC_HINT

        self.current_resolution_idx = c.settings["resolution"]
        self.fullscreen = False
        self.fps = c.settings["fps"]
        self.fps_locked = False
        if c.settings["is_training"]:
            pr.set_target_fps(c.settings["fps"])
            self.fps_locked = True

        self.render_texture = None
        self.tick()
        self.reset()
        self.keys_pressed = set()
        self.fonts = self.load_fonts()

        (self.post_shader, self.scanlines_shader,
        self.paddle_shader, self.particle_shader,
        self.puck_shader) = self.load_shaders()

        (self.paddle_data_buffer, self.paddle_count_buffer,
        self.light_data_buffer, self.resolution_buffer,
        self.y_extremes_buffer, self.paddle_pos_buffer,
        self.puck_pos_buffer, self.object_data_buffer,
        self.paddle_radius_buffer) = self.create_shader_buffers()

        self.shader = self.post_shader
        self.set_shader_uniform(self.post_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.paddle_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.puck_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.post_shader, "yExtremes", self.y_extremes_buffer, pr.SHADER_UNIFORM_VEC2)

        self.light_intensities = []
        self.particles = []

        self.set_shapes_texture()
        if not c.settings["is_training"]:
            self.toggle_fullscreen()

    def set_shader_uniform(self, shader, uniform_name, value, type, count=None):
        location = pr.get_shader_location(shader, uniform_name)
        if count is None:
            pr.set_shader_value(shader, location, value, type)
        else:
            pr.set_shader_value_v(shader, location, value, type, count)

    def load_shaders(self):
        post_shader = pr.load_shader("shaders/default.vs", "shaders/post.fs")
        scanlines_shader = pr.load_shader("shaders/scanlines.vs", "shaders/scanlines.fs")
        paddle_shader = pr.load_shader("shaders/default.vs", "shaders/paddle.fs")
        particle_shader = pr.load_shader("shaders/default.vs", "shaders/particle.fs")
        puck_shader = pr.load_shader("shaders/default.vs", "shaders/puck.fs")
        # path_shader = pr.load_shader("shaders/default.vs", "shaders/path.fs")

        return post_shader, scanlines_shader, paddle_shader, particle_shader, puck_shader

    def create_shader_buffers(self):
        max_paddles = c.settings["team_size"] * 2
        paddle_data_buffer = pr.ffi.new("float[]", max_paddles * 9)
        paddle_count_buffer = pr.ffi.new("int *", 0)
        paddle_radius_buffer = pr.ffi.new("float *", 0)
        light_data_buffer = pr.ffi.new("float[]", 10 * 3 * 2)
        resolution_buffer = pr.ffi.new("float[2]", [float(self.get_resolution()[0]), float(self.get_resolution()[1])])
        y_extremes_buffer = pr.ffi.new("float[2]", [self.world_to_screen_coord((0, 0))[1], self.world_to_screen_coord((0, c.settings["field_height"]))[1]])
        paddle_pos_buffer = pr.ffi.new("float[2]", [0,0])
        puck_pos_buffer = pr.ffi.new("float[2]", [0,0])
        object_data_buffer = pr.ffi.new("float[]", 5 * 3)

        return paddle_data_buffer, paddle_count_buffer, light_data_buffer, resolution_buffer, y_extremes_buffer, paddle_pos_buffer, puck_pos_buffer, object_data_buffer, paddle_radius_buffer

    def load_fonts(self):
        font_bold = pr.load_font_ex("fonts/Roboto-Bold.ttf", 100, None, 0)
        font_regular = pr.load_font_ex("fonts/Roboto-Regular.ttf", 100, None, 0)
        fonts = {
            "reward": font_bold,
            "time_left": font_bold,
            "steps_left": font_regular,
            "score": font_bold,
            "model_name": font_bold,
            "reward_breakdown": font_regular,
        }
        return fonts

    def set_shapes_texture(self):
        image = pr.gen_image_color(1, 1, pr.WHITE)
        texture = pr.load_texture_from_image(image)
        pr.set_texture_filter(texture, pr.TEXTURE_FILTER_POINT)
        pr.unload_image(image)
        pr.set_shapes_texture(texture, pr.Rectangle(0.0, 0.0, 1.0, 1.0))

    def begin_drawing_paddle(self, paddle):
        paddle_pos_screen = self.world_to_screen_coord(paddle.pos)
        self.paddle_pos_buffer[0] = paddle_pos_screen[0]
        self.paddle_pos_buffer[1] = paddle_pos_screen[1]
        paddle_radius_screen = self.world_to_screen_length(paddle.radius)
        self.paddle_radius_buffer[0] = paddle_radius_screen
        self.set_shader_uniform(self.paddle_shader, "paddlePos", self.paddle_pos_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.paddle_shader, "paddleRadius", self.paddle_radius_buffer, pr.SHADER_UNIFORM_FLOAT)
        pr.begin_shader_mode(self.paddle_shader)

    def begin_drawing_puck(self, puck):
        puck_pos_screen = self.world_to_screen_coord(puck.pos)
        self.puck_pos_buffer[0] = puck_pos_screen[0]
        self.puck_pos_buffer[1] = puck_pos_screen[1]
        self.set_shader_uniform(self.puck_shader, "puckPos", self.puck_pos_buffer, pr.SHADER_UNIFORM_VEC2)
        pr.begin_shader_mode(self.puck_shader)

    def end_drawing_puck(self):
        pr.end_shader_mode()

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
        self.particles = list(filter(lambda x: x.is_alive(), self.particles))
        for particle in self.particles:
            particle.update()

    def draw_particles(self):
        pr.begin_shader_mode(self.particle_shader)
        for particle in self.particles:
            particle.draw()

        pr.end_shader_mode()

    def add_temporary_particles(self, pos, vel, colors):
        if not h.full_visuals():
            return

        n = int(vel / 10)
        particles = Particle.random_particles(pos, n, colors)
        self.particles += particles

    def update_light_data(self, lights):
        for i, light in enumerate(lights):
            pos = self.world_to_screen_coord(light.pos)
            base_i = 3 * 2 * i
            color = h.color_float(light.color)
            self.light_data_buffer[base_i] = pos[0]
            self.light_data_buffer[base_i + 1] = pos[1]
            self.light_data_buffer[base_i + 2] = light.intensity
            self.light_data_buffer[base_i + 3] = color[0]
            self.light_data_buffer[base_i + 4] = color[1]
            self.light_data_buffer[base_i + 5] = color[2]

        self.set_shader_uniform(self.post_shader, "LightBuffer", self.light_data_buffer, pr.SHADER_UNIFORM_VEC3, 6 * 10)
        self.set_shader_uniform(self.paddle_shader, "LightBuffer", self.light_data_buffer, pr.SHADER_UNIFORM_VEC3, 6 * 10)
        self.set_shader_uniform(self.puck_shader, "LightBuffer", self.light_data_buffer, pr.SHADER_UNIFORM_VEC3, 6 * 10)

    def update_object_data(self, objects):
        for i, object in enumerate(objects):
            pos = self.world_to_screen_coord(object.pos)
            radius = self.world_to_screen_length(object.radius)
            self.object_data_buffer[3*i] = pos[0]
            self.object_data_buffer[3*i + 1] = pos[1]
            self.object_data_buffer[3*i + 2] = radius

        self.set_shader_uniform(self.post_shader, "ObjectBuffer", self.object_data_buffer, pr.SHADER_UNIFORM_VEC3, 3 * len(objects))


    def get_resolution(self):
        return c.resolutions[self.current_resolution_idx]

    def draw_fps(self, x, y):
        self.draw_text(f"FPS: {pr.get_fps()}", "steps_left", (255,255,255), (x, y), "left", 0, 40)

    def update_paddle_data(self, paddles):
        self.paddle_count_buffer[0] = len(paddles)
        for i, paddle in enumerate(paddles):
            screen_pos = self.world_to_screen_coord(paddle.pos)
            base_index = i * 9
            self.paddle_data_buffer[base_index] = paddle.color[0] / 255.0
            self.paddle_data_buffer[base_index + 1] = paddle.color[1] / 255.0
            self.paddle_data_buffer[base_index + 2] = paddle.color[2] / 255.0
            self.paddle_data_buffer[base_index + 3] = screen_pos[0]
            self.paddle_data_buffer[base_index + 4] = screen_pos[1]
            self.paddle_data_buffer[base_index + 5] = self.world_to_screen_length(paddle.radius)
            self.paddle_data_buffer[base_index + 6] = paddle.charging_alpha() if paddle.charging_dash else 0

        self.set_shader_uniform(self.post_shader, "PaddleBuffer", self.paddle_data_buffer, pr.SHADER_UNIFORM_VEC3, len(paddles) * 3)
        self.set_shader_uniform(self.post_shader, "paddleCount", self.paddle_count_buffer, pr.SHADER_UNIFORM_INT)

    def calculate_scaling_and_shift(self):
        x_stretch = (self.get_resolution()[0]) / c.settings["field_width"]
        y_stretch = (self.get_resolution()[1]) / c.settings["field_height"]
        scaling = min(x_stretch, y_stretch)

        if x_stretch > y_stretch:
            shift = ((self.get_resolution()[0] - c.settings["field_width"] * scaling) / 2, 0)
        elif y_stretch > x_stretch:
            shift = (0, (self.get_resolution()[1] - c.settings["field_height"] * scaling) / 2)
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
            if c.settings["is_training"]:
                self.fps_locked = not self.fps_locked
                if self.fps_locked:
                    pr.set_target_fps(c.settings["fps"])
                else:
                    pr.set_target_fps(0)
                print(f"Setting FPS locked: {self.fps_locked}")
        elif pr.KEY_R in new_presses:
            c.settings["no_render"] = not c.settings["no_render"]
            print(f"Setting rendering to {not c.settings["no_render"]}")
        elif pr.KEY_M in new_presses:
            c.settings["no_sound"] = not c.settings["no_sound"]
            print(f"Setting sound to {not c.settings["no_sound"]}")
        elif pr.KEY_F in new_presses:
            self.toggle_fullscreen()
            print(f"Toggling fullscreen")
        elif pr.KEY_U in new_presses:
            self.change_resolution()
            print(f"Changing resolution")
        elif pr.KEY_P in new_presses:
            c.settings["paused"] = not c.settings["paused"]
            print(f"Paused game")
        elif pr.KEY_V in new_presses:
            h.save_model_name()
            print(f"Saving name of current model")

        return new_presses

    def take_paused_step(self, new_presses):
        if pr.KEY_O in new_presses:
            return True

        return False

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            pr.set_window_size(self.monitor_width, self.monitor_height)
            pr.toggle_fullscreen()
            pr.set_window_size(self.monitor_width, self.monitor_height)
            pr.hide_cursor()
        else:
            pr.set_window_size(1280, 720)
            pr.toggle_fullscreen()
            pr.set_window_size(1280, 720)
            pr.show_cursor()

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
        self.resolution_buffer[0] = float(self.get_resolution()[0])
        self.resolution_buffer[1] = float(self.get_resolution()[1])
        self.y_extremes_buffer[0] = self.world_to_screen_coord((0, 0))[1]
        self.y_extremes_buffer[1] = self.world_to_screen_coord((0, c.settings["field_height"]))[1]
        self.set_shader_uniform(self.post_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.paddle_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.puck_shader, "resolution", self.resolution_buffer, pr.SHADER_UNIFORM_VEC2)
        self.set_shader_uniform(self.post_shader, "yExtremes", self.y_extremes_buffer, pr.SHADER_UNIFORM_VEC2)
        self.shader = self.post_shader

    def handle_events(self):
        new_presses = self.handle_keyboard_input()
        return (not pr.window_should_close(), new_presses)

    def begin_drawing(self):
        pr.begin_texture_mode(self.render_texture)

    def end_drawing(self, paddles=None, shader=True):
        pr.end_texture_mode()
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        if h.full_visuals() and self.shader is not None and shader:
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

        if h.full_visuals() and self.shader is not None and shader:
            pr.end_shader_mode()

        pr.end_drawing()

    def tick(self):
        if c.settings["is_training"]:
            g.current_time += c.settings["delta_t"] / c.settings["original_delta_t"]
        else:
            g.current_time = g.clock.get_time() + 85

    def actual_time(self):
        return pr.get_time()

    def tuple_to_color(self, color_tuple):
        return pr.Color(*color_tuple) if len(color_tuple) == 4 else pr.Color(*color_tuple, 255)
        # if len(color_tuple) == 3:
        #     return pr.Color(color_tuple[0], color_tuple[1], color_tuple[2], 255)
        # elif len(color_tuple) == 4:
        #     return pr.Color(color_tuple[0], color_tuple[1], color_tuple[2], color_tuple[3])
        # else:
        #     raise ValueError("Color tuple should have 3 or 4 elements")

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

    def draw_rectangle_fast(self, pyray_color, pos, size):
        pos = self.world_to_screen_coord_fast(pos)
        size = (self.world_to_screen_length_fast(size[0]), self.world_to_screen_length_fast(size[1]))
        pr.draw_rectangle(int(pos[0]), int(pos[1]), int(size[0]), int(size[1]), pyray_color)

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

    def draw_circle_fast(self, pos, radius, pyray_color):
        pos = self.world_to_screen_coord(pos)
        radius = self.world_to_screen_length(radius)
        pr.draw_circle_sector(pr.Vector2(*pos), radius, 0, 360, max(30, int(radius/2)), pyray_color)

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

    def world_to_screen_coord_fast(self, coord):
        x = int(coord[0] * self.scaling_factor + self.shift[0])
        y = int(coord[1] * self.scaling_factor + self.shift[1])
        return (x, y)

    def world_to_screen_length(self, length):
        return round(length * self.scaling_factor)

    def world_to_screen_length_fast(self, length):
        return int(length * self.scaling_factor)

    def draw_text(self, text, font_name, color, position, alignment="left", rotation=0.0, font_size=20):
        color = self.tuple_to_color(color)
        position = self.world_to_screen_coord(position)
        font_size = self.world_to_screen_length(font_size)
        font = self.fonts[font_name]
        text_width = pr.measure_text_ex(font, text, font_size, 0).x
        if alignment == "center":
            position = (position[0] - text_width // 2, position[1])
        elif alignment == "right":
            position = (position[0] - text_width, position[1])

        pr.draw_text_pro(font, text, pr.Vector2(position[0], position[1]), pr.Vector2(0, 0), rotation, font_size, 0, color)

    def draw_dict(self, dictionary, font_name, pos, font_size=20):
        x, y = pos
        line_height = 30
        items = list(dictionary.items())
        total_height = len(items) * line_height
        y -= total_height

        if "total" in dictionary:
            keys = [key for key in dictionary if key != "total"] + ["total"]
        else:
            keys = [key for key in dictionary]

        decimal_places = 2

        for key in keys:
            value = dictionary[key]
            try:
                value = value if not isinstance(value[0], float) else (round(value[0]), value[1])
            except:
                pass
            label_text = f"{key}:"
            try:
                self.draw_text(label_text, font_name, (255,255,255), (x - 550 - pr.measure_text(".", font_size), y), "left", 0, font_size)
                self.draw_text(str(value[0]), font_name, (255,255,255), (x - 120, y), "right", 0, font_size)
                self.draw_text(f"{value[1]:.{decimal_places}f}", font_name, (255,255,255), (x, y), "right", 0, font_size)
            except:
                self.draw_text(label_text, font_name, (255,255,255), (x - 550 - pr.measure_text(".", font_size), y), "left", 0, font_size)
                self.draw_text(str(value), font_name, (255,255,255), (x - 120, y), "right", 0, font_size)
                # self.draw_text(str(value), font_name, (255,255,255), (x, y), "right", 0, font_size)

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

    # def draw_puck_path(self, puck):
    #     color = (255,255,255)
    #     # target_color = g.sound_handler.target_color()
    #     for pos, alpha in puck.puck_path:
    #         curr_color = (*color, int(30 * alpha))
    #         self.draw_circle(pos, puck.radius * (alpha ** (1/6)), curr_color)

    # def draw_trail(self, trail):
    #     color = (255,255,255)
    #     # target_color = g.sound_handler.target_color()
    #     for pos, alpha in puck.puck_path:
    #         curr_color = (*color, int(30 * alpha))
    #         self.draw_circle(pos, puck.radius * (alpha ** (1/6)), curr_color)

    # def draw_gradient_rectangles(self, points, radius):
    #     num_points = len(points)
    #     if num_points > 1:
    #         for i in range(num_points - 1):
    #             pr.begin_shader_mode(self.path_shader)
    #             x1, y1 = self.world_to_screen_coord(points[i])
    #             x2, y2 = self.world_to_screen_coord(points[i+1])
    #             angle = math.atan2(y2 - y1, x2 - x1)
    #             length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    #             alpha_1 = (float(i) / float(num_points)) * 0.5
    #             alpha_2 = ((float(i + 1)) / float(num_points)) * 0.5

    #             colStart = (1.0, 1.0, 1.0, alpha_1)
    #             colEnd = (1.0, 1.0, 1.0, alpha_2)

    #             self.path_col_start_buffer[0] = colStart[0]
    #             self.path_col_start_buffer[1] = colStart[1]
    #             self.path_col_start_buffer[2] = colStart[2]
    #             self.path_col_start_buffer[3] = colStart[3]
    #             self.path_col_end_buffer[0] = colEnd[0]
    #             self.path_col_end_buffer[1] = colEnd[1]
    #             self.path_col_end_buffer[2] = colEnd[2]
    #             self.path_col_end_buffer[3] = colEnd[3]
    #             self.set_shader_uniform(self.path_shader, "colStart", self.path_col_start_buffer, pr.SHADER_UNIFORM_VEC4)
    #             self.set_shader_uniform(self.path_shader, "colEnd", self.path_col_end_buffer, pr.SHADER_UNIFORM_VEC4)

    #             pr.draw_rectangle_pro(
    #                 pr.Rectangle((x1 + x2) / 2, (y1 + y2) / 2, length, radius * 2),
    #                 pr.Vector2(length / 2, radius),
    #                 np.degrees(angle),
    #                 pr.WHITE
    #             )
    #             pr.end_shader_mode()


    def close(self):
        for font in self.fonts.values():
            pr.unload_font(font)
        pr.close_window()

