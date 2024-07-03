from pathlib import Path
import random
import os
import numpy as np
import constants as c
import globals as g

def field_mid_x():
    return c.settings['field_width'] / 2

def field_mid_y():
    return c.settings['field_height'] / 2

def field_mid():
    return np.array([field_mid_x(), field_mid_y()])

def field_left():
    return 0.0

def field_right():
    return c.settings['field_width']

def field_top():
    return 0.0

def field_bot():
    return c.settings['field_height']

def corner_top_left():
    return np.array([c.settings['corner_radius'], c.settings['corner_radius']])

def corner_top_right():
    return np.array([field_right() - c.settings['corner_radius'], c.settings['corner_radius']])

def corner_bot_left():
    return np.array([c.settings['corner_radius'], field_bot() - c.settings['corner_radius']])

def corner_bot_right():
    return np.array([field_right() - c.settings['corner_radius'], field_bot() - c.settings['corner_radius']])

def goal_pos(goal_idx):
    if goal_idx == 1:
        goal_pos = np.array([0, field_mid_y()])
    else:
        goal_pos = np.array([field_right(), field_mid_y()])

    return goal_pos

def scale(vec, x_max, y_max):
    return np.array([vec[0] / x_max, vec[1] / y_max])

def goal_top():
    return (c.settings['field_height'] - c.settings['goal_height']) / 2

def goal_top_pos(goal_idx):
    return (0, goal_top()) if goal_idx == 1 else (c.settings['field_width'], goal_top())

def goal_bot_pos(goal_idx):
    return (0, goal_bottom()) if goal_idx == 1 else (c.settings['field_width'], goal_bottom())

def goal_bottom():
    return c.settings['goal_height'] + (c.settings['field_height'] - c.settings['goal_height']) / 2

def dist_alpha(dist):
    return 1.0 - min(1.0, dist / c.settings['field_width'])

def interpolate_color_rgb(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )

def rgb_to_hsl(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    l = (max_val + min_val) / 2

    if max_val == min_val:
        h = s = 0
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6

    return h, s, l

def hsl_to_rgb(h, s, l):
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            t += 1 if t < 0 else 0
            t -= 1 if t > 1 else 0
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return int(r * 255), int(g * 255), int(b * 255)

def interpolate_color(color1, color2, t):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    h1, s1, l1 = rgb_to_hsl(r1, g1, b1)
    h2, s2, l2 = rgb_to_hsl(r2, g2, b2)

    h = h1 + (h2 - h1) * t
    s = s1 + (s2 - s1) * t
    l = l1 + (l2 - l1) * t

    if abs(h2 - h1) > 0.5:
        if h2 > h1:
            h1 += 1.0
        else:
            h2 += 1.0
        h = h1 + (h2 - h1) * t
        h = h - 1.0 if h > 1.0 else h

    return hsl_to_rgb(h, s, l)

def modify_hsl(rgb, h_mod=0, s_mod=0, l_mod=0):
    r, g, b = rgb

    h, s, l = rgb_to_hsl(r, g, b)

    h = (h + h_mod) % 1.0
    s = max(0, min(1, s + s_mod))
    l = max(0, min(1, l + l_mod))

    return hsl_to_rgb(h, s, l)

def set_h(rgb, h):
    r, g, b = rgb
    _, s, l = rgb_to_hsl(r, g, b)
    return hsl_to_rgb(h, s, l)

def set_s(rgb, s):
    r, g, b = rgb
    h, _, l = rgb_to_hsl(r, g, b)
    return hsl_to_rgb(h, s, l)

def set_l(rgb, l):
    r, g, b = rgb
    h, s, _ = rgb_to_hsl(r, g, b)
    return hsl_to_rgb(h, s, l)

def clamp_s(rgb, mins, maxs):
    r, g, b = rgb
    h, s, l = rgb_to_hsl(r, g, b)
    s = max(mins, min(maxs, s))
    return hsl_to_rgb(h, s, l)

def clamp_l(rgb, minl, maxl):
    r, g, b = rgb
    h, s, l = rgb_to_hsl(r, g, b)
    l = max(minl, min(maxl, l))
    return hsl_to_rgb(h, s, l)

def save_text_to_file(text, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(text + '\n\n')
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_latest_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(base_path, latest_model)

def get_latest_model_path_with_algorithm(base_path, algorithm=None, players_per_team=1):
    if algorithm is None:
        algorithm = random.choice(['PPO'])

    models = [f for f in os.listdir(base_path) if f.startswith(f"{players_per_team}_{algorithm}") and f.endswith('.zip')]
    if not models:
        return None, algorithm

    latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return os.path.join(base_path, latest_model), algorithm


def get_random_model_path(base_path, prefix):
    models = [f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not models:
        return None

    models.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
    random_index = max(0, len(models) - int(np.abs(np.random.normal(0, 0.25, 1)[0]) * len(models)) - 1)
    print(f"random_index: {random_index}")
    random_model = models[random_index]
    return os.path.join(base_path, random_model)

def get_next_model_path(base_path, algorithm, players_per_team=1):
    models = [f for f in os.listdir(base_path) if f.startswith(f"{players_per_team}_{algorithm}") and f.endswith('.zip')]
    if not models:
        next_model_number = 1
    else:
        latest_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_number = int(latest_model.split('_')[-1].split('.')[0])
        next_model_number = latest_number + 1
    return os.path.join(base_path, f"{players_per_team}_{algorithm}_{next_model_number}.zip")

def get_sorted_zip_files(directory):
    dir_path = Path(directory).resolve()
    zip_files = [f for f in dir_path.glob('*.zip') if f.is_file()]
    sorted_files = sorted(zip_files, key=lambda x: x.stat().st_mtime)
    return [f.name for f in sorted_files]

def get_random_model_with_algorithm(players_per_team=1):
    models = get_sorted_zip_files('models')
    models = list(filter(lambda x: x.startswith(str(players_per_team)), models))
    if len(models) == 0:
        return None, None

    random_index = max(0, len(models) - int(np.abs(np.random.normal(0, 0.25, 1)[0]) * len(models)) - 1)
    random_model = models[random_index]
    algorithm = get_model_algorithm(random_model)
    path = os.path.join('models', random_model)
    return path, algorithm

def save_model_name():
    save_text_to_file(g.current_model_name, 'model_name/name.txt')

def get_model_algorithm(file_name):
    return file_name.split('_')[1].strip()

def clip_vector_length_inplace(vector, max_length=1):
    magnitude = np.linalg.norm(vector)
    if magnitude > max_length:
        vector *= (max_length / magnitude)

    return vector

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(dot_product / magnitudes) * 360 / (2 * np.pi)

def signed_angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    angle = np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

    angle_deg = np.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg