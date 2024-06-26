import globals as g

class UI():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UI, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True

    def draw(self, steps_left, current_reward, round_reward, time_left, score):
        self.draw_time_left(time_left)
        self.draw_score(score)
        if g.SETTINGS['is_training']:
            self.draw_reward(current_reward, round_reward)
            self.draw_steps_left(str(steps_left))

    def draw_steps_left(self, steps_left):
        text = f"steps left: {steps_left}"
        g.framework.draw_text(text, 'steps_left', g.STEPS_LEFT_COLOR, g.STEPS_LEFT_POS, centered=True)

    def draw_score(self, score):
        text1 = str(score[0])
        text2 = str(score[1])
        pos1 = (g.WIDTH / 2 - g.SCORE_POS[0], g.SCORE_POS[1])
        pos2 = (g.WIDTH / 2 + g.SCORE_POS[0], g.SCORE_POS[1])
        g.framework.draw_text(text1, 'score', g.SCORE_COLOR, pos1, centered=True)
        g.framework.draw_text(text2, 'score', g.SCORE_COLOR, pos2, centered=True)
        g.framework.draw_circle((pos1[0] - 60, pos1[1] - 5), 20, g.PADDLE_COLOR_1)
        g.framework.draw_circle((pos2[0] + 60, pos2[1] - 5), 20, g.PADDLE_COLOR_2)

    def draw_reward(self, current_reward, round_reward):
        current_reward = f"curr reward: {current_reward:.5}"
        g.framework.draw_text(current_reward, 'reward', g.REWARD_COLOR, g.REWARD_POS)

        round_reward = f"tot reward: {round_reward:.5}"
        g.framework.draw_text(round_reward, 'reward', g.REWARD_COLOR, (g.REWARD_POS[0], g.REWARD_POS[1] + 30))

    def draw_time_left(self, time_left):
        g.framework.draw_text(str(time_left), 'time_left', g.TIME_COLOR, g.TIME_POS, centered=True)



# class Slider:
#     def __init__(self, screen, x, y, width, height, min_val, max_val, start_val, label):
#         self.screen = screen
#         self.rect = pygame.Rect(x, y, width, height)
#         self.min_val = min_val
#         self.max_val = max_val
#         self.value = start_val
#         self.label = label
#         self.handle_rect = pygame.Rect(x + (start_val - min_val) / (max_val - min_val) * width, y, 10, height)
#         self.handle_color = g.interpolate_color((255,255,255), g.PADDLE_COLOR_2, 0.5)
#         self.text_color = g.TIME_COLOR
#         self.bg_color = g.interpolate_color((255,255,255), g.BG_COLOR, 0.5)
#         self.font = pygame.font.SysFont('Arial', int(20 * g.WIDTH / 2000))
#         self.dragging = False
#         self.rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
#         self.rect_surface.fill((*self.bg_color, 30))

#     def draw(self):
#         self.screen.blit(self.rect_surface, (self.rect.x, self.rect.y))
#         pygame.draw.rect(self.screen, self.handle_color, self.handle_rect)
#         label_surface = self.font.render(f'{self.label}: {self.value:.2f}', True, self.text_color)
#         self.screen.blit(label_surface, (self.rect.x, self.rect.y - int(25 * g.WIDTH / 2000)))

#     def handle_event(self, event):
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             if self.handle_rect.collidepoint(event.pos):
#                 self.dragging = True
#         elif event.type == pygame.MOUSEBUTTONUP:
#             self.dragging = False
#         elif event.type == pygame.MOUSEMOTION:
#             if self.dragging:
#                 mouse_x = event.pos[0]
#                 new_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
#                 self.handle_rect.x = new_x
#                 self.value = self.min_val + (new_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
#                 g.REWARD_POLICY[self.label] = self.value

#     def get_value(self):
#         return self.value
        
# def create_sliders(self):
#     y = int(g.HEIGHT - 400 * g.WIDTH / 2000)
#     x = int(g.WIDTH - 270 * g.WIDTH / 2000)
#     for key, value in g.REWARD_POLICY.items():
#         slider = Slider(self.screen, x, y, int(200 * g.WIDTH / 2000), int(20 * g.WIDTH / 2000), -max(np.abs(value) * 2, 10), max(np.abs(value * 2), 10), value, key)
#         self.sliders.append(slider)
#         y += int(50 * g.WIDTH / 2000)        
        
# def update_sliders(self):
#     for slider in self.sliders:
#         g.REWARD_POLICY[slider.label] = slider.get_value()            