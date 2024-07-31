import globals as g

class Trail:
    def __init__(self, decay, color, radius):
        self.entries = []
        self.decay = decay
        self.color = color
        self.radius = radius

    def update(self, new_entry):
        self.entries.append(new_entry)
        self.entries = list(map(lambda x: (x[0], x[1] * self.decay), self.entries))
        self.entries = list(filter(lambda x: x[1] > 0.01, self.entries))
        self.entries = sorted(self.entries, key=lambda x: x[1])

    def draw(self):
        # target_color = g.sound_handler.target_color()
        for pos, alpha in self.entries:
            curr_color = (*self.color, int(30 * alpha))
            g.framework.draw_circle_fast(pos, self.radius * (alpha ** (1/6)), curr_color)
