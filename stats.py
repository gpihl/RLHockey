import numpy as np

class Stats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        if self.n < 2:
            return self.mean, 0
        else:
            variance = self.M2 / (self.n - 1)
            return self.mean, np.sqrt(variance)
