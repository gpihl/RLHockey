import pyray as pr

class Clock:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Clock, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.start_time = pr.get_time()
        self.paused_time = 0
        self.is_paused = False
        self.pause_start_time = 0
        self.pause()

    def pause(self):
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = pr.get_time()

    def unpause(self):
        if self.is_paused:
            self.is_paused = False
            self.paused_time += pr.get_time() - self.pause_start_time

    def get_time(self):
        if self.is_paused:
            return self.pause_start_time - self.start_time - self.paused_time
        else:
            return pr.get_time() - self.start_time - self.paused_time