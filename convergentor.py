class Convergentor:
    """Bez napovedy."""
    def __init__(self, init_tip = 200000):
        self.upper = None
        self.lower = None
        self._init_tip = init_tip

    def log_upper(self, new_upper):
        if (self.lower is not None) and (new_upper < self.lower):
            raise ValueError("Reported upper limit lower than the current lower limit!")
        if (self.upper is None) or (self.upper > new_upper):
            self.upper = new_upper
    
    def log_lower(self, new_lower):
        if (self.upper is not None) and (new_lower > self.upper):
            raise ValueError("Reported lower limit is larger than the current upper limit!")
        if (self.lower is None) or (self.lower < new_lower):
            self.lower = new_lower

    def tip(self):
        if (self.upper is None) and (self.lower is None):
            return self._init_tip
        elif (self.upper is None) and (self.lower is not None):
            return (1.4 * self.lower)
        elif (self.upper is not None) and (self.lower is None):
            return (0.8 * self.upper)
        elif (self.upper is not None) and (self.lower is not None):
            return (0.65 * self.upper + 0.35 * self.lower)