class Convergentor:
    """Bez napovedy."""
    def __init__(self):
        print("Convergentor initialized!")
        self.upper = None
        self.lower = None

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
            return 200000
        elif (self.upper is None) and (self.lower is not None):
            return (1.5 * self.lower)
        elif (self.upper is not None) and (self.lower is None):
            return (0.93 * self.upper)
        elif (self.upper is not None) and (self.lower is not None):
            return (0.75 * self.upper + 0.25 * self.lower)