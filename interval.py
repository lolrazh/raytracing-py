from __future__ import annotations
from dataclasses import dataclass
from rtweekend import INFINITY

@dataclass
class interval:
    """A class to manage real-valued intervals."""
    min: float
    max: float

    def __init__(self, min_val: float = INFINITY, max_val: float = -INFINITY):
        """Initialize an interval. Default interval is empty."""
        self.min = min_val
        self.max = max_val
    
    def size(self) -> float:
        """Get the size of the interval."""
        return self.max - self.min
    
    def contains(self, x: float) -> bool:
        """Return true if the interval contains the value x."""
        return self.min <= x <= self.max
    
    def surrounds(self, x: float) -> bool:
        """Return true if the interval strictly contains the value x."""
        return self.min < x < self.max
    
    def clamp(self, x: float) -> float:
        """Clamp the value x to the interval."""
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x

# Static interval instances
empty = interval(INFINITY, -INFINITY)
universe = interval(-INFINITY, INFINITY) 