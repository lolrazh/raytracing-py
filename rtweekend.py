from __future__ import annotations
import math
import random
from typing import Final

# Constants
INFINITY: Final[float] = float('inf')
PI: Final[float] = math.pi

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * PI / 180.0

def random_double() -> float:
    """Returns a random real in [0,1)."""
    return random.random()

def random_double_range(min_val: float, max_val: float) -> float:
    """Returns a random real in [min,max)."""
    return min_val + (max_val - min_val) * random_double() 