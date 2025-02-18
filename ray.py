from __future__ import annotations
from vec3 import vec3, point3

class ray:
    def __init__(self, origin: point3 = None, direction: vec3 = None):
        """Initialize a ray with origin and direction.
        If no parameters provided, creates a default ray."""
        self.orig: point3 = origin if origin is not None else point3()
        self.dir: vec3 = direction if direction is not None else vec3()
    
    def origin(self) -> point3:
        """Get the ray's origin."""
        return self.orig
    
    def direction(self) -> vec3:
        """Get the ray's direction."""
        return self.dir
    
    def at(self, t: float) -> point3:
        """Get the point at parameter t along the ray.
        P(t) = A + tb, where:
        - P is the position
        - A is the origin
        - b is the direction
        - t is the parameter"""
        return self.orig + t * self.dir 