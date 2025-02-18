from __future__ import annotations
import math
from rtweekend import random_double, random_double_range


class vec3:
    def __init__(self, e0: float = 0, e1: float = 0, e2: float = 0):
        self.e = [e0, e1, e2]
    
    def x(self) -> float:
        return self.e[0]
    
    def y(self) -> float:
        return self.e[1]
    
    def z(self) -> float:
        return self.e[2]
    
    def __neg__(self) -> vec3:
        return vec3(-self.e[0], -self.e[1], -self.e[2])
    
    def __getitem__(self, i: int) -> float:
        return self.e[i]
    
    def __setitem__(self, i: int, value: float):
        self.e[i] = value
    
    def __iadd__(self, v: vec3) -> vec3:
        self.e[0] += v.e[0]
        self.e[1] += v.e[1]
        self.e[2] += v.e[2]
        return self
    
    def __imul__(self, t: float) -> vec3:
        self.e[0] *= t
        self.e[1] *= t
        self.e[2] *= t
        return self
    
    def __itruediv__(self, t: float) -> vec3:
        return self.__imul__(1/t)
    
    def length(self) -> float:
        return math.sqrt(self.length_squared())
    
    def length_squared(self) -> float:
        return self.e[0]*self.e[0] + self.e[1]*self.e[1] + self.e[2]*self.e[2]
    
    def near_zero(self) -> bool:
        """Return true if the vector is close to zero in all dimensions."""
        s = 1e-8
        return (abs(self.e[0]) < s) and (abs(self.e[1]) < s) and (abs(self.e[2]) < s)
    
    def __str__(self) -> str:
        return f"{self.e[0]} {self.e[1]} {self.e[2]}"
    
    # Vector Utility Functions
    def __add__(self, v: vec3) -> vec3:
        return vec3(self.e[0] + v.e[0], self.e[1] + v.e[1], self.e[2] + v.e[2])
    
    def __sub__(self, v: vec3) -> vec3:
        return vec3(self.e[0] - v.e[0], self.e[1] - v.e[1], self.e[2] - v.e[2])
    
    def __mul__(self, other) -> vec3:
        if isinstance(other, vec3):
            return vec3(self.e[0] * other.e[0], self.e[1] * other.e[1], self.e[2] * other.e[2])
        # other is a scalar
        return vec3(self.e[0] * other, self.e[1] * other, self.e[2] * other)
    
    def __rmul__(self, other) -> vec3:
        return self.__mul__(other)
    
    def __truediv__(self, t: float) -> vec3:
        return self * (1/t)

    @staticmethod
    def random() -> vec3:
        """Generate a random vector with components in [0,1)."""
        return vec3(random_double(), random_double(), random_double())
    
    @staticmethod
    def random_range(min_val: float, max_val: float) -> vec3:
        """Generate a random vector with components in [min,max)."""
        return vec3(
            random_double_range(min_val, max_val),
            random_double_range(min_val, max_val),
            random_double_range(min_val, max_val)
        )


# Type aliases
point3 = vec3   # 3D point
color = vec3    # RGB color


# Vector Utility Functions
def dot(u: vec3, v: vec3) -> float:
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]

def cross(u: vec3, v: vec3) -> vec3:
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0])

def unit_vector(v: vec3) -> vec3:
    return v / v.length()

def random_unit_vector() -> vec3:
    """Generate a random unit vector on the unit sphere surface."""
    while True:
        p = vec3.random_range(-1, 1)
        length_squared = p.length_squared()
        # Avoid the "black hole" around the center and points outside unit sphere
        if 1e-160 < length_squared <= 1:
            return p / math.sqrt(length_squared)

def random_in_unit_disk() -> vec3:
    """Generate a random point inside a unit disk in the XY plane."""
    while True:
        p = vec3(random_double_range(-1, 1), random_double_range(-1, 1), 0)
        if p.length_squared() < 1:
            return p

def random_on_hemisphere(normal: vec3) -> vec3:
    """Generate a random vector on the hemisphere around the normal."""
    on_unit_sphere = random_unit_vector()
    # If in the same hemisphere as the normal, keep it, otherwise flip it
    if dot(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    else:
        return -on_unit_sphere

def reflect(v: vec3, n: vec3) -> vec3:
    """Reflect vector v around normal n.
    
    Args:
        v: The vector to reflect
        n: The normal vector (must be unit vector)
        
    Returns:
        The reflected vector
    """
    return v - 2*dot(v,n)*n

def refract(uv: vec3, n: vec3, etai_over_etat: float) -> vec3:
    """Refract a vector according to Snell's law.
    
    Args:
        uv: The incident unit vector
        n: The normal vector
        etai_over_etat: The ratio of refractive indices (η/η')
        
    Returns:
        The refracted vector
    """
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -math.sqrt(abs(1.0 - r_out_perp.length_squared())) * n
    return r_out_perp + r_out_parallel 