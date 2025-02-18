from __future__ import annotations
from dataclasses import dataclass
from vec3 import vec3, point3, dot
from ray import ray
from typing import Optional

@dataclass
class hit_record:
    """Record of a ray hit with a surface.
    
    This class groups together all information about a ray-object intersection.
    The material pointer allows for polymorphic material behavior without
    needing to know the specific material type at intersection time.
    """
    p: point3 = point3(0, 0, 0)        # Point of intersection
    normal: vec3 = vec3(0, 0, 0)       # Surface normal at intersection
    mat: Optional['material'] = None    # Material of the hit surface (forward reference)
    t: float = 0.0                     # Ray parameter at intersection
    front_face: bool = True            # Whether the ray hit the front face

    def set_face_normal(self, r: ray, outward_normal: vec3) -> None:
        """Sets the hit record normal vector.
        
        Args:
            r: The incoming ray
            outward_normal: The outward normal vector (assumed to have unit length)
            
        The normal always points against the ray:
        - front_face is true if the ray is hitting the front face
        - normal points against the ray for front faces
        - normal points with the ray for back faces
        """
        self.front_face = dot(r.direction(), outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal 