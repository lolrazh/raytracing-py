from __future__ import annotations
from vec3 import vec3, point3, dot
from ray import ray
from hittable import hittable
from hit_record import hit_record
from interval import interval
from material import material
import math

class sphere(hittable):
    """A hittable sphere with a material.
    
    The sphere is defined by:
    - A center point
    - A radius
    - A material that determines how rays interact with the surface
    """
    
    def __init__(self, center: point3, radius: float, mat: material):
        """Initialize a sphere with a center, radius, and material.
        
        Args:
            center: The center point of the sphere
            radius: The radius of the sphere (must be non-negative)
            mat: The material that determines how rays interact with the sphere
        """
        self.center = center
        self.radius = max(0, radius)  # Ensure non-negative radius
        self.mat = mat
    
    def hit(self, r: ray, ray_t: interval, rec: hit_record) -> bool:
        """Determine if ray hits this sphere.
        
        Args:
            r: The ray to test
            ray_t: The interval of valid hit distances
            rec: The hit record to fill with intersection information
            
        Returns:
            True if the ray hits the sphere, False otherwise
            
        If there is a hit, the hit record will contain:
        - The point of intersection
        - The normal at the intersection point
        - The material of the sphere
        - The t parameter of the intersection
        - Whether the hit was on the front face
        """
        oc = self.center - r.origin()
        a = r.direction().length_squared()
        h = dot(r.direction(), oc)
        c = oc.length_squared() - self.radius*self.radius
        
        discriminant = h*h - a*c
        if discriminant < 0:
            return False
            
        sqrtd = math.sqrt(discriminant)
        
        # Find the nearest root that lies in the acceptable range
        root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return False
        
        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(r, outward_normal)
        rec.mat = self.mat  # Set the material in the hit record
        
        return True 