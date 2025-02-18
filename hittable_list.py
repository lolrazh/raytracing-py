from __future__ import annotations
from typing import List
from hittable import hittable
from hit_record import hit_record
from ray import ray
from interval import interval

class hittable_list(hittable):
    """A list of hittable objects.
    
    Python Implementation Notes:
    - Unlike C++, we don't need shared_ptr as Python handles memory management
    - Python's garbage collection automatically handles reference counting
    - Python's List is used instead of C++'s std::vector
    - append() is equivalent to C++'s push_back()
    """
    
    def __init__(self, object: hittable = None):
        """Initialize an empty list or with a single object.
        
        In C++, this would use shared_ptr and make_shared.
        In Python, we can work with objects directly."""
        self.objects: List[hittable] = []  # Equivalent to std::vector in C++
        if object is not None:
            self.add(object)
    
    def clear(self) -> None:
        """Remove all objects from the list.
        Python will automatically handle cleanup of removed objects."""
        self.objects.clear()
    
    def add(self, object: hittable) -> None:
        """Add a hittable object to the list.
        
        Args:
            object: A hittable object (automatically reference counted by Python)
        
        Note: Equivalent to push_back() in C++."""
        self.objects.append(object)
    
    def hit(self, r: ray, ray_t: interval, rec: hit_record) -> bool:
        """Determine if ray hits any object in the list.
        
        Args:
            r: The ray to test
            ray_t: The interval to consider
            rec: hit_record to store the closest hit
            
        Returns:
            bool: True if any object was hit
            
        Note: Objects are automatically reference counted and cleaned up by Python."""
        temp_rec = hit_record()
        hit_anything = False
        closest_so_far = ray_t.max
        
        for object in self.objects:
            if object.hit(r, interval(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.front_face
                rec.mat = temp_rec.mat  # Don't forget to copy the material!
        
        return hit_anything 