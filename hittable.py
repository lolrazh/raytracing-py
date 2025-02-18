from __future__ import annotations
from abc import ABC, abstractmethod
from ray import ray
from interval import interval
from hit_record import hit_record

class hittable(ABC):
    """Abstract base class for anything that a ray might hit."""
    
    @abstractmethod
    def hit(self, r: ray, ray_t: interval, rec: hit_record) -> bool:
        """Test if ray hits this object within the given t interval."""
        pass 