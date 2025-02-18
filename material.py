from __future__ import annotations
from abc import ABC, abstractmethod
from vec3 import vec3, color, random_unit_vector, unit_vector, reflect, dot, refract
from ray import ray
from hit_record import hit_record
from rtweekend import random_double
import math

class material(ABC):
    """Abstract base class for materials.
    
    A material specifies how rays scatter off a surface:
    - Whether the ray is absorbed or scattered
    - How the ray is scattered
    - How much the ray should be attenuated
    """
    
    @abstractmethod
    def scatter(self, r_in: ray, rec: hit_record) -> tuple[bool, color, ray]:
        """Scatter a ray off a surface.
        
        Args:
            r_in: The incoming ray
            rec: The hit record containing intersection information
            
        Returns:
            A tuple containing:
            - bool: Whether the ray was scattered
            - color: The attenuation of the ray
            - ray: The scattered ray (if any)
        """
        return False, color(0,0,0), ray()

class lambertian(material):
    """A Lambertian (diffuse) material.
    
    Properties:
    - Always scatters rays
    - Attenuation is determined by the albedo
    - Scattering direction is randomized
    """
    
    def __init__(self, albedo: color):
        """Initialize the material with an albedo.
        
        Args:
            albedo: The color that determines how much light is reflected
        """
        self.albedo = albedo
    
    def scatter(self, r_in: ray, rec: hit_record) -> tuple[bool, color, ray]:
        """Scatter a ray off the Lambertian surface.
        
        The scattered direction is the normal plus a random unit vector.
        If this random direction is zero, we use the normal instead.
        
        Args:
            r_in: The incoming ray (unused in Lambertian scattering)
            rec: The hit record containing the normal and intersection point
            
        Returns:
            - True (always scatters)
            - The albedo as attenuation
            - The scattered ray
        """
        scatter_direction = rec.normal + random_unit_vector()
        
        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = rec.normal
            
        scattered = ray(rec.p, scatter_direction)
        return True, self.albedo, scattered

class metal(material):
    """A metal (reflective) material.
    
    Properties:
    - Always reflects rays according to the reflection law
    - Attenuation is determined by the albedo
    - Reflection can be fuzzy based on the fuzz parameter
    - Absorbs rays that scatter below the surface
    """
    
    def __init__(self, albedo: color, fuzz: float = 0.0):
        """Initialize the material with an albedo and fuzziness.
        
        Args:
            albedo: The color that determines how much light is reflected
            fuzz: Controls the fuzziness of reflections (0=perfect mirror, 1=maximum fuzz)
        """
        self.albedo = albedo
        # Clamp fuzz to [0,1]
        self.fuzz = min(fuzz, 1.0) if fuzz > 0 else 0.0
    
    def scatter(self, r_in: ray, rec: hit_record) -> tuple[bool, color, ray]:
        """Scatter a ray off the metal surface.
        
        The scattered direction is the perfect reflection of the incident ray
        around the surface normal, plus some random perturbation based on fuzziness.
        Rays that would scatter below the surface are absorbed.
        
        Args:
            r_in: The incoming ray
            rec: The hit record containing the normal and intersection point
            
        Returns:
            - True if ray is scattered (not absorbed), False otherwise
            - The albedo as attenuation
            - The scattered ray
        """
        # Get perfect reflection direction and normalize it
        reflected = unit_vector(reflect(unit_vector(r_in.direction()), rec.normal))
        
        # Add fuzzy perturbation
        scattered_direction = reflected + self.fuzz * random_unit_vector()
        scattered = ray(rec.p, scattered_direction)
        
        # Return true only if the ray scatters above the surface
        return dot(scattered.direction(), rec.normal) > 0, self.albedo, scattered 

class dielectric(material):
    """A dielectric (glass-like) material.
    
    Properties:
    - Refracts rays according to Snell's law when possible
    - Reflects rays based on Schlick's approximation and total internal reflection
    - Perfect transmission (no attenuation)
    - Refractive index determines how much the ray bends
    """
    
    def __init__(self, refraction_index: float):
        """Initialize the material with a refractive index.
        
        Args:
            refraction_index: The refractive index (η) of the material
                            (1.0 for vacuum, 1.33 for water, 1.5 for glass)
        """
        self.refraction_index = refraction_index
    
    @staticmethod
    def reflectance(cosine: float, ref_idx: float) -> float:
        """Calculate reflectance using Schlick's approximation.
        
        This approximation gives us realistic angle-dependent reflectivity.
        
        Args:
            cosine: Cosine of the angle between ray and normal
            ref_idx: Ratio of refractive indices
            
        Returns:
            The reflectance (probability of reflection)
        """
        # Use Schlick's approximation for reflectance
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow((1 - cosine), 5)
    
    def scatter(self, r_in: ray, rec: hit_record) -> tuple[bool, color, ray]:
        """Scatter a ray through the dielectric material.
        
        The ray is either reflected or refracted based on:
        1. Total internal reflection (when angle too shallow)
        2. Schlick approximation (probability of reflection varies with angle)
        
        Args:
            r_in: The incoming ray
            rec: The hit record containing the normal and intersection point
            
        Returns:
            - True (always scatters)
            - White color (no attenuation)
            - The scattered ray
        """
        attenuation = color(1.0, 1.0, 1.0)  # Glass doesn't absorb any light
        
        # Calculate the ratio of refractive indices
        # If entering the material: air(1.0)/glass(η)
        # If leaving the material: glass(η)/air(1.0)
        refraction_ratio = (1.0 / self.refraction_index 
                          if rec.front_face else self.refraction_index)
        
        unit_direction = unit_vector(r_in.direction())
        
        # Calculate cos(θ) and sin(θ)
        cos_theta = min(dot(-unit_direction, rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        # Check for total internal reflection
        cannot_refract = refraction_ratio * sin_theta > 1.0
        
        # Choose reflection or refraction
        if cannot_refract or self.reflectance(cos_theta, refraction_ratio) > random_double():
            # Must reflect (total internal reflection or Schlick probability)
            direction = reflect(unit_direction, rec.normal)
        else:
            # Can refract
            direction = refract(unit_direction, rec.normal, refraction_ratio)
        
        scattered = ray(rec.p, direction)
        return True, attenuation, scattered 