from vec3 import vec3
from interval import interval
import math

# Type aliases
color = vec3    # RGB color

def linear_to_gamma(linear_component: float) -> float:
    """Transform from linear to gamma space (gamma 2).
    
    Args:
        linear_component: Color component in linear space
        
    Returns:
        Color component in gamma space
    """
    return math.sqrt(linear_component) if linear_component > 0 else 0

def write_color(out, pixel_color: color, samples_per_pixel: int = 1) -> None:
    """Write the translated [0,255] value of each color component.
    
    For multiple samples, we need to:
    1. Divide the color by the number of samples
    2. Transform from linear to gamma space (gamma 2)
    3. Clamp the resulting color components to [0,1]
    4. Transform to [0,255] for output
    """
    # Divide the color by the number of samples
    scale = 1.0 / samples_per_pixel
    r = pixel_color.x() * scale
    g = pixel_color.y() * scale
    b = pixel_color.z() * scale

    # Transform from linear to gamma space
    r = linear_to_gamma(r)
    g = linear_to_gamma(g)
    b = linear_to_gamma(b)

    # Clamp the color components to [0,1]
    intensity = interval(0.000, 0.999)
    r = intensity.clamp(r)
    g = intensity.clamp(g)
    b = intensity.clamp(b)

    # Write the translated [0,255] value of each color component
    ir = int(256 * r)
    ig = int(256 * g)
    ib = int(256 * b)

    print(f"{ir} {ig} {ib}", file=out) 