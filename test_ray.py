from ray import ray
from vec3 import point3, vec3

# Create a ray
origin = point3(1, 2, 3)
direction = vec3(4, 5, 6)
r = ray(origin, direction)

# Test origin and direction
print(f"Origin: {r.origin()}")      # Should print: 1 2 3
print(f"Direction: {r.direction()}") # Should print: 4 5 6

# Test points along the ray
t1 = 0.5
p1 = r.at(t1)
print(f"Point at t={t1}: {p1}")     # Should print: 3 4.5 6

t2 = 1.0
p2 = r.at(t2)
print(f"Point at t={t2}: {p2}")     # Should print: 5 7 9

# Test default constructor
r2 = ray()
print(f"Default ray origin: {r2.origin()}")        # Should print: 0 0 0
print(f"Default ray direction: {r2.direction()}")  # Should print: 0 0 0 