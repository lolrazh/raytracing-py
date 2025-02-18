from vec3 import vec3, dot, cross, unit_vector

# Test vector creation and basic operations
v1 = vec3(1, 2, 3)
v2 = vec3(4, 5, 6)

# Test addition
v3 = v1 + v2
print(f"v1 + v2 = {v3}")  # Should print: 5 7 9

# Test scalar multiplication
v4 = v1 * 2
print(f"v1 * 2 = {v4}")  # Should print: 2 4 6

# Test dot product
d = dot(v1, v2)
print(f"dot(v1, v2) = {d}")  # Should print: 32

# Test cross product
c = cross(v1, v2)
print(f"cross(v1, v2) = {c}")  # Should print: -3 6 -3

# Test unit vector
u = unit_vector(v1)
print(f"unit_vector(v1) = {u}")  # Should print: 0.267261 0.534522 0.801784
print(f"length = {u.length()}")  # Should print: 1.0 