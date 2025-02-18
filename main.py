from vec3 import point3, color, vec3, random_double
from hittable_list import hittable_list
from sphere import sphere
from camera import camera
from material import lambertian, metal, dielectric
from rtweekend import random_double_range

# World
world = hittable_list()

# Ground material and sphere
material_ground = lambertian(color(0.5, 0.5, 0.5))
world.add(sphere(point3(0, -1000, 0), 1000, material_ground))

# Generate random small spheres
for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random_double()
        center = point3(a + 0.9*random_double(), 0.2, b + 0.9*random_double())

        # Only place sphere if it's far enough from the big spheres
        if (center - point3(4, 0.2, 0)).length() > 0.9:
            # Choose sphere material randomly
            if choose_mat < 0.8:
                # Diffuse material
                albedo = color.random() * color.random()
                sphere_material = lambertian(albedo)
                world.add(sphere(center, 0.2, sphere_material))
            elif choose_mat < 0.95:
                # Metal material
                albedo = color.random_range(0.5, 1)
                fuzz = random_double_range(0, 0.5)
                sphere_material = metal(albedo, fuzz)
                world.add(sphere(center, 0.2, sphere_material))
            else:
                # Glass material
                sphere_material = dielectric(1.5)
                world.add(sphere(center, 0.2, sphere_material))

# Add three large spheres
material1 = dielectric(1.5)
world.add(sphere(point3(0, 1, 0), 1.0, material1))

material2 = lambertian(color(0.4, 0.2, 0.1))
world.add(sphere(point3(-4, 1, 0), 1.0, material2))

material3 = metal(color(0.7, 0.6, 0.5), 0.0)
world.add(sphere(point3(4, 1, 0), 1.0, material3))

# Camera
cam = camera()

# Image settings
cam.aspect_ratio = 16.0 / 9.0
cam.image_width = 1200
cam.samples_per_pixel = 500
cam.max_depth = 50

# Camera position and orientation
cam.vfov = 20
cam.lookfrom = point3(13, 2, 3)
cam.lookat = point3(0, 0, 0)
cam.vup = vec3(0, 1, 0)

# Depth of field settings
cam.defocus_angle = 0.6
cam.focus_dist = 10.0

# Render
cam.render(world) 