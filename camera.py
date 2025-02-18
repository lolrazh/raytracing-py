from __future__ import annotations
import sys
from vec3 import vec3, point3, color, unit_vector, random_unit_vector, cross, random_in_unit_disk
from ray import ray
from hittable import hittable, hit_record
from interval import interval
from rtweekend import INFINITY, random_double, degrees_to_radians
from color import write_color
from material import lambertian, metal, dielectric
from sphere import sphere
import math
import numpy as np
import numba
from numba import cuda, float32, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# Define structured types for GPU data
sphere_type = np.dtype([
    ('center_x', np.float32),
    ('center_y', np.float32),
    ('center_z', np.float32),
    ('radius', np.float32),
])

material_type = np.dtype([
    ('type', np.int32),
    ('albedo_x', np.float32),
    ('albedo_y', np.float32),
    ('albedo_z', np.float32),
    ('fuzz', np.float32),
    ('ref_idx', np.float32),
])

camera_data_type = np.dtype([
    ('origin_x', np.float32),
    ('origin_y', np.float32),
    ('origin_z', np.float32),
    ('lower_left_corner_x', np.float32),
    ('lower_left_corner_y', np.float32),
    ('lower_left_corner_z', np.float32),
    ('horizontal_x', np.float32),
    ('horizontal_y', np.float32),
    ('horizontal_z', np.float32),
    ('vertical_x', np.float32),
    ('vertical_y', np.float32),
    ('vertical_z', np.float32),
    ('width', np.int32),
    ('height', np.int32),
])

# CUDA helper functions for vector operations
@cuda.jit(device=True)
def normalize(v):
    """Normalize a vector."""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length > 0:
        inv_length = float32(1.0 / length)
        return (float32(v[0] * inv_length), float32(v[1] * inv_length), float32(v[2] * inv_length))
    return (float32(v[0]), float32(v[1]), float32(v[2]))

@cuda.jit(device=True)
def dot_cuda(a, b):
    """Compute dot product of two vectors."""
    return float32(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

@cuda.jit(device=True)
def add(a, b):
    """Add two vectors."""
    return (float32(a[0] + b[0]), float32(a[1] + b[1]), float32(a[2] + b[2]))

@cuda.jit(device=True)
def subtract(a, b):
    """Subtract two vectors."""
    return (float32(a[0] - b[0]), float32(a[1] - b[1]), float32(a[2] - b[2]))

@cuda.jit(device=True)
def multiply_scalar(v, t):
    """Multiply vector by scalar, handling each component separately."""
    x = float32(v[0]) * float32(t)
    y = float32(v[1]) * float32(t)
    z = float32(v[2]) * float32(t)
    return (float32(x), float32(y), float32(z))

@cuda.jit(device=True)
def multiply_vectors(a, b):
    """Multiply two vectors component by component."""
    x = float32(a[0]) * float32(b[0])
    y = float32(a[1]) * float32(b[1])
    z = float32(a[2]) * float32(b[2])
    return (float32(x), float32(y), float32(z))

@cuda.jit(device=True)
def near_zero(v):
    """Check if vector is near zero in all dimensions."""
    s = 1e-8
    return abs(v[0]) < s and abs(v[1]) < s and abs(v[2]) < s

@cuda.jit(device=True)
def random_float_cuda(thread_id, rng_states):
    """Generate a random float between 0 and 1 using CUDA."""
    return xoroshiro128p_uniform_float32(rng_states, thread_id)

@cuda.jit(device=True)
def random_in_unit_disk_cuda(thread_id, rng_states):
    """Generate a random point inside a unit disk in the XY plane using CUDA."""
    while True:
        x = 2.0 * random_float_cuda(thread_id, rng_states) - 1.0
        y = 2.0 * random_float_cuda(thread_id, rng_states) - 1.0
        p = (x, y, 0.0)
        if dot_cuda(p, p) < 1.0:
            return p

@cuda.jit(device=True)
def random_in_unit_sphere_cuda(thread_id, rng_states):
    """Generate a random point inside a unit sphere using CUDA."""
    while True:
        x = 2.0 * random_float_cuda(thread_id, rng_states) - 1.0
        y = 2.0 * random_float_cuda(thread_id, rng_states) - 1.0
        z = 2.0 * random_float_cuda(thread_id, rng_states) - 1.0
        p = (x, y, z)
        if dot_cuda(p, p) < 1.0:
            return p

@cuda.jit(device=True)
def random_unit_vector_cuda(thread_id, rng_states):
    """Generate a random unit vector using CUDA random number generator."""
    return normalize(random_in_unit_sphere_cuda(thread_id, rng_states))

@cuda.jit(device=True)
def random_on_hemisphere_cuda(normal, thread_id, rng_states):
    """Generate a random vector on the hemisphere around a normal using CUDA."""
    on_unit_sphere = random_unit_vector_cuda(thread_id, rng_states)
    if dot_cuda(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    return (-on_unit_sphere[0], -on_unit_sphere[1], -on_unit_sphere[2])

@cuda.jit(device=True)
def reflect_cuda(v, n):
    """Reflect a vector around a normal."""
    dot_prod = dot_cuda(v, n)
    return (v[0] - 2 * dot_prod * n[0],
            v[1] - 2 * dot_prod * n[1],
            v[2] - 2 * dot_prod * n[2])

@cuda.jit(device=True)
def refract_cuda(uv, n, etai_over_etat):
    """Refract a vector according to Snell's law."""
    cos_theta = min(-dot_cuda(uv, n), 1.0)
    r_out_perp_x = etai_over_etat * (uv[0] + cos_theta * n[0])
    r_out_perp_y = etai_over_etat * (uv[1] + cos_theta * n[1])
    r_out_perp_z = etai_over_etat * (uv[2] + cos_theta * n[2])
    r_out_perp = (r_out_perp_x, r_out_perp_y, r_out_perp_z)
    
    r_out_parallel_factor = -math.sqrt(abs(1.0 - dot_cuda(r_out_perp, r_out_perp)))
    return (r_out_perp[0] + r_out_parallel_factor * n[0],
            r_out_perp[1] + r_out_parallel_factor * n[1],
            r_out_perp[2] + r_out_parallel_factor * n[2])

@cuda.jit(device=True)
def schlick_cuda(cosine, ref_idx):
    """Compute Schlick's approximation for Fresnel reflectance."""
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * math.pow((1 - cosine), 5)

@cuda.jit(device=True)
def cuda_ray_color(ray_origin, ray_direction, world_spheres, world_materials, max_depth, thread_id, rng_states):
    """CUDA device function to compute ray color."""
    # Initialize accumulated color and current ray
    attenuation = (float32(1.0), float32(1.0), float32(1.0))
    current_origin = ray_origin
    current_direction = ray_direction
    depth = max_depth
    
    while depth > 0:
        # Ray-sphere intersection test
        closest_t = float32(INFINITY)
        hit_anything = False
        hit_sphere_idx = -1
        
        # Test intersection with each sphere
        for i in range(world_spheres.size):
            center = (float32(world_spheres[i]['center_x']),
                     float32(world_spheres[i]['center_y']),
                     float32(world_spheres[i]['center_z']))
            radius = float32(world_spheres[i]['radius'])
            
            oc = subtract(current_origin, center)
            a = dot_cuda(current_direction, current_direction)
            half_b = dot_cuda(oc, current_direction)
            c = dot_cuda(oc, oc) - radius * radius
            
            discriminant = half_b * half_b - a * c
            
            if discriminant > 0:
                root = float32((-half_b - math.sqrt(discriminant)) / a)
                if root < closest_t and root > float32(0.001):
                    closest_t = root
                    hit_anything = True
                    hit_sphere_idx = i
        
        if not hit_anything:
            # Background color: blue-to-white gradient
            unit_direction = normalize(current_direction)
            a = float32(0.5) * (unit_direction[1] + float32(1.0))
            background = (
                float32((1.0 - a) + a * 0.5),  # Blend from white (1,1,1) to light blue (0.5,0.7,1.0)
                float32((1.0 - a) + a * 0.7),
                float32((1.0 - a) + a * 1.0)
            )
            return multiply_vectors(attenuation, background)
        
        # Compute hit point and normal
        hit_point = add(current_origin, multiply_scalar(current_direction, closest_t))
        center = (float32(world_spheres[hit_sphere_idx]['center_x']),
                 float32(world_spheres[hit_sphere_idx]['center_y']),
                 float32(world_spheres[hit_sphere_idx]['center_z']))
        outward_normal = normalize(subtract(hit_point, center))
        
        # Material properties
        material = world_materials[hit_sphere_idx]
        material_type = material['type']
        
        if material_type == 0:  # Lambertian
            scatter_direction = add(outward_normal, random_unit_vector_cuda(thread_id, rng_states))
            if near_zero(scatter_direction):
                scatter_direction = outward_normal
            
            mat_attenuation = (float32(material['albedo_x']), 
                             float32(material['albedo_y']), 
                             float32(material['albedo_z']))
            attenuation = multiply_vectors(attenuation, mat_attenuation)
            current_origin = hit_point
            current_direction = scatter_direction
            
        elif material_type == 1:  # Metal
            reflected = reflect_cuda(normalize(current_direction), outward_normal)
            fuzz = float32(material['fuzz'])
            
            if fuzz > 0:
                random_vec = random_unit_vector_cuda(thread_id, rng_states)
                scattered = add(reflected, multiply_scalar(random_vec, fuzz))
            else:
                scattered = reflected
            
            if dot_cuda(scattered, outward_normal) > 0:
                mat_attenuation = (float32(material['albedo_x']), 
                                 float32(material['albedo_y']), 
                                 float32(material['albedo_z']))
                attenuation = multiply_vectors(attenuation, mat_attenuation)
                current_origin = hit_point
                current_direction = scattered
            else:
                return (float32(0.0), float32(0.0), float32(0.0))
            
        elif material_type == 2:  # Dielectric
            mat_attenuation = (float32(1.0), float32(1.0), float32(1.0))
            refraction_ratio = (float32(1.0 / material['ref_idx']) 
                              if dot_cuda(current_direction, outward_normal) < 0 
                              else float32(material['ref_idx']))
            
            unit_direction = normalize(current_direction)
            cos_theta = min(-dot_cuda(unit_direction, outward_normal), float32(1.0))
            sin_theta = math.sqrt(float32(1.0 - cos_theta * cos_theta))
            
            cannot_refract = refraction_ratio * sin_theta > float32(1.0)
            rng = random_float_cuda(thread_id, rng_states)
            
            if cannot_refract or schlick_cuda(cos_theta, refraction_ratio) > rng:
                scattered = reflect_cuda(unit_direction, outward_normal)
            else:
                scattered = refract_cuda(unit_direction, outward_normal, refraction_ratio)
            
            attenuation = multiply_vectors(attenuation, mat_attenuation)
            current_origin = hit_point
            current_direction = scattered
        
        else:
            return (float32(1.0), float32(0.0), float32(0.0))  # Default red for unhandled materials
        
        depth -= 1
    
    # If we've exceeded max depth, return black
    return (float32(0.0), float32(0.0), float32(0.0))

@cuda.jit
def render_kernel(image, camera_data, world_spheres, world_materials, samples_per_pixel, max_depth, rng_states):
    """CUDA kernel for parallel rendering."""
    # Get the 2D thread position within the grid
    x, y = cuda.grid(2)
    
    # Calculate unique thread ID
    block_size = cuda.blockDim.x * cuda.blockDim.y
    block_id = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
    thread_id = block_id * block_size + cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    
    if x >= camera_data['width'] or y >= camera_data['height']:
        return
    
    pixel_color = (float32(0.0), float32(0.0), float32(0.0))
    
    # Take multiple samples for each pixel
    for s in range(samples_per_pixel):
        # Generate ray for this pixel
        u = float32((x + random_float_cuda(thread_id, rng_states)) / float32(camera_data['width'] - 1))
        v = float32((y + random_float_cuda(thread_id, rng_states)) / float32(camera_data['height'] - 1))
        
        ray_direction = (
            float32(camera_data['lower_left_corner_x'] + u * camera_data['horizontal_x'] + v * camera_data['vertical_x'] - camera_data['origin_x']),
            float32(camera_data['lower_left_corner_y'] + u * camera_data['horizontal_y'] + v * camera_data['vertical_y'] - camera_data['origin_y']),
            float32(camera_data['lower_left_corner_z'] + u * camera_data['horizontal_z'] + v * camera_data['vertical_z'] - camera_data['origin_z'])
        )
        
        ray_origin = (float32(camera_data['origin_x']), 
                     float32(camera_data['origin_y']), 
                     float32(camera_data['origin_z']))
        
        # Get color for this sample
        color = cuda_ray_color(ray_origin, ray_direction, world_spheres, world_materials, max_depth, thread_id, rng_states)
        pixel_color = add(pixel_color, color)
    
    # Average and gamma correct
    scale = float32(1.0 / samples_per_pixel)
    r = float32(math.sqrt(scale * pixel_color[0]))
    g = float32(math.sqrt(scale * pixel_color[1]))
    b = float32(math.sqrt(scale * pixel_color[2]))
    
    # Store in image array
    idx = (y * camera_data['width'] + x) * 3
    image[idx] = min(255, max(0, int(256 * r)))
    image[idx + 1] = min(255, max(0, int(256 * g)))
    image[idx + 2] = min(255, max(0, int(256 * b)))

class camera:
    """A camera for rendering a scene."""
    
    def __init__(self):
        """Initialize default camera parameters."""
        # Public parameters - can be modified before rendering
        self.aspect_ratio: float = 1.0     # Ratio of image width over height
        self.image_width: int = 100        # Rendered image width in pixel count
        self.samples_per_pixel: int = 10   # Count of random samples for each pixel
        self.max_depth: int = 10           # Maximum number of ray bounces
        self.vfov: float = 90              # Vertical field of view (in degrees)
        
        # Camera position and orientation
        self.lookfrom = point3(0, 0, 0)    # Point camera is looking from
        self.lookat = point3(0, 0, -1)     # Point camera is looking at
        self.vup = vec3(0, 1, 0)           # Camera-relative "up" direction
        
        # Defocus blur parameters
        self.defocus_angle: float = 0      # Variation angle of rays through each pixel
        self.focus_dist: float = 10        # Distance from camera to plane of perfect focus
        
        # Private members - initialized when render is called
        self._image_height: int = 0        # Rendered image height
        self._center = point3()            # Camera center
        self._pixel00_loc = point3()       # Location of pixel 0, 0
        self._pixel_delta_u = vec3()       # Offset to pixel to the right
        self._pixel_delta_v = vec3()       # Offset to pixel below
        self._u = vec3()                   # Camera frame basis vectors
        self._v = vec3()
        self._w = vec3()
        self._defocus_disk_u = vec3()      # Defocus disk basis vectors
        self._defocus_disk_v = vec3()
    
    def _initialize(self) -> None:
        """Initialize camera geometry."""
        # Calculate image height and ensure it's at least 1
        self._image_height = int(self.image_width / self.aspect_ratio)
        self._image_height = max(1, self._image_height)
        
        self._center = self.lookfrom
        
        # Determine viewport dimensions
        theta = degrees_to_radians(self.vfov)
        h = math.tan(theta/2)
        viewport_height = 2 * h * self.focus_dist
        viewport_width = viewport_height * (float(self.image_width) / self._image_height)
        
        # Calculate the u,v,w unit basis vectors for the camera coordinate frame
        self._w = unit_vector(self.lookfrom - self.lookat)
        self._u = unit_vector(cross(self.vup, self._w))
        self._v = cross(self._w, self._u)
        
        # Calculate the vectors across the horizontal and down the vertical viewport edges
        viewport_u = viewport_width * self._u      # Vector across viewport horizontal edge
        viewport_v = viewport_height * -self._v    # Vector down viewport vertical edge
        
        # Calculate the horizontal and vertical delta vectors from pixel to pixel
        self._pixel_delta_u = viewport_u / self.image_width
        self._pixel_delta_v = viewport_v / self._image_height
        
        # Calculate the location of the upper left pixel
        viewport_upper_left = (self._center - self.focus_dist * self._w 
                             - viewport_u/2 - viewport_v/2)
        self._pixel00_loc = viewport_upper_left + 0.5 * (self._pixel_delta_u + self._pixel_delta_v)
        
        # Calculate the camera defocus disk basis vectors
        defocus_radius = self.focus_dist * math.tan(degrees_to_radians(self.defocus_angle / 2))
        self._defocus_disk_u = self._u * defocus_radius
        self._defocus_disk_v = self._v * defocus_radius
    
    def _defocus_disk_sample(self) -> point3:
        """Returns a random point in the camera defocus disk."""
        p = random_in_unit_disk()
        return self._center + (p.x() * self._defocus_disk_u) + (p.y() * self._defocus_disk_v)
    
    def _sample_square(self) -> vec3:
        """Returns a random point in the square surrounding a pixel."""
        px = random_double() - 0.5
        py = random_double() - 0.5
        return vec3(px, py, 0)
    
    def _get_ray(self, i: int, j: int) -> ray:
        """Get a randomly sampled ray for pixel at i,j."""
        # Get random offset in pixel square
        offset = self._sample_square()
        
        # Calculate pixel sample position
        pixel_sample = (self._pixel00_loc 
                       + (i + offset.x()) * self._pixel_delta_u 
                       + (j + offset.y()) * self._pixel_delta_v)
        
        # Get ray origin (camera center or random point in defocus disk)
        ray_origin = (self._center if self.defocus_angle <= 0 
                     else self._defocus_disk_sample())
        ray_direction = pixel_sample - ray_origin
        
        return ray(ray_origin, ray_direction)
    
    def _ray_color(self, r: ray, depth: int, world: hittable) -> color:
        """Calculate the color for a ray."""
        # If we've exceeded the ray bounce limit, no more light is gathered
        if depth <= 0:
            return color(0, 0, 0)

        rec = hit_record()
        
        if world.hit(r, interval(0.001, INFINITY), rec):
            # Get scattered ray and attenuation from the material
            scattered_success, attenuation, scattered = rec.mat.scatter(r, rec)
            if scattered_success:
                return attenuation * self._ray_color(scattered, depth-1, world)
            return color(0, 0, 0)
        
        # Background: blue-to-white gradient
        unit_direction = unit_vector(r.direction())
        a = 0.5 * (unit_direction[1] + float32(1.0))
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)
    
    def render_gpu(self, world: hittable) -> None:
        """Render the scene using GPU acceleration."""
        self._initialize()
        
        # Prepare camera data for GPU
        camera_data = np.zeros(1, dtype=camera_data_type)
        camera_data['origin_x'] = self._center.x()
        camera_data['origin_y'] = self._center.y()
        camera_data['origin_z'] = self._center.z()
        camera_data['lower_left_corner_x'] = self._pixel00_loc.x()
        camera_data['lower_left_corner_y'] = self._pixel00_loc.y()
        camera_data['lower_left_corner_z'] = self._pixel00_loc.z()
        camera_data['horizontal_x'] = self._pixel_delta_u.x() * self.image_width
        camera_data['horizontal_y'] = self._pixel_delta_u.y() * self.image_width
        camera_data['horizontal_z'] = self._pixel_delta_u.z() * self.image_width
        camera_data['vertical_x'] = self._pixel_delta_v.x() * self._image_height
        camera_data['vertical_y'] = self._pixel_delta_v.y() * self._image_height
        camera_data['vertical_z'] = self._pixel_delta_v.z() * self._image_height
        camera_data['width'] = self.image_width
        camera_data['height'] = self._image_height
        
        # Convert world data to GPU-friendly format
        world_spheres = []
        world_materials = []
        
        # Assuming world is a hittable_list containing spheres
        for obj in world.objects:
            if isinstance(obj, sphere):
                sphere_data = np.zeros(1, dtype=sphere_type)
                sphere_data['center_x'] = obj.center.x()
                sphere_data['center_y'] = obj.center.y()
                sphere_data['center_z'] = obj.center.z()
                sphere_data['radius'] = obj.radius
                world_spheres.append(sphere_data[0])
                
                material_data = np.zeros(1, dtype=material_type)
                if isinstance(obj.mat, lambertian):
                    material_data['type'] = 0
                    material_data['albedo_x'] = obj.mat.albedo.x()
                    material_data['albedo_y'] = obj.mat.albedo.y()
                    material_data['albedo_z'] = obj.mat.albedo.z()
                elif isinstance(obj.mat, metal):
                    material_data['type'] = 1
                    material_data['albedo_x'] = obj.mat.albedo.x()
                    material_data['albedo_y'] = obj.mat.albedo.y()
                    material_data['albedo_z'] = obj.mat.albedo.z()
                    material_data['fuzz'] = obj.mat.fuzz
                elif isinstance(obj.mat, dielectric):
                    material_data['type'] = 2
                    material_data['ref_idx'] = obj.mat.refraction_index
                else:
                    material_data['type'] = 0
                    material_data['albedo_x'] = 0.8
                    material_data['albedo_y'] = 0.8
                    material_data['albedo_z'] = 0.8
                world_materials.append(material_data[0])
        
        # Convert lists to numpy arrays
        world_spheres = np.array(world_spheres, dtype=sphere_type)
        world_materials = np.array(world_materials, dtype=material_type)
        
        # Prepare output image array
        image = np.zeros((self._image_height * self.image_width * 3), dtype=np.uint8)
        
        # Configure CUDA grid
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.image_width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self._image_height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Initialize random number generator states
        total_threads = blocks_per_grid[0] * blocks_per_grid[1] * threads_per_block[0] * threads_per_block[1]
        rng_states = create_xoroshiro128p_states(total_threads, seed=1)
        
        # Copy data to device
        d_image = cuda.to_device(image)
        d_camera_data = cuda.to_device(camera_data)
        d_world_spheres = cuda.to_device(world_spheres)
        d_world_materials = cuda.to_device(world_materials)
        
        # Launch kernel
        render_kernel[blocks_per_grid, threads_per_block](
            d_image, d_camera_data[0], d_world_spheres, d_world_materials,
            self.samples_per_pixel, self.max_depth, rng_states)
        
        # Copy result back to host
        image = d_image.copy_to_host()
        
        # Write the PPM image
        with open('image.ppm', 'w') as img_file:
            img_file.write(f"P3\n{self.image_width} {self._image_height}\n255\n")
            for i in range(0, len(image), 3):
                img_file.write(f"{image[i]} {image[i+1]} {image[i+2]}\n")
        
        print("\rDone.                 ", file=sys.stderr)

    def render(self, world: hittable) -> None:
        """Render the scene, choosing between CPU and GPU implementation."""
        try:
            # Check if CUDA is available
            if cuda.is_available():
                print("Using GPU acceleration...", file=sys.stderr)
                self.render_gpu(world)
            else:
                print("CUDA not available, falling back to CPU...", file=sys.stderr)
                self.render_cpu(world)
        except Exception as e:
            print(f"GPU rendering failed: {e}, falling back to CPU...", file=sys.stderr)
            self.render_cpu(world)

    def render_cpu(self, world: hittable) -> None:
        """Original CPU-based render method."""
        # Move existing render method code here
        self._initialize()
        
        with open('image.ppm', 'w') as img_file:
            img_file.write(f"P3\n{self.image_width} {self._image_height}\n255\n")
            
            for j in range(self._image_height):
                print(f"\rScanlines remaining: {self._image_height - j}", 
                      end=' ', file=sys.stderr, flush=True)
                for i in range(self.image_width):
                    pixel_color = color(0, 0, 0)
                    for _ in range(self.samples_per_pixel):
                        r = self._get_ray(i, j)
                        pixel_color += self._ray_color(r, self.max_depth, world)
                    
                    write_color(img_file, pixel_color, self.samples_per_pixel)
            
            print("\rDone.                 ", file=sys.stderr) 