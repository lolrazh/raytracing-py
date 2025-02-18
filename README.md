# GPU-Accelerated Ray Tracer in Python

![image.png]

A GPU-accelerated implementation of Peter Shirley's "Ray Tracing in One Weekend" using Python, CUDA, and Numba.

## Features

- Full implementation of ray tracing with:
  - Multiple materials (Lambertian, Metal, Dielectric)
  - Reflection and refraction
  - Antialiasing
  - Depth of field
  - Gamma correction
- GPU acceleration using CUDA through Numba
- Support for complex scenes with multiple objects
- Configurable camera parameters
- Both CPU and GPU rendering paths with automatic fallback

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Required Python packages:
  ```
  numba==0.57.1  # Specific version required for CUDA compatibility
  numpy>=1.24,<1.25
  cupy-cuda12x  # Replace 12x with your CUDA version
  ```

## Installation

1. Ensure you have CUDA toolkit installed (compatible with your GPU)
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install numba==0.57.1 numpy cupy-cuda12x
   ```

## Usage

Basic usage:
```python
from camera import camera
from hittable_list import hittable_list
from sphere import sphere
from material import lambertian, metal, dielectric
from vec3 import color, point3

# Create a scene
world = hittable_list()

# Add objects with materials
ground_material = lambertian(color(0.5, 0.5, 0.5))
world.add(sphere(point3(0, -1000, 0), 1000, ground_material))

# Configure camera
cam = camera()
cam.aspect_ratio = 16.0 / 9.0
cam.image_width = 1200
cam.samples_per_pixel = 500
cam.max_depth = 50

# Render scene
cam.render(world)
```

## GPU Implementation Details

### Data Structures
The GPU implementation uses structured NumPy arrays for efficient data transfer and access:

```python
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
```

### Key Components

1. **CUDA Helper Functions**
   - Vector operations (normalize, dot product, add, subtract)
   - Random number generation
   - Material calculations (reflection, refraction, Fresnel)

2. **Ray Tracing Kernel**
   - Iterative ray bouncing (non-recursive)
   - Material-specific scatter functions
   - Efficient parallel computation

3. **Memory Management**
   - Structured arrays for geometry and materials
   - Efficient data transfer between CPU and GPU
   - Thread-specific random number generation

### Performance Considerations

- Use `float32` for all calculations
- Avoid recursion in CUDA functions
- Minimize memory transfers
- Efficient thread indexing
- Proper handling of parallel random number generation

## Troubleshooting

Common issues and solutions:

1. **CUDA Initialization Errors**
   - Ensure CUDA toolkit is properly installed
   - Check GPU driver version
   - Verify Numba and CuPy versions match CUDA version

2. **Type Errors**
   - Ensure all numeric values use `float32`
   - Check for proper type conversion in vector operations
   - Verify structure alignment in NumPy dtypes

3. **Performance Issues**
   - Adjust thread block size (default: 16x16)
   - Optimize samples per pixel vs. render time
   - Monitor GPU memory usage

## Implementation Notes

### Critical Considerations

1. **Type Consistency**
   - All vector operations must use `float32`
   - Explicit type conversion for numeric operations
   - Consistent structure padding in NumPy dtypes

2. **CUDA Limitations**
   - No recursion in device functions
   - No dynamic memory allocation
   - Limited Python features in CUDA code

3. **Random Number Generation**
   - Thread-specific RNG states
   - Proper seeding for reproducibility
   - Efficient parallel generation

### Best Practices

1. **Code Organization**
   - Separate CPU and GPU implementations
   - Clear error handling and fallback paths
   - Modular CUDA helper functions

2. **Memory Management**
   - Minimize host-device transfers
   - Use structured arrays for complex data
   - Proper cleanup of CUDA resources

3. **Performance Optimization**
   - Balance thread block size
   - Minimize divergent execution
   - Efficient memory access patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Peter Shirley for the original "Ray Tracing in One Weekend" book
- The Numba team for CUDA support in Python
- The ray tracing community for various optimizations and insights

## References

1. Shirley, Peter. "Ray Tracing in One Weekend"
2. Numba CUDA Documentation
3. NVIDIA CUDA Programming Guide 
