import numpy as np
import struct


def create_vector_field(size_x, size_y, size_z):
    x, y, z = np.meshgrid(np.linspace(0, 2 * np.pi, size_x),
                          np.linspace(0, 2 * np.pi, size_y),
                          np.linspace(0, 2 * np.pi, size_z))

    u = np.sin(x) * np.cos(z)
    v = np.cos(x) * np.sin(y)
    w = np.sin(z) * np.cos(y)

    return u, v, w


def write_binary_fga_file(filename, u, v, w):
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'FGA\0')  # File signature
        f.write(struct.pack('<I', 1))  # Version number
        f.write(struct.pack('<III', u.shape[0], u.shape[1], u.shape[2]))  # Dimensions
        f.write(struct.pack('<fff', 1.0, 1.0, 1.0))  # Scale

        # Write vector data
        for k in range(u.shape[2]):
            for j in range(u.shape[1]):
                for i in range(u.shape[0]):
                    x = max(min(u[i, j, k], 1), -1)
                    y = max(min(v[i, j, k], 1), -1)
                    z = max(min(w[i, j, k], 1), -1)
                    f.write(struct.pack('<fff', x, y, z))


# Example usage
size_x, size_y, size_z = 32, 32, 32  # Adjust size as needed
u, v, w = create_vector_field(size_x, size_y, size_z)
write_binary_fga_file("vector_field.fga", u, v, w)