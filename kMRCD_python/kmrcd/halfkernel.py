import numpy as np

def halfkernel(N1, N2, minx, r1, r2, noise, ratio):
    # Generate random angles
    phi1 = np.random.rand(N1) * np.pi
    phi2 = np.random.rand(N2) * np.pi

    # Calculate inner data
    inner_x1 = minx + r1 * np.sin(phi1) - 0.5 * noise + noise * np.random.rand(N1)
    inner_x2 = r1 * ratio * np.cos(phi1) - 0.5 * noise + noise * np.random.rand(N1)
    inner = np.column_stack((inner_x1, inner_x2, np.ones(N1)))

    # Calculate outer data
    outer_x1 = minx + r2 * np.sin(phi2) - 0.5 * noise + noise * np.random.rand(N2)
    outer_x2 = r2 * ratio * np.cos(phi2) - 0.5 * noise + noise * np.random.rand(N2)
    outer = np.column_stack((outer_x1, outer_x2, np.zeros(N2)))

    # Combine inner and outer data
    data = np.vstack((inner, outer))

    return data