from itertools import product

import numpy as np
outlier_ratio = np.arange(0.0, 1.0, 0.1)
maximum_distance_ratio = np.arange(0.80, 1.0, 0.02)

# Permutation Lookup Table
# PERM_LUT = list(product(outlier_ratio, maximum_distance_ratio))
PERM_LUT = [
    (0.5, 0.9),
    (0.5, 0.84),
    (0.5, 0.86),
    (0.5, 0.88),
    (0.49, 0.9),
    (0.49, 0.84),
    (0.49, 0.86),
    (0.49, 0.88),
    (0.48, 0.9),
    (0.48, 0.84),
    (0.48, 0.86),
    (0.48, 0.88),
    (0.47, 0.9),
    (0.47, 0.84),
    (0.47, 0.86),
    (0.47, 0.88),
    (0.46, 0.9),
    (0.46, 0.84),
    (0.46, 0.85),
    (0.46, 0.84),
    (0.45, 0.84),
    (0.44, 0.82),
    (0.45, 0.82),
    (0.46, 0.82),
    (0.47, 0.82),
]