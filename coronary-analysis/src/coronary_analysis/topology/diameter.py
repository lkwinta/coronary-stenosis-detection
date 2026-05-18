from scipy.ndimage import distance_transform_edt

import numpy as np


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(mask.astype(bool))


def estimate_branch_diameters(
    path: np.ndarray,
    distance_map: np.ndarray,
) -> np.ndarray:
    return np.array([2 * distance_map[int(r), int(c)] for r, c in path])
