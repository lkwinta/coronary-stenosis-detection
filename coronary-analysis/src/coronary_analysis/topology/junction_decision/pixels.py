from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from coronary_analysis.topology.skeleton import DIRECTIONS, classify_skeleton_pixels


def in_bounds(row: int, col: int, shape: tuple[int, int]) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def iter_neighbors(row: int, col: int, shape: tuple[int, int]) -> Iterator[tuple[int, int]]:
    for d_row, d_col in DIRECTIONS:
        n_row = row + d_row
        n_col = col + d_col
        if in_bounds(n_row, n_col, shape):
            yield n_row, n_col


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    endpoints, _ = classify_skeleton_pixels(skeleton.astype(np.uint8))
    mask = np.zeros(skeleton.shape, dtype=bool)
    if len(endpoints):
        mask[endpoints[:, 0], endpoints[:, 1]] = True
    return mask


def junction_pixel_mask(skeleton: np.ndarray) -> np.ndarray:
    _, junctions = classify_skeleton_pixels(skeleton.astype(np.uint8))
    mask = np.zeros(skeleton.shape, dtype=bool)
    if len(junctions):
        mask[junctions[:, 0], junctions[:, 1]] = True
    return mask
