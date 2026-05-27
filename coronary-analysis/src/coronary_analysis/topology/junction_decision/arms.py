from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from .pixels import iter_neighbors


Point = tuple[int, int]


@dataclass(frozen=True)
class Arm:
    start: Point
    path: np.ndarray
    short: bool = False

    @property
    def length(self) -> int:
        return int(len(self.path))


def extract_arms_for_group(
    skeleton: np.ndarray,
    group: dict[str, Any],
    all_junction_pixel_mask: np.ndarray,
    remove_radius: int,
    max_arm_steps: int,
    min_arm_len: int,
    keep_short_arms: bool = False,
) -> list[Arm]:
    removed_region = make_circular_mask(skeleton.shape, group["center"], remove_radius)
    return extract_arms_from_removed_region(
        skeleton=skeleton,
        removed_region_mask=removed_region,
        all_junction_pixel_mask=all_junction_pixel_mask,
        max_arm_steps=max_arm_steps,
        min_arm_len=min_arm_len,
        keep_short_arms=keep_short_arms,
    )


def extract_arms_from_removed_region(
    skeleton: np.ndarray,
    removed_region_mask: np.ndarray,
    all_junction_pixel_mask: np.ndarray,
    max_arm_steps: int,
    min_arm_len: int,
    keep_short_arms: bool = False,
) -> list[Arm]:
    starts, cut_skeleton = find_border_starts(skeleton, removed_region_mask)
    stop_mask = all_junction_pixel_mask.copy()
    stop_mask[removed_region_mask] = False
    return build_arms(
        cut_skeleton, starts, stop_mask, max_arm_steps, min_arm_len, keep_short_arms
    )


def make_circular_mask(
    shape: tuple[int, int], center: np.ndarray, radius: int
) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2) <= radius


def find_border_starts(
    skeleton: np.ndarray,
    removed_region_mask: np.ndarray,
) -> tuple[list[Point], np.ndarray]:
    cut_skeleton = skeleton.astype(bool).copy()
    cut_skeleton[removed_region_mask] = False
    border = border_skeleton_pixels(cut_skeleton, removed_region_mask)
    starts = nearest_points_per_component(border, removed_region_mask)
    return starts, cut_skeleton


def border_skeleton_pixels(
    cut_skeleton: np.ndarray, removed_region_mask: np.ndarray
) -> np.ndarray:
    dilated_region = ndi.binary_dilation(
        removed_region_mask,
        structure=ndi.generate_binary_structure(2, 2),
        iterations=1,
    )
    return dilated_region & ~removed_region_mask & cut_skeleton


def nearest_points_per_component(
    border: np.ndarray,
    removed_region_mask: np.ndarray,
) -> list[Point]:
    labelled = label(border, connectivity=2)
    removed_coords = np.argwhere(removed_region_mask)
    center = removed_coords.mean(axis=0) if len(removed_coords) else np.array([0, 0])
    starts: list[Point] = []
    for region in regionprops(labelled):
        starts.append(nearest_component_point(region.coords, center))
    return starts


def nearest_component_point(points: np.ndarray, center: np.ndarray) -> Point:
    distances = np.linalg.norm(points - center[None, :], axis=1)
    return tuple(map(int, points[np.argmin(distances)]))


def build_arms(
    skeleton: np.ndarray,
    starts: list[Point],
    stop_mask: np.ndarray,
    max_arm_steps: int,
    min_arm_len: int,
    keep_short_arms: bool,
) -> list[Arm]:
    arms: list[Arm] = []
    for start in starts:
        path = trace_arm(skeleton, start, stop_mask, max_arm_steps)
        arm = build_arm(start, path, min_arm_len, keep_short_arms)
        if arm is not None:
            arms.append(arm)
    return arms


def build_arm(
    start: Point,
    path: np.ndarray,
    min_arm_len: int,
    keep_short_arms: bool,
) -> Arm | None:
    if len(path) >= min_arm_len:
        return Arm(start=start, path=path, short=False)

    if keep_short_arms and len(path) >= 2:
        return Arm(start=start, path=path, short=True)

    return None


def trace_arm(
    skeleton: np.ndarray,
    start: Point,
    stop_junction_mask: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    path = [start]
    previous = None
    current = start
    for _ in range(max_steps):
        candidates = next_arm_candidates(
            skeleton, current, previous, stop_junction_mask
        )
        if not candidates:
            break
        next_point = choose_next_arm_point(candidates, current, previous)
        previous = current
        current = next_point
        path.append(current)
    return np.asarray(path, dtype=int)


def next_arm_candidates(
    skeleton: np.ndarray,
    current: Point,
    previous: Point | None,
    stop_junction_mask: np.ndarray,
) -> list[Point]:
    candidates: list[Point] = []
    for point in iter_neighbors(*current, skeleton.shape):
        if skeleton[point] == 0:
            continue
        if previous is not None and point == previous:
            continue
        if stop_junction_mask[point]:
            continue
        candidates.append(point)
    return candidates


def choose_next_arm_point(
    candidates: list[Point],
    current: Point,
    previous: Point | None,
) -> Point:
    if previous is None or len(candidates) == 1:
        return candidates[0]
    previous_vector = np.asarray(current) - np.asarray(previous)
    return min(candidates, key=lambda point: turn_cost(previous_vector, current, point))


def turn_cost(previous_vector: np.ndarray, current: Point, point: Point) -> float:
    next_vector = np.asarray(point) - np.asarray(current)
    return float(
        -np.dot(previous_vector, next_vector)
        / (np.linalg.norm(previous_vector) * np.linalg.norm(next_vector) + 1e-6)
    )
