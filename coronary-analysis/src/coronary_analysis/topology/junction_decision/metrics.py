from __future__ import annotations

from typing import Any

import numpy as np


def local_thickness_score(distance_map: np.ndarray | None, center: np.ndarray, radius: int) -> float:
    values = sample_disk_values(distance_map, center, radius)
    return float(np.mean(values)) if values else 0.0


def max_thickness_score(distance_map: np.ndarray | None, center: np.ndarray, radius: int) -> float:
    values = sample_disk_values(distance_map, center, radius)
    return float(np.max(values)) if values else 0.0


def sample_disk_values(distance_map: np.ndarray | None, center: np.ndarray, radius: int) -> list[float]:
    if distance_map is None:
        return []
    cy, cx = int(round(float(center[0]))), int(round(float(center[1])))
    values: list[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                append_distance_value(values, distance_map, cy + dy, cx + dx)
    return values


def append_distance_value(values: list[float], distance_map: np.ndarray, row: int, col: int) -> None:
    if 0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]:
        values.append(float(distance_map[row, col]))


def continuation_cost(
    image_gray: np.ndarray,
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    center: np.ndarray,
    angle_weight: float = 1.0,
    curvature_weight: float = 0.30,
    intensity_weight: float = 0.35,
) -> float:
    angle_cost = opposite_direction_cost(arm_a, arm_b, center)
    curvature_cost = mean_local_curvature(arm_a, arm_b)
    intensity_cost = intensity_profile_similarity(image_gray, arm_a, arm_b)
    return float(angle_weight * angle_cost + curvature_weight * curvature_cost + intensity_weight * intensity_cost)


def opposite_direction_cost(arm_a: dict[str, Any], arm_b: dict[str, Any], center: np.ndarray) -> float:
    vec_a = arm_direction_from_center(arm_a, center)
    vec_b = arm_direction_from_center(arm_b, center)
    dot = np.clip(np.dot(vec_a, vec_b), -1, 1)
    return float((dot + 1.0) / 2.0)


def arm_direction_from_center(arm: dict[str, Any], center: np.ndarray, lookahead: int = 12) -> np.ndarray:
    path = arm["path"]
    if len(path) < 2:
        return np.array([0.0, 0.0])
    index = min(lookahead, len(path) - 1)
    return normalize_vec(np.asarray(path[index], dtype=float) - np.asarray(center, dtype=float))


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector * 0
    return vector / norm


def mean_local_curvature(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> float:
    return (arm_local_curvature(arm_a) + arm_local_curvature(arm_b)) / 2.0


def arm_local_curvature(arm: dict[str, Any], n_points: int = 20) -> float:
    path = arm["path"][:n_points]
    if len(path) < 4:
        return 1.0
    directions = [normalize_vec(path[index] - path[index - 1]) for index in range(1, len(path))]
    return mean_direction_change(directions)


def mean_direction_change(directions: list[np.ndarray]) -> float:
    total = 0.0
    count = 0
    for index in range(1, len(directions)):
        dot = np.clip(np.dot(directions[index - 1], directions[index]), -1, 1)
        total += np.arccos(dot)
        count += 1
    return float(total / max(count, 1))


def intensity_profile_similarity(
    image_gray: np.ndarray,
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    n_points: int = 25,
) -> float:
    intensity_a = sample_intensity_along_arm(image_gray, arm_a, n_points)
    intensity_b = sample_intensity_along_arm(image_gray, arm_b, n_points)
    common_len = min(len(intensity_a), len(intensity_b))
    if common_len < 3:
        return 1.0
    return normalized_profile_distance(intensity_a[:common_len], intensity_b[:common_len])


def sample_intensity_along_arm(image_gray: np.ndarray, arm: dict[str, Any], n_points: int) -> np.ndarray:
    values: list[float] = []
    for row, col in arm["path"][:n_points]:
        row = int(np.clip(row, 0, image_gray.shape[0] - 1))
        col = int(np.clip(col, 0, image_gray.shape[1] - 1))
        values.append(float(image_gray[row, col]))
    return np.asarray(values, dtype=float) if values else np.array([0.0])


def normalized_profile_distance(intensity_a: np.ndarray, intensity_b: np.ndarray) -> float:
    norm_a = normalize_profile(intensity_a)
    norm_b = normalize_profile(intensity_b)
    correlation = np.mean(norm_a * norm_b)
    return float(1.0 - correlation)


def normalize_profile(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-6)
