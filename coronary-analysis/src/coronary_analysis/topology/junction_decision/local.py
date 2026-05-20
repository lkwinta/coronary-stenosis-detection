from __future__ import annotations

from typing import Any

import numpy as np

from coronary_analysis.topology.skeleton import prune_skeleton

from .arms import extract_arms_for_group
from .model import JunctionDecisionConfig
from .pixels import junction_pixel_mask


def refine_arms_locally(
    skeleton: np.ndarray,
    group: dict[str, Any],
    config: JunctionDecisionConfig,
) -> list[dict[str, Any]]:
    y1, y2, x1, x2 = crop_bounds(group["center"], skeleton.shape, config.local_crop_size)
    local_skeleton = skeleton[y1:y2, x1:x2].astype(np.uint8)
    local_skeleton = prune_skeleton(local_skeleton, config.local_prune_min_branch_length).astype(bool)
    local_group = make_local_group(group, y1, x1)
    local_arms = extract_arms_for_group(
        skeleton=local_skeleton,
        group=local_group,
        all_junction_pixel_mask=junction_pixel_mask(local_skeleton),
        remove_radius=config.local_remove_radius,
        max_arm_steps=config.local_max_arm_steps,
        min_arm_len=config.local_min_arm_len,
        keep_short_arms=config.local_keep_short_arms,
    )
    return to_global_arms(local_arms, y1, x1)


def crop_bounds(center: np.ndarray, shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    cy, cx = center
    cy = int(round(float(cy)))
    cx = int(round(float(cx)))
    half = size // 2
    y1 = max(0, cy - half)
    y2 = min(shape[0], cy + half)
    x1 = max(0, cx - half)
    x2 = min(shape[1], cx + half)
    return y1, y2, x1, x2


def make_local_group(group: dict[str, Any], y_offset: int, x_offset: int) -> dict[str, Any]:
    local_center = np.asarray(group["center"], dtype=float) - np.array([y_offset, x_offset], dtype=float)
    return {
        "id": group["id"],
        "center": local_center,
        "pixels": np.array([[int(round(local_center[0])), int(round(local_center[1]))]]),
        "area": group.get("area", 1),
        "global_center": group["center"],
        "crop_offset": (y_offset, x_offset),
    }


def to_global_arms(
    local_arms: list[dict[str, Any]],
    y_offset: int,
    x_offset: int,
) -> list[dict[str, Any]]:
    global_arms: list[dict[str, Any]] = []
    for arm in local_arms:
        path = arm["path"].copy()
        path[:, 0] += y_offset
        path[:, 1] += x_offset
        global_arms.append({**arm, "path": path, "start": tuple(path[0]), "source": "local"})
    return global_arms
