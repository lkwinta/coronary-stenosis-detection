"""Junction decision logic moved out of the exploratory notebook.

This module contains the same decision flow as
``batch_two_separate_plots_certain_only.ipynb``:

* detect/group junction pixels on the skeleton,
* extract arms around every junction group,
* optionally re-skeletonize locally when the first pass returns ``not``,
* assign one of: ``certain``, ``false`` or ``not``.

The code is intentionally independent from plotting and dataset scanning so it can be
used from the main ``analyze.py`` pipeline after segmentation/topology steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Literal

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, medial_axis, remove_small_objects, skeletonize

JunctionLabel = Literal["certain", "false", "not"]

NEIGHBORS_8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass(frozen=True)
class JunctionDecisionConfig:
    # Skeleton / local clean-up defaults copied from the notebook.
    min_object_size: int = 30
    use_medial_axis: bool = True
    prune_min_arm_len: int = 4
    prune_iterations: int = 1

    # Junction grouping.
    junction_group_dilation: int = 3
    min_junction_area: int = 1

    # Junction analysis.
    remove_radius: int = 2
    max_arm_steps: int = 60
    min_arm_len: int = 4

    # Local fallback.
    enable_local_reskeleton: bool = True
    local_crop_size: int = 80
    local_use_medial_axis: bool = True
    local_remove_radius: int = 2
    local_max_arm_steps: int = 80
    local_min_arm_len: int = 2
    local_keep_short_arms: bool = True
    local_prune_min_arm_len: int = 2
    local_prune_iterations: int = 0

    # Decision thresholds.
    fake_mean_cost_thr: float = 0.25
    fake_max_cost_thr: float = 0.40
    allow_two_arm_fake: bool = True
    min_area_for_two_arm_fake: int = 6
    enable_thickness_fake: bool = True
    thickness_radius: int = 5
    thickness_fake_thr: float = 2.0
    thickness_area_thr: int = 4


@dataclass
class JunctionDecision:
    label: JunctionLabel
    reason: str
    center: np.ndarray
    group: dict[str, Any]
    arms: list[dict[str, Any]]
    n_arms: int
    best_pairing: dict[str, Any] | None
    thickness_mean: float
    thickness_max: float
    used_local_reskeleton: bool = False


@dataclass
class JunctionDecisionResult:
    junction_groups: list[dict[str, Any]]
    all_junction_pixel_mask: np.ndarray
    decisions: list[JunctionDecision] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"certain": 0, "false": 0, "not": 0}
        for decision in self.decisions:
            counts[decision.label] += 1
        return counts

    @property
    def certain(self) -> list[JunctionDecision]:
        return [decision for decision in self.decisions if decision.label == "certain"]

    @property
    def false(self) -> list[JunctionDecision]:
        return [decision for decision in self.decisions if decision.label == "false"]

    @property
    def not_classified(self) -> list[JunctionDecision]:
        return [decision for decision in self.decisions if decision.label == "not"]


def _as_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(float, copy=False)
    if arr.ndim == 3:
        return arr.astype(float, copy=False).mean(axis=2)
    raise ValueError(f"Expected 2D or 3D image array, got shape={arr.shape!r}")


def in_bounds(row: int, col: int, shape: tuple[int, int]) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def count_neighbors(skeleton: np.ndarray) -> np.ndarray:
    skel = skeleton.astype(np.uint8)
    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    return ndi.convolve(skel, kernel, mode="constant", cval=0)


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    return (skeleton > 0) & (count_neighbors(skeleton) == 1)


def junction_pixel_mask(skeleton: np.ndarray, min_neighbors: int = 3) -> np.ndarray:
    return (skeleton > 0) & (count_neighbors(skeleton) >= min_neighbors)


def trace_from_endpoint(skeleton: np.ndarray, start: tuple[int, int], max_steps: int = 100) -> np.ndarray:
    path = [start]
    prev = None
    curr = start
    shape = skeleton.shape

    for _ in range(max_steps):
        row, col = curr
        candidates: list[tuple[int, int]] = []
        for d_row, d_col in NEIGHBORS_8:
            n_row, n_col = row + d_row, col + d_col
            if not in_bounds(n_row, n_col, shape):
                continue
            if skeleton[n_row, n_col] == 0:
                continue
            if prev is not None and (n_row, n_col) == prev:
                continue
            candidates.append((n_row, n_col))

        if len(candidates) != 1:
            break

        prev, curr = curr, candidates[0]
        path.append(curr)

    return np.asarray(path, dtype=int)


def prune_short_spurs(skeleton: np.ndarray, min_len: int = 4, iterations: int = 1) -> np.ndarray:
    skel = skeleton.astype(bool).copy()

    for _ in range(iterations):
        endpoints = np.argwhere(endpoint_mask(skel))
        to_remove: list[tuple[int, int]] = []

        for row, col in endpoints:
            path = trace_from_endpoint(skel, (int(row), int(col)), max_steps=min_len + 2)
            if len(path) <= min_len:
                to_remove.extend(tuple(map(int, point)) for point in path[:-1])

        for row, col in to_remove:
            skel[row, col] = False

    return skel


def make_skeleton(mask: np.ndarray, use_medial_axis: bool = True) -> tuple[np.ndarray, np.ndarray]:
    bool_mask = mask.astype(bool)
    if use_medial_axis:
        skel, dist = medial_axis(bool_mask, return_distance=True)
        return skel.astype(bool), dist

    skel = skeletonize(bool_mask).astype(bool)
    dist = ndi.distance_transform_edt(bool_mask)
    return skel, dist


def group_junction_pixels(
    skeleton: np.ndarray,
    *,
    min_neighbors: int = 3,
    dilation_radius: int = 3,
    min_area: int = 1,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    jmask = junction_pixel_mask(skeleton, min_neighbors=min_neighbors)

    if dilation_radius > 0:
        grouped = ndi.binary_dilation(
            jmask,
            structure=ndi.generate_binary_structure(2, 2),
            iterations=dilation_radius,
        )
    else:
        grouped = jmask

    labelled = label(grouped, connectivity=2)
    groups: list[dict[str, Any]] = []

    for idx, region in enumerate(regionprops(labelled)):
        if region.area < min_area:
            continue
        coords = region.coords
        groups.append(
            {
                "id": idx,
                "center": coords.mean(axis=0),
                "pixels": coords,
                "area": int(region.area),
            }
        )

    return groups, jmask


def trace_arm(skeleton: np.ndarray, start: tuple[int, int], stop_junction_mask: np.ndarray, max_steps: int = 40) -> np.ndarray:
    shape = skeleton.shape
    path = [start]
    prev = None
    curr = start

    for _ in range(max_steps):
        row, col = curr
        candidates: list[tuple[int, int]] = []

        for d_row, d_col in NEIGHBORS_8:
            n_row, n_col = row + d_row, col + d_col
            if not in_bounds(n_row, n_col, shape):
                continue
            if skeleton[n_row, n_col] == 0:
                continue
            if prev is not None and (n_row, n_col) == prev:
                continue
            candidates.append((n_row, n_col))

        if len(candidates) == 0:
            break

        non_junction = [point for point in candidates if not stop_junction_mask[point[0], point[1]]]
        if len(non_junction) == 0:
            break

        if prev is None or len(non_junction) == 1:
            nxt = non_junction[0]
        else:
            v_prev = np.asarray(curr) - np.asarray(prev)
            best_point = None
            best_score = None

            for point in non_junction:
                v_next = np.asarray(point) - np.asarray(curr)
                score = -np.dot(v_prev, v_next) / (
                    np.linalg.norm(v_prev) * np.linalg.norm(v_next) + 1e-6
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_point = point
            nxt = best_point

        prev = curr
        curr = nxt
        path.append(curr)

    return np.asarray(path, dtype=int)


def make_circular_mask(shape: tuple[int, int], center: np.ndarray, radius: int) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2) <= radius


def label_border_components(skeleton: np.ndarray, removed_region_mask: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    skel_cut = skeleton.astype(bool).copy()
    skel_cut[removed_region_mask] = False

    dilated = ndi.binary_dilation(
        removed_region_mask,
        structure=ndi.generate_binary_structure(2, 2),
        iterations=1,
    )

    border = dilated & (~removed_region_mask) & skel_cut
    labelled = label(border, connectivity=2)

    starts: list[tuple[int, int]] = []
    coords = np.argwhere(removed_region_mask)
    center = coords.mean(axis=0) if len(coords) else np.array([0, 0])

    for region in regionprops(labelled):
        points = region.coords
        dist = np.linalg.norm(points - center[None, :], axis=1)
        starts.append(tuple(map(int, points[np.argmin(dist)])))

    return starts, skel_cut


def extract_arms_from_removed_region(
    skeleton: np.ndarray,
    removed_region_mask: np.ndarray,
    all_junction_pixel_mask: np.ndarray,
    *,
    max_arm_steps: int,
    min_arm_len: int,
    keep_short_arms: bool = False,
) -> list[dict[str, Any]]:
    starts, skel_cut = label_border_components(skeleton, removed_region_mask)
    stop_mask = all_junction_pixel_mask.copy()
    stop_mask[removed_region_mask] = False

    arms: list[dict[str, Any]] = []
    for start in starts:
        path = trace_arm(
            skeleton=skel_cut,
            start=start,
            stop_junction_mask=stop_mask,
            max_steps=max_arm_steps,
        )

        if len(path) < min_arm_len:
            if keep_short_arms and len(path) >= 2:
                arms.append({"start": start, "path": path, "length": int(len(path)), "short": True})
            continue

        arms.append({"start": start, "path": path, "length": int(len(path)), "short": False})

    return arms


def extract_arms_for_group(
    skeleton: np.ndarray,
    group: dict[str, Any],
    all_junction_pixel_mask: np.ndarray,
    *,
    remove_radius: int,
    max_arm_steps: int,
    min_arm_len: int,
    keep_short_arms: bool = False,
) -> list[dict[str, Any]]:
    removed = make_circular_mask(skeleton.shape, group["center"], remove_radius)
    return extract_arms_from_removed_region(
        skeleton=skeleton,
        removed_region_mask=removed,
        all_junction_pixel_mask=all_junction_pixel_mask,
        max_arm_steps=max_arm_steps,
        min_arm_len=min_arm_len,
        keep_short_arms=keep_short_arms,
    )


def crop_bounds(center: np.ndarray, shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    cy, cx = center
    cy, cx = int(round(float(cy))), int(round(float(cx)))
    half = size // 2

    y1 = max(0, cy - half)
    y2 = min(shape[0], cy + half)
    x1 = max(0, cx - half)
    x2 = min(shape[1], cx + half)
    return y1, y2, x1, x2


def local_reskeletonize_group(global_mask: np.ndarray, group: dict[str, Any], config: JunctionDecisionConfig) -> dict[str, Any]:
    y1, y2, x1, x2 = crop_bounds(group["center"], global_mask.shape, config.local_crop_size)

    local_mask = global_mask[y1:y2, x1:x2].copy()
    local_mask = binary_closing(local_mask, disk(1))
    local_mask = remove_small_objects(
        local_mask.astype(bool),
        min_size=max(5, config.min_object_size // 3),
    )

    local_skeleton, local_dist = make_skeleton(local_mask, use_medial_axis=config.local_use_medial_axis)

    if config.local_prune_iterations > 0:
        local_skeleton = prune_short_spurs(
            local_skeleton,
            min_len=config.local_prune_min_arm_len,
            iterations=config.local_prune_iterations,
        )

    local_center = np.asarray(group["center"], dtype=float) - np.array([y1, x1], dtype=float)
    local_group = {
        "id": group["id"],
        "center": local_center,
        "pixels": np.array([[int(round(local_center[0])), int(round(local_center[1]))]]),
        "area": group.get("area", 1),
        "global_center": group["center"],
        "crop_offset": (y1, x1),
    }

    local_junction_mask = junction_pixel_mask(local_skeleton, min_neighbors=3)
    local_arms = extract_arms_for_group(
        skeleton=local_skeleton,
        group=local_group,
        all_junction_pixel_mask=local_junction_mask,
        remove_radius=config.local_remove_radius,
        max_arm_steps=config.local_max_arm_steps,
        min_arm_len=config.local_min_arm_len,
        keep_short_arms=config.local_keep_short_arms,
    )

    global_arms: list[dict[str, Any]] = []
    for arm in local_arms:
        path = arm["path"].copy()
        path[:, 0] += y1
        path[:, 1] += x1
        global_arms.append({**arm, "path": path, "start": tuple(path[0]), "source": "local"})

    return {
        "local_mask": local_mask,
        "local_skeleton": local_skeleton,
        "local_dist": local_dist,
        "local_center": local_center,
        "offset": (y1, x1),
        "arms": global_arms,
    }


def local_thickness_score(distance_map: np.ndarray | None, center: np.ndarray, radius: int) -> float:
    if distance_map is None:
        return 0.0

    cy, cx = int(round(float(center[0]))), int(round(float(center[1])))
    values: list[float] = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y, x = cy + dy, cx + dx
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                values.append(float(distance_map[y, x]))

    return float(np.mean(values)) if values else 0.0


def max_thickness_score(distance_map: np.ndarray | None, center: np.ndarray, radius: int) -> float:
    if distance_map is None:
        return 0.0

    cy, cx = int(round(float(center[0]))), int(round(float(center[1])))
    values: list[float] = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y, x = cy + dy, cx + dx
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                values.append(float(distance_map[y, x]))

    return float(np.max(values)) if values else 0.0


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector * 0
    return vector / norm


def arm_direction_from_center(arm: dict[str, Any], center: np.ndarray, lookahead: int = 12) -> np.ndarray:
    path = arm["path"]
    if len(path) < 2:
        return np.array([0.0, 0.0])

    k = min(lookahead, len(path) - 1)
    p0 = np.asarray(center, dtype=float)
    p1 = np.asarray(path[k], dtype=float)
    return normalize_vec(p1 - p0)


def arm_local_curvature(arm: dict[str, Any], n_points: int = 20) -> float:
    path = arm["path"][:n_points]
    if len(path) < 4:
        return 1.0

    directions = [normalize_vec(path[i] - path[i - 1]) for i in range(1, len(path))]
    total = 0.0
    count = 0

    for i in range(1, len(directions)):
        dot = np.clip(np.dot(directions[i - 1], directions[i]), -1, 1)
        total += np.arccos(dot)
        count += 1

    return float(total / max(count, 1))


def sample_intensity_along_arm(image_gray: np.ndarray, arm: dict[str, Any], n_points: int = 25) -> np.ndarray:
    values: list[float] = []
    for row, col in arm["path"][:n_points]:
        row = int(np.clip(row, 0, image_gray.shape[0] - 1))
        col = int(np.clip(col, 0, image_gray.shape[1] - 1))
        values.append(float(image_gray[row, col]))

    return np.asarray(values, dtype=float) if values else np.array([0.0])


def intensity_profile_similarity(image_gray: np.ndarray, arm_a: dict[str, Any], arm_b: dict[str, Any], n_points: int = 25) -> float:
    intensity_a = sample_intensity_along_arm(image_gray, arm_a, n_points=n_points)
    intensity_b = sample_intensity_along_arm(image_gray, arm_b, n_points=n_points)

    common_len = min(len(intensity_a), len(intensity_b))
    if common_len < 3:
        return 1.0

    intensity_a = intensity_a[:common_len]
    intensity_b = intensity_b[:common_len]
    intensity_a = (intensity_a - intensity_a.mean()) / (intensity_a.std() + 1e-6)
    intensity_b = (intensity_b - intensity_b.mean()) / (intensity_b.std() + 1e-6)

    correlation = np.mean(intensity_a * intensity_b)
    return float(1.0 - correlation)


def continuation_cost(
    image_gray: np.ndarray,
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    center: np.ndarray,
    *,
    angle_weight: float = 1.0,
    curvature_weight: float = 0.30,
    intensity_weight: float = 0.35,
) -> float:
    vec_a = arm_direction_from_center(arm_a, center)
    vec_b = arm_direction_from_center(arm_b, center)

    dot = np.clip(np.dot(vec_a, vec_b), -1, 1)
    angle_cost = (dot + 1.0) / 2.0
    curvature_cost = (arm_local_curvature(arm_a) + arm_local_curvature(arm_b)) / 2.0
    intensity_cost = intensity_profile_similarity(image_gray, arm_a, arm_b)

    return float(angle_weight * angle_cost + curvature_weight * curvature_cost + intensity_weight * intensity_cost)


def all_pairings(indices: Iterable[int]) -> Iterable[list[tuple[int, int]]]:
    indices = list(indices)
    if len(indices) == 0:
        yield []
        return

    first = indices[0]
    for i in range(1, len(indices)):
        second = indices[i]
        rest = indices[1:i] + indices[i + 1 :]
        for pairing in all_pairings(rest):
            yield [(first, second)] + pairing


def best_pairing_cost(image_gray: np.ndarray, arms: list[dict[str, Any]], center: np.ndarray) -> dict[str, Any] | None:
    n_arms = len(arms)
    if n_arms < 2:
        return None

    pair_costs: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(n_arms), 2):
        key = tuple(sorted((i, j)))
        pair_costs[key] = continuation_cost(image_gray, arms[i], arms[j], center)

    candidate_pairings: list[list[tuple[int, int]]] = []
    if n_arms % 2 == 0:
        candidate_pairings = list(all_pairings(range(n_arms)))
    else:
        for missing in range(n_arms):
            rest = [i for i in range(n_arms) if i != missing]
            candidate_pairings.extend(all_pairings(rest))

    best: dict[str, Any] | None = None
    for pairing in candidate_pairings:
        costs = [pair_costs[tuple(sorted((a, b)))] for a, b in pairing]
        mean_cost = float(np.mean(costs)) if costs else 999.0
        max_cost = float(np.max(costs)) if costs else 999.0

        if best is None or mean_cost < best["mean_cost"]:
            best = {"pairs": pairing, "mean_cost": mean_cost, "max_cost": max_cost}

    return best


def decide_label_from_arms(
    image_gray: np.ndarray,
    group: dict[str, Any],
    center: np.ndarray,
    arms: list[dict[str, Any]],
    distance_map: np.ndarray | None,
    config: JunctionDecisionConfig,
) -> tuple[JunctionLabel, dict[str, Any] | None, str, float, float]:
    n_arms = len(arms)
    area = group.get("area", 0)

    thickness_mean = local_thickness_score(distance_map, center, config.thickness_radius)
    thickness_max = max_thickness_score(distance_map, center, config.thickness_radius)

    if n_arms < 3:
        if config.allow_two_arm_fake and n_arms == 2 and area >= config.min_area_for_two_arm_fake:
            return "false", None, "two_arm_area_false", thickness_mean, thickness_max

        if config.enable_thickness_fake and n_arms <= 2:
            if area >= config.thickness_area_thr and (
                thickness_mean >= config.thickness_fake_thr
                or thickness_max >= config.thickness_fake_thr + 1.0
            ):
                return "false", None, "thickness_false", thickness_mean, thickness_max

        return "not", None, "too_few_arms", thickness_mean, thickness_max

    best = best_pairing_cost(image_gray, arms, center)

    if best is None:
        return "not", None, "no_pairing", thickness_mean, thickness_max

    if (
        n_arms >= 4
        and best["mean_cost"] <= config.fake_mean_cost_thr
        and best["max_cost"] <= config.fake_max_cost_thr
    ):
        return "false", best, "good_pairing_false", thickness_mean, thickness_max

    if n_arms >= 3 and best["mean_cost"] > config.fake_max_cost_thr:
        return "certain", best, "bad_pairing_certain", thickness_mean, thickness_max

    if n_arms >= 4 and best["mean_cost"] <= (config.fake_max_cost_thr + 0.15):
        return "false", best, "soft_false", thickness_mean, thickness_max

    return "certain", best, "soft_certain", thickness_mean, thickness_max


def classify_single_junction(
    image_gray: np.ndarray,
    mask_clean: np.ndarray,
    skeleton: np.ndarray,
    group: dict[str, Any],
    all_junction_pixel_mask: np.ndarray,
    distance_map: np.ndarray | None,
    config: JunctionDecisionConfig,
) -> JunctionDecision:
    center = group["center"]

    arms = extract_arms_for_group(
        skeleton=skeleton,
        group=group,
        all_junction_pixel_mask=all_junction_pixel_mask,
        remove_radius=config.remove_radius,
        max_arm_steps=config.max_arm_steps,
        min_arm_len=config.min_arm_len,
        keep_short_arms=False,
    )

    label_out, best, reason, thickness_mean, thickness_max = decide_label_from_arms(
        image_gray, group, center, arms, distance_map, config
    )

    used_local = False

    if config.enable_local_reskeleton and label_out == "not":
        local_debug = local_reskeletonize_group(mask_clean, group, config)
        local_arms = local_debug["arms"]
        local_label, local_best, local_reason, _, _ = decide_label_from_arms(
            image_gray, group, center, local_arms, distance_map, config
        )

        if local_label != "not":
            label_out = local_label
            arms = local_arms
            best = local_best
            reason = "local_" + local_reason
            used_local = True
        elif len(local_arms) > len(arms):
            arms = local_arms
            reason = "local_more_arms_but_not"
            used_local = True

    return JunctionDecision(
        label=label_out,
        reason=reason,
        center=center,
        group=group,
        arms=arms,
        n_arms=len(arms),
        best_pairing=best,
        thickness_mean=thickness_mean,
        thickness_max=thickness_max,
        used_local_reskeleton=used_local,
    )


def run_junction_decision(
    image: np.ndarray,
    mask_clean: np.ndarray,
    skeleton: np.ndarray,
    distance_map: np.ndarray | None = None,
    config: JunctionDecisionConfig | None = None,
) -> JunctionDecisionResult:
    """Run notebook-equivalent junction classification on an already-built skeleton.

    Parameters are the objects already available in the main analysis flow:
    ``image`` from inference loading, cleaned binary ``mask_clean``, pruned
    ``skeleton`` and optional ``distance_map``. If ``distance_map`` is omitted,
    it is computed from ``mask_clean``.
    """
    if config is None:
        config = JunctionDecisionConfig()

    image_gray = _as_gray(image)
    mask_bool = mask_clean.astype(bool)
    skel_bool = skeleton.astype(bool)

    if distance_map is None:
        distance_map = ndi.distance_transform_edt(mask_bool)

    junction_groups, all_junction_pixel_mask = group_junction_pixels(
        skel_bool,
        dilation_radius=config.junction_group_dilation,
        min_area=config.min_junction_area,
    )

    decisions = [
        classify_single_junction(
            image_gray=image_gray,
            mask_clean=mask_bool,
            skeleton=skel_bool,
            group=group,
            all_junction_pixel_mask=all_junction_pixel_mask,
            distance_map=distance_map,
            config=config,
        )
        for group in junction_groups
    ]

    return JunctionDecisionResult(
        junction_groups=junction_groups,
        all_junction_pixel_mask=all_junction_pixel_mask,
        decisions=decisions,
    )
