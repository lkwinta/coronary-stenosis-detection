"""Oriented vessel-segment feature extraction for XGBoost stenosis inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd

from coronary_analysis.topology import estimate_branch_diameters


@dataclass(frozen=True)
class OrientedSegmentConfig:
    segment_length_px: float = 10.0
    segment_step_px: float = 10.0
    min_segment_length_px: float = 4.0
    centerline_sample_step_px: float = 1.0
    width_scale: float = 1.15
    min_patch_width_px: float = 4.0
    max_patch_width_px: float = 40.0
    save_centerline_points: bool = True
    save_patch_pixels: bool = False


def yx_to_xy(points_yx: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_yx, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([pts[:, 1], pts[:, 0]])


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.array([0.0])
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def interpolate_along_polyline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts, len(distances), axis=0)

    cum = cumulative_lengths(pts)
    seg_lens = np.diff(cum)
    out = []
    for d in distances:
        d = float(np.clip(d, 0.0, cum[-1]))
        idx = np.searchsorted(cum, d, side="right") - 1
        idx = min(max(idx, 0), len(seg_lens) - 1)
        if seg_lens[idx] <= 1e-8:
            out.append(pts[idx])
        else:
            t = (d - cum[idx]) / seg_lens[idx]
            out.append(pts[idx] + t * (pts[idx + 1] - pts[idx]))
    return np.asarray(out, dtype=float)


def segment_distance_ranges(total_length: float, config: OrientedSegmentConfig) -> list[tuple[float, float]]:
    if total_length < config.min_segment_length_px:
        return []
    ranges: list[tuple[float, float]] = []
    d0 = 0.0
    while d0 < total_length:
        d1 = min(d0 + config.segment_length_px, total_length)
        if d1 - d0 >= config.min_segment_length_px:
            ranges.append((float(d0), float(d1)))
        if d1 >= total_length:
            break
        d0 += config.segment_step_px
    return ranges


def sample_centerline_segment(path_xy: np.ndarray, d0: float, d1: float, config: OrientedSegmentConfig) -> np.ndarray:
    n_samples = max(3, int(math.ceil((d1 - d0) / config.centerline_sample_step_px)) + 1)
    distances = np.linspace(d0, d1, n_samples)
    return interpolate_along_polyline(path_xy, distances)


def local_diameters_for_centerline(centerline_xy: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    pts = np.asarray(centerline_xy, dtype=float)
    if len(pts) == 0:
        return np.array([np.nan], dtype=float)
    h, w = dist_map.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return 2.0 * np.asarray(dist_map[ys, xs], dtype=float)


def oriented_rectangle_from_centerline(centerline_xy: np.ndarray, width_px: float) -> tuple[np.ndarray | None, float]:
    pts = np.asarray(centerline_xy, dtype=float)
    if len(pts) < 2:
        return None, np.nan
    p0 = pts[0]
    p1 = pts[-1]
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        return None, np.nan
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]], dtype=float)
    half_w = float(width_px) / 2.0
    vertices = np.array([
        p0 + normal * half_w,
        p1 + normal * half_w,
        p1 - normal * half_w,
        p0 - normal * half_w,
    ], dtype=float)
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return vertices, angle_deg


def polygon_to_mask(poly_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    poly = np.asarray(poly_xy, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return mask
    poly_i = np.round(poly).astype(np.int32)
    poly_i[:, 0] = np.clip(poly_i[:, 0], 0, width - 1)
    poly_i[:, 1] = np.clip(poly_i[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [poly_i], 1)
    return mask


def mask_pixels_xy(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.where(mask > 0)
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


def safe_branch_diameters(path_yx: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    try:
        d = np.asarray(estimate_branch_diameters(path_yx, dist_map), dtype=float)
        return d if len(d) else np.array([np.nan], dtype=float)
    except Exception:
        return local_diameters_for_centerline(yx_to_xy(path_yx), dist_map)


def _junction_counts(junction_decision: Any | None) -> dict[str, int]:
    if junction_decision is None:
        return {}
    try:
        return dict(junction_decision.counts)
    except Exception:
        return {}


def graph_to_oriented_segment_rows(
    skel_obj: Any,
    branch_data: pd.DataFrame,
    distance_map: np.ndarray,
    *,
    image_id: int = 0,
    file_name: str = "",
    junction_decision: Any | None = None,
    used_junction_cleanup: bool = True,
    config: OrientedSegmentConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert a vessel graph into oriented 10px segment feature rows."""
    config = config or OrientedSegmentConfig()
    counts = _junction_counts(junction_decision)
    h, w = distance_map.shape[:2]
    rows: list[dict[str, Any]] = []
    global_segment_id = 0

    for branch_id in range(len(branch_data)):
        try:
            path_yx = np.asarray(skel_obj.path_coordinates(branch_id), dtype=float)
        except Exception:
            continue
        if len(path_yx) < 2:
            continue

        path_xy = yx_to_xy(path_yx)
        total_len = polyline_length(path_xy)
        if total_len < config.min_segment_length_px:
            continue

        branch_row = branch_data.iloc[branch_id].to_dict()
        branch_len = float(branch_row.get("branch_distance", total_len))
        euclidean = float(branch_row.get("euclidean_distance", np.nan))
        if not np.isfinite(euclidean) or euclidean <= 0:
            euclidean = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
        tortuosity = float(branch_len / euclidean) if euclidean > 0 else np.nan
        branch_diam = safe_branch_diameters(path_yx, distance_map)

        for local_segment_id, (d0, d1) in enumerate(segment_distance_ranges(total_len, config)):
            centerline_xy = sample_centerline_segment(path_xy, d0, d1, config)
            segment_len = polyline_length(centerline_xy)
            if segment_len < config.min_segment_length_px:
                continue

            local_diam = local_diameters_for_centerline(centerline_xy, distance_map)
            source_diam = local_diam if np.isfinite(local_diam).any() else branch_diam
            mean_d = float(np.nanmean(source_diam))
            min_d = float(np.nanmin(source_diam))
            max_d = float(np.nanmax(source_diam))
            std_d = float(np.nanstd(source_diam))
            if not np.isfinite(mean_d) or mean_d <= 0:
                mean_d = config.min_patch_width_px / config.width_scale

            patch_width = float(np.clip(mean_d * config.width_scale, config.min_patch_width_px, config.max_patch_width_px))
            vertices_xy, angle_deg = oriented_rectangle_from_centerline(centerline_xy, patch_width)
            if vertices_xy is None:
                continue
            patch_mask = polygon_to_mask(vertices_xy, h, w)
            patch_area = int(patch_mask.sum())
            if patch_area == 0:
                continue

            cx = float(np.mean(centerline_xy[:, 0]))
            cy = float(np.mean(centerline_xy[:, 1]))
            x_min = float(np.min(vertices_xy[:, 0]))
            y_min = float(np.min(vertices_xy[:, 1]))
            x_max = float(np.max(vertices_xy[:, 0]))
            y_max = float(np.max(vertices_xy[:, 1]))

            row: dict[str, Any] = {
                "image_id": int(image_id),
                "file_name": file_name,
                "branch_id": int(branch_id),
                "oriented_segment_id": int(global_segment_id),
                "local_segment_id": int(local_segment_id),
                "arc_start_px": float(d0),
                "arc_end_px": float(d1),
                "center_x": cx,
                "center_y": cy,
                "segment_length_px": float(segment_len),
                "patch_width_px": float(patch_width),
                "patch_area_px": int(patch_area),
                "angle_deg": float(angle_deg),
                "oriented_box_vertices_xy": json.dumps(vertices_xy.round(2).tolist()),
                "axis_bbox_x_min": x_min,
                "axis_bbox_y_min": y_min,
                "axis_bbox_x_max": x_max,
                "axis_bbox_y_max": y_max,
                "axis_bbox_width": x_max - x_min,
                "axis_bbox_height": y_max - y_min,
                "mean_diameter": mean_d,
                "min_diameter": min_d,
                "max_diameter": max_d,
                "std_diameter": std_d,
                "diameter_drop": float((mean_d - min_d) / (mean_d + 1e-8)),
                "branch_distance": branch_len,
                "branch_euclidean_distance": float(euclidean),
                "branch_tortuosity": tortuosity,
                "branch_type": int(branch_row.get("branch_type", -1)) if pd.notna(branch_row.get("branch_type", np.nan)) else -1,
                "n_junction_certain": int(counts.get("certain", 0)),
                "n_junction_false": int(counts.get("false", 0)),
                "n_junction_not": int(counts.get("not", 0)),
                "used_junction_cleanup": bool(used_junction_cleanup),
            }
            if config.save_centerline_points:
                row["centerline_points_xy"] = json.dumps(centerline_xy.round(2).tolist())
            if config.save_patch_pixels:
                row["patch_pixels_xy"] = json.dumps(mask_pixels_xy(patch_mask))
            rows.append(row)
            global_segment_id += 1
    return rows
