from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from .pixels import junction_pixel_mask


def find_junction_groups(
    skeleton: np.ndarray,
    dilation_radius: int,
    min_area: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    junction_mask = junction_pixel_mask(skeleton)
    grouped_mask = dilate_junction_pixels(junction_mask, dilation_radius)
    groups = describe_groups(grouped_mask, min_area)
    return groups, junction_mask


def dilate_junction_pixels(junction_mask: np.ndarray, dilation_radius: int) -> np.ndarray:
    if dilation_radius <= 0:
        return junction_mask
    return ndi.binary_dilation(
        junction_mask,
        structure=ndi.generate_binary_structure(2, 2),
        iterations=dilation_radius,
    )


def describe_groups(grouped_mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    labelled = label(grouped_mask, connectivity=2)
    groups: list[dict[str, Any]] = []
    for region_id, region in enumerate(regionprops(labelled)):
        if region.area >= min_area:
            groups.append(describe_group(region_id, region))
    return groups


def describe_group(region_id: int, region) -> dict[str, Any]:
    coords = region.coords
    return {
        "id": region_id,
        "center": coords.mean(axis=0),
        "pixels": coords,
        "area": int(region.area),
    }
