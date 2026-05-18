from dataclasses import dataclass
from pathlib import Path
from coronary_analysis.inference import (
    load_image,
    load_segmentation_model,
    predict_mask,
)
from coronary_analysis.topology import (
    clean_mask,
    skeletonize_mask,
    prune_skeleton,
    build_vessel_graph,
    compute_topology_stats,
    compute_distance_map,
    estimate_branch_diameters,
)
from coronary_analysis.utils import get_device

import numpy as np
import torch


@dataclass
class AnalysisResult:
    image: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    stats: dict
    branch_details: list[dict]


def run_analysis(
    image_path: str | Path,
    model_path: str | Path,
    device: torch.device | None = None,
    encoder_name: str = "resnet34",
    img_size: int = 256,
    threshold: float = 0.5,
    closing_radius: float = 2,
    max_hole_size: int = 50,
    min_object_size: int = 50,
    min_branch_length: int = 15,
) -> AnalysisResult:
    if device is None:
        device = get_device()

    image = load_image(image_path)
    model = load_segmentation_model(
        str(model_path), device=device, encoder_name=encoder_name
    )
    mask = predict_mask(
        image, model, device=device, img_size=img_size, threshold=threshold
    )
    mask = clean_mask(
        mask,
        closing_radius=closing_radius,
        max_hole_size=max_hole_size,
        min_object_size=min_object_size,
    )
    skeleton = skeletonize_mask(mask)
    skeleton = prune_skeleton(skeleton, min_branch_length=min_branch_length)
    skel_obj, branch_data = build_vessel_graph(skeleton)
    stats = compute_topology_stats(branch_data)
    dist_map = compute_distance_map(mask)
    branch_details = []

    for i in range(len(branch_data)):
        path = skel_obj.path_coordinates(i)
        diameters = estimate_branch_diameters(path, dist_map)
        branch_details.append(
            {
                "branch_id": i,
                "mean_diameter": float(diameters.mean()),
                "min_diameter": float(diameters.min()),
                "max_diameter": float(diameters.max()),
                "length": float(branch_data.iloc[i]["branch_distance"]),
                "branch_type": int(branch_data.iloc[i]["branch_type"]),
            }
        )

    return AnalysisResult(
        image=image,
        mask=mask,
        skeleton=skeleton,
        stats=stats,
        branch_details=branch_details,
    )
