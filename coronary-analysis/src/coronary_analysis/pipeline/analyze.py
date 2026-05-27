from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from coronary_analysis.inference import (
    load_image,
    load_segmentation_model,
    predict_mask,
)
from coronary_analysis.topology import (
    build_vessel_graph,
    clean_mask,
    compute_distance_map,
    compute_topology_stats,
    estimate_branch_diameters,
    prune_skeleton,
    skeletonize_mask,
)
from coronary_analysis.topology.junction_decision import (
    DEFAULT_JUNCTION_DECISION_CONFIG,
    JunctionDecision,
    JunctionDecisionConfig,
    JunctionDecisionResult,
    JunctionLabel,
    run_junction_decision,
)
from coronary_analysis.topology.junction_decision.graph_cleanup import (
    remove_false_junctions_from_skeleton,
)
from coronary_analysis.utils import get_device


@dataclass(frozen=True)
class AnalysisConfig:
    encoder_name: str = "resnet34"
    img_size: int = 256
    threshold: float = 0.5
    closing_radius: float = 2
    max_hole_size: int = 50
    min_object_size: int = 50
    min_branch_length: int = 15
    junction_config: JunctionDecisionConfig = field(
        default_factory=lambda: DEFAULT_JUNCTION_DECISION_CONFIG
    )


@dataclass
class AnalysisResult:
    image: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    stats: dict
    branch_details: list[dict]
    junction_decision: JunctionDecisionResult

    @property
    def junction_groups(self) -> list[dict]:
        return self.junction_decision.junction_groups

    @property
    def junction_results(self) -> list[JunctionDecision]:
        return self.junction_decision.decisions

    @property
    def junction_counts(self) -> dict[str, int]:
        return self.junction_decision.counts

    @property
    def certain_junctions(self) -> list[JunctionDecision]:
        return self.junction_decision.certain


def run_analysis(
    image_path: str | Path,
    model_path: str | Path,
    device: torch.device | None = None,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    if device is None:
        device = get_device()

    config = config or AnalysisConfig()

    image = load_image(image_path)
    model = load_segmentation_model(
        str(model_path),
        device=device,
        encoder_name=config.encoder_name,
    )

    mask = predict_mask(
        image,
        model,
        device=device,
        img_size=config.img_size,
        threshold=config.threshold,
    )

    mask = clean_mask(
        mask,
        closing_radius=config.closing_radius,
        max_hole_size=config.max_hole_size,
        min_object_size=config.min_object_size,
    )

    skeleton = skeletonize_mask(mask)
    skeleton = prune_skeleton(skeleton, min_branch_length=config.min_branch_length)
    dist_map = compute_distance_map(mask)

    junction_decision = run_junction_decision(
        image=image,
        mask_clean=mask,
        skeleton=skeleton,
        distance_map=dist_map,
        config=config.junction_config,
    )

    skeleton, _ = remove_false_junctions_from_skeleton(
        skeleton=skeleton,
        junction_decision=junction_decision,
        config=config.junction_config,
    )

    skel_obj, branch_data = build_vessel_graph(skeleton)
    stats = compute_topology_stats(branch_data)

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

    stats = {
        **stats,
        "junction_counts": junction_decision.counts,
        "n_junction_groups": len(junction_decision.junction_groups),
        "n_certain_junctions": junction_decision.counts[JunctionLabel.CERTAIN.value],
        "n_false_junctions": junction_decision.counts[JunctionLabel.FALSE.value],
        "n_not_junctions": junction_decision.counts[JunctionLabel.NOT.value],
    }

    return AnalysisResult(
        image=image,
        mask=mask,
        skeleton=skeleton,
        stats=stats,
        branch_details=branch_details,
        junction_decision=junction_decision,
    )
