from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .arms import extract_arms_for_group
from .grouping import find_junction_groups
from .local import refine_arms_locally
from .metrics import local_thickness_score, max_thickness_score
from .model import (
    DEFAULT_JUNCTION_DECISION_CONFIG,
    JunctionDecision,
    JunctionDecisionConfig,
    JunctionDecisionResult,
    JunctionLabel,
)
from .pairing import best_pairing_cost


ArmData = dict[str, Any]


@dataclass(frozen=True, slots=True)
class JunctionDecisionData:
    label: JunctionLabel
    best_pairing: dict[str, Any] | None
    reason: str
    thickness_mean: float
    thickness_max: float


def run_junction_decision(
    image: np.ndarray,
    mask_clean: np.ndarray,
    skeleton: np.ndarray,
    distance_map: np.ndarray | None = None,
    config: JunctionDecisionConfig | None = None,
) -> JunctionDecisionResult:
    config = config or DEFAULT_JUNCTION_DECISION_CONFIG
    image_gray = as_gray(image)
    mask_bool = mask_clean.astype(bool)
    skeleton_bool = skeleton.astype(bool)
    distance_map = distance_map_or_default(distance_map, mask_bool)
    junction_groups, all_junction_pixel_mask = find_junction_groups(
        skeleton_bool,
        dilation_radius=config.junction_group_dilation,
        min_area=config.min_junction_area,
    )
    decisions = classify_junction_groups(
        image_gray,
        skeleton_bool,
        junction_groups,
        all_junction_pixel_mask,
        distance_map,
        config,
    )
    return JunctionDecisionResult(junction_groups, all_junction_pixel_mask, decisions)


def as_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(float, copy=False)
    if array.ndim == 3:
        return array.astype(float, copy=False).mean(axis=2)
    raise ValueError(f"Expected 2D or 3D image array, got shape={array.shape!r}")


def distance_map_or_default(
    distance_map: np.ndarray | None, mask: np.ndarray
) -> np.ndarray:
    if distance_map is not None:
        return distance_map
    return ndi.distance_transform_edt(mask)


def classify_junction_groups(
    image_gray: np.ndarray,
    skeleton: np.ndarray,
    groups: list[dict[str, Any]],
    all_junction_pixel_mask: np.ndarray,
    distance_map: np.ndarray,
    config: JunctionDecisionConfig,
) -> list[JunctionDecision]:
    return [
        classify_single_junction(
            image_gray,
            skeleton,
            group,
            all_junction_pixel_mask,
            distance_map,
            config,
        )
        for group in groups
    ]


def classify_single_junction(
    image_gray: np.ndarray,
    skeleton: np.ndarray,
    group: dict[str, Any],
    all_junction_pixel_mask: np.ndarray,
    distance_map: np.ndarray | None,
    config: JunctionDecisionConfig,
) -> JunctionDecision:
    arms = extract_initial_arms(skeleton, group, all_junction_pixel_mask, config)
    decision_data = decide_label_from_arms(
        image_gray, group, group["center"], arms, distance_map, config
    )
    if should_refine_locally(decision_data.label, config):
        arms, decision_data, used_local_refine = apply_local_refine(
            image_gray,
            skeleton,
            group,
            arms,
            decision_data,
            distance_map,
            config,
        )
    else:
        used_local_refine = False
    return build_decision(group, arms, decision_data, used_local_refine)


def extract_initial_arms(
    skeleton: np.ndarray,
    group: dict[str, Any],
    all_junction_pixel_mask: np.ndarray,
    config: JunctionDecisionConfig,
) -> list[ArmData]:
    arms = extract_arms_for_group(
        skeleton=skeleton,
        group=group,
        all_junction_pixel_mask=all_junction_pixel_mask,
        remove_radius=config.remove_radius,
        max_arm_steps=config.max_arm_steps,
        min_arm_len=config.min_arm_len,
    )
    return normalize_arms(arms)


def normalize_arms(arms: list[Any]) -> list[ArmData]:
    return [arm_to_dict(arm) for arm in arms]


def arm_to_dict(arm: Any) -> ArmData:
    if isinstance(arm, dict):
        return arm

    if is_dataclass(arm):
        return {field.name: getattr(arm, field.name) for field in fields(arm)}

    path = getattr(arm, "path", None)
    if path is None:
        raise TypeError(f"Expected arm with a path field, got {type(arm).__name__}")

    return {"path": path}


def should_refine_locally(label: JunctionLabel, config: JunctionDecisionConfig) -> bool:
    return config.enable_local_refine and label is JunctionLabel.NOT


def apply_local_refine(
    image_gray: np.ndarray,
    skeleton: np.ndarray,
    group: dict[str, Any],
    arms: list[ArmData],
    decision_data: JunctionDecisionData,
    distance_map: np.ndarray | None,
    config: JunctionDecisionConfig,
) -> tuple[
    list[ArmData],
    JunctionDecisionData,
    bool,
]:
    local_arms = normalize_arms(refine_arms_locally(skeleton, group, config))
    local_decision_data = decide_label_from_arms(
        image_gray,
        group,
        group["center"],
        local_arms,
        distance_map,
        config,
    )
    if local_decision_data.label is not JunctionLabel.NOT:
        return local_arms, with_local_reason(local_decision_data), True
    if len(local_arms) > len(arms):
        return local_arms, with_reason(decision_data, "local_more_arms_but_not"), True
    return arms, decision_data, False


def with_local_reason(
    decision_data: JunctionDecisionData,
) -> JunctionDecisionData:
    return JunctionDecisionData(
        label=decision_data.label,
        best_pairing=decision_data.best_pairing,
        reason=f"local_{decision_data.reason}",
        thickness_mean=decision_data.thickness_mean,
        thickness_max=decision_data.thickness_max,
    )


def with_reason(
    decision_data: JunctionDecisionData,
    reason: str,
) -> JunctionDecisionData:
    return JunctionDecisionData(
        label=decision_data.label,
        best_pairing=decision_data.best_pairing,
        reason=reason,
        thickness_mean=decision_data.thickness_mean,
        thickness_max=decision_data.thickness_max,
    )


def build_decision(
    group: dict[str, Any],
    arms: list[ArmData],
    decision_data: JunctionDecisionData,
    used_local_refine: bool,
) -> JunctionDecision:
    return JunctionDecision(
        label=decision_data.label,
        reason=decision_data.reason,
        center=group["center"],
        group=group,
        arms=arms,
        n_arms=len(arms),
        best_pairing=decision_data.best_pairing,
        thickness_mean=decision_data.thickness_mean,
        thickness_max=decision_data.thickness_max,
        used_local_refine=used_local_refine,
    )


def decide_label_from_arms(
    image_gray: np.ndarray,
    group: dict[str, Any],
    center: np.ndarray,
    arms: list[ArmData],
    distance_map: np.ndarray | None,
    config: JunctionDecisionConfig,
) -> JunctionDecisionData:
    thickness_mean, thickness_max = thickness_scores(distance_map, center, config)
    fake_reason = classify_low_arm_count(
        len(arms), group, thickness_mean, thickness_max, config
    )
    if fake_reason is not None:
        return JunctionDecisionData(
            label=fake_reason[0],
            best_pairing=None,
            reason=fake_reason[1],
            thickness_mean=thickness_mean,
            thickness_max=thickness_max,
        )

    if len(arms) > config.max_pairing_arms:
        return JunctionDecisionData(
            label=JunctionLabel.NOT,
            best_pairing=None,
            reason="too_many_arms_for_pairing",
            thickness_mean=thickness_mean,
            thickness_max=thickness_max,
        )

    best = best_pairing_cost(
        image_gray,
        arms,
        center,
        config.max_pairing_arms,
    )
    if best is None:
        return JunctionDecisionData(
            label=JunctionLabel.NOT,
            best_pairing=None,
            reason="no_pairing",
            thickness_mean=thickness_mean,
            thickness_max=thickness_max,
        )
    label, reason = classify_pairing(len(arms), best, config)
    return JunctionDecisionData(
        label=label,
        best_pairing=best,
        reason=reason,
        thickness_mean=thickness_mean,
        thickness_max=thickness_max,
    )


def thickness_scores(
    distance_map: np.ndarray | None,
    center: np.ndarray,
    config: JunctionDecisionConfig,
) -> tuple[float, float]:
    return (
        local_thickness_score(distance_map, center, config.thickness_radius),
        max_thickness_score(distance_map, center, config.thickness_radius),
    )


def classify_low_arm_count(
    n_arms: int,
    group: dict[str, Any],
    thickness_mean: float,
    thickness_max: float,
    config: JunctionDecisionConfig,
) -> tuple[JunctionLabel, str] | None:
    if n_arms >= 3:
        return None
    if is_two_arm_fake(n_arms, group, config):
        return JunctionLabel.FALSE, "two_arm_area_false"
    if is_thickness_fake(n_arms, group, thickness_mean, thickness_max, config):
        return JunctionLabel.FALSE, "thickness_false"
    return JunctionLabel.NOT, "too_few_arms"


def is_two_arm_fake(
    n_arms: int, group: dict[str, Any], config: JunctionDecisionConfig
) -> bool:
    return (
        config.allow_two_arm_fake
        and n_arms == 2
        and group.get("area", 0) >= config.min_area_for_two_arm_fake
    )


def is_thickness_fake(
    n_arms: int,
    group: dict[str, Any],
    thickness_mean: float,
    thickness_max: float,
    config: JunctionDecisionConfig,
) -> bool:
    if not config.enable_thickness_fake or n_arms > 2:
        return False
    if group.get("area", 0) < config.thickness_area_threshold:
        return False
    return (
        thickness_mean >= config.thickness_fake_threshold
        or thickness_max
        >= config.thickness_fake_threshold + config.thickness_fake_max_extra
    )


def classify_pairing(
    n_arms: int,
    best: dict[str, Any],
    config: JunctionDecisionConfig,
) -> tuple[JunctionLabel, str]:
    if is_good_false_pairing(n_arms, best, config):
        return JunctionLabel.FALSE, "good_pairing_false"
    if is_bad_certain_pairing(n_arms, best, config):
        return JunctionLabel.CERTAIN, "bad_pairing_certain"
    if is_soft_false_pairing(n_arms, best, config):
        return JunctionLabel.FALSE, "soft_false"
    return JunctionLabel.CERTAIN, "soft_certain"


def is_good_false_pairing(
    n_arms: int, best: dict[str, Any], config: JunctionDecisionConfig
) -> bool:
    return (
        n_arms >= 4
        and best["mean_cost"] <= config.fake_mean_cost_threshold
        and best["max_cost"] <= config.fake_max_cost_threshold
    )


def is_bad_certain_pairing(
    n_arms: int, best: dict[str, Any], config: JunctionDecisionConfig
) -> bool:
    return n_arms >= 3 and best["mean_cost"] > config.fake_max_cost_threshold


def is_soft_false_pairing(
    n_arms: int, best: dict[str, Any], config: JunctionDecisionConfig
) -> bool:
    return (
        n_arms >= 4
        and best["mean_cost"]
        <= config.fake_max_cost_threshold + config.soft_false_mean_cost_extra
    )
