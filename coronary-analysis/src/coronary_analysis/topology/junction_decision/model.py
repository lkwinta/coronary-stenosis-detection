from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_PATH = (Path(__file__).resolve().parents[2] / "configs" / "junction_decision_config.json")

class JunctionLabel(str, Enum):
    CERTAIN = "certain"
    FALSE = "false"
    NOT = "not"


@dataclass(frozen=True)
class JunctionDecisionConfig:
    junction_group_dilation: int
    min_junction_area: int
    remove_radius: int
    max_arm_steps: int
    min_arm_len: int
    enable_local_refine: bool
    local_crop_size: int
    local_remove_radius: int
    local_max_arm_steps: int
    local_min_arm_len: int
    local_keep_short_arms: bool
    local_prune_min_branch_length: int
    fake_mean_cost_threshold: float
    fake_max_cost_threshold: float
    allow_two_arm_fake: bool
    min_area_for_two_arm_fake: int
    enable_thickness_fake: bool
    thickness_radius: int
    thickness_fake_threshold: float
    thickness_area_threshold: int

    @property
    def enable_local_reskeleton(self) -> bool:
        return self.enable_local_refine

    @property
    def fake_mean_cost_thr(self) -> float:
        return self.fake_mean_cost_threshold

    @property
    def fake_max_cost_thr(self) -> float:
        return self.fake_max_cost_threshold

    @property
    def thickness_fake_thr(self) -> float:
        return self.thickness_fake_threshold

    @property
    def thickness_area_thr(self) -> int:
        return self.thickness_area_threshold


def load_junction_decision_config(
    path: str | Path = CONFIG_PATH,
) -> JunctionDecisionConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return JunctionDecisionConfig(**data)


DEFAULT_JUNCTION_DECISION_CONFIG = load_junction_decision_config()


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
    used_local_refine: bool = False

    @property
    def used_local_reskeleton(self) -> bool:
        return self.used_local_refine


@dataclass
class JunctionDecisionResult:
    junction_groups: list[dict[str, Any]]
    all_junction_pixel_mask: np.ndarray
    decisions: list[JunctionDecision] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = {label.value: 0 for label in JunctionLabel}
        for decision in self.decisions:
            counts[decision.label.value] += 1
        return counts

    @property
    def certain(self) -> list[JunctionDecision]:
        return self.by_label(JunctionLabel.CERTAIN)

    @property
    def false(self) -> list[JunctionDecision]:
        return self.by_label(JunctionLabel.FALSE)

    @property
    def not_classified(self) -> list[JunctionDecision]:
        return self.by_label(JunctionLabel.NOT)

    def by_label(self, label: JunctionLabel) -> list[JunctionDecision]:
        return [decision for decision in self.decisions if decision.label is label]