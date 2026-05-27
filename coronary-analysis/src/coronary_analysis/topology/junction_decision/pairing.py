from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np

from .metrics import continuation_cost


def best_pairing_cost(
    image_gray: np.ndarray,
    arms: list[dict[str, Any]],
    center: np.ndarray,
    max_pairing_arms: int = 8,
) -> dict[str, Any] | None:
    if len(arms) < 2:
        return None
    if len(arms) > max_pairing_arms:
        return None

    pair_costs = compute_pair_costs(image_gray, arms, center)
    candidate_pairings = list(candidate_arm_pairings(len(arms)))
    return select_best_pairing(pair_costs, candidate_pairings)


def compute_pair_costs(
    image_gray: np.ndarray,
    arms: list[dict[str, Any]],
    center: np.ndarray,
) -> dict[tuple[int, int], float]:
    pair_costs: dict[tuple[int, int], float] = {}
    for first, second in combinations(range(len(arms)), 2):
        pair_costs[(first, second)] = continuation_cost(
            image_gray, arms[first], arms[second], center
        )
    return pair_costs


def candidate_arm_pairings(n_arms: int) -> Iterable[list[tuple[int, int]]]:
    if n_arms % 2 == 0:
        yield from all_pairings(range(n_arms))
        return
    for missing in range(n_arms):
        remaining = [index for index in range(n_arms) if index != missing]
        yield from all_pairings(remaining)


def all_pairings(indices: Iterable[int]) -> Iterable[list[tuple[int, int]]]:
    indices = list(indices)
    if not indices:
        yield []
        return
    first = indices[0]
    for index in range(1, len(indices)):
        second = indices[index]
        remaining = indices[1:index] + indices[index + 1 :]
        for pairing in all_pairings(remaining):
            yield [(first, second)] + pairing


def select_best_pairing(
    pair_costs: dict[tuple[int, int], float],
    candidate_pairings: list[list[tuple[int, int]]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for pairing in candidate_pairings:
        candidate = describe_pairing(pair_costs, pairing)
        if best is None or candidate["mean_cost"] < best["mean_cost"]:
            best = candidate
    return best


def describe_pairing(
    pair_costs: dict[tuple[int, int], float],
    pairing: list[tuple[int, int]],
) -> dict[str, Any]:
    costs = [pair_costs[tuple(sorted(pair))] for pair in pairing]
    return {
        "pairs": pairing,
        "mean_cost": float(np.mean(costs)) if costs else 999.0,
        "max_cost": float(np.max(costs)) if costs else 999.0,
    }
