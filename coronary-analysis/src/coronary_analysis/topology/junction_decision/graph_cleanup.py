from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from coronary_analysis.topology.junction_decision.model import (
    DEFAULT_JUNCTION_DECISION_CONFIG,
    JunctionDecision,
    JunctionDecisionConfig,
    JunctionDecisionResult,
    JunctionLabel,
)
from coronary_analysis.topology.pixel_graph import (
    PixelNode,
    PixelPair,
    build_skeleton_pixel_graph,
    pixel_graph_to_skeleton,
)


def remove_false_junctions_from_skeleton(
    skeleton: np.ndarray,
    junction_decision: JunctionDecisionResult,
    *,
    config: JunctionDecisionConfig = DEFAULT_JUNCTION_DECISION_CONFIG,
) -> tuple[np.ndarray, nx.Graph]:
    graph = build_skeleton_pixel_graph(skeleton)

    for decision in junction_decision.decisions:
        if decision.label is not JunctionLabel.FALSE:
            continue

        node = _find_nearest_node(
            graph,
            decision.center,
            config.graph_cleanup_max_center_distance,
        )
        if node is None:
            continue

        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= 2:
            pairs = _neighbor_pairs(node, neighbors, decision, config)
            _connect_pairs(graph, node, pairs)

        graph.remove_node(node)

    return pixel_graph_to_skeleton(graph, skeleton.shape), graph


def _find_nearest_node(
    graph: nx.Graph,
    center: np.ndarray,
    max_distance: float,
) -> PixelNode | None:
    center = np.asarray(center, dtype=float)

    best_node = None
    best_distance = float("inf")

    for node in graph.nodes:
        distance = float(np.linalg.norm(np.asarray(node, dtype=float) - center))
        if distance < best_distance:
            best_node = node
            best_distance = distance

    if best_distance > max_distance:
        return None

    return best_node


def _neighbor_pairs(
    node: PixelNode,
    neighbors: list[PixelNode],
    decision: JunctionDecision,
    config: JunctionDecisionConfig,
) -> list[PixelPair]:
    pairs = _pairs_from_best_pairing(neighbors, decision, config)
    if pairs:
        return pairs

    return _pairs_by_opposite_direction(node, neighbors)


def _pairs_from_best_pairing(
    neighbors: list[PixelNode],
    decision: JunctionDecision,
    config: JunctionDecisionConfig,
) -> list[PixelPair]:
    best_pairing = decision.best_pairing
    if not best_pairing or "pairs" not in best_pairing:
        return []

    arm_to_neighbor = _map_arms_to_neighbors(neighbors, decision, config)

    pairs: list[PixelPair] = []
    used: set[PixelNode] = set()

    for arm_a, arm_b in best_pairing["pairs"]:
        neighbor_a = arm_to_neighbor.get(int(arm_a))
        neighbor_b = arm_to_neighbor.get(int(arm_b))

        if neighbor_a is None or neighbor_b is None:
            continue
        if neighbor_a == neighbor_b:
            continue
        if neighbor_a in used or neighbor_b in used:
            continue

        pairs.append((neighbor_a, neighbor_b))
        used.add(neighbor_a)
        used.add(neighbor_b)

    return pairs


def _map_arms_to_neighbors(
    neighbors: list[PixelNode],
    decision: JunctionDecision,
    config: JunctionDecisionConfig,
) -> dict[int, PixelNode]:
    center = np.asarray(decision.center, dtype=float)
    mapping: dict[int, PixelNode] = {}
    used_neighbors: set[PixelNode] = set()

    for arm_index, arm in enumerate(decision.arms):
        arm_direction = _arm_direction(arm, center)
        if arm_direction is None:
            continue

        best_neighbor = None
        best_score = -float("inf")

        for neighbor in neighbors:
            if neighbor in used_neighbors:
                continue

            neighbor_direction = _normalize(np.asarray(neighbor, dtype=float) - center)
            score = float(np.dot(arm_direction, neighbor_direction))

            if score > best_score:
                best_neighbor = neighbor
                best_score = score

        if (
            best_neighbor is not None
            and best_score > config.graph_cleanup_neighbor_score_threshold
        ):
            mapping[arm_index] = best_neighbor
            used_neighbors.add(best_neighbor)

    return mapping


def _arm_direction(arm: dict[str, Any], center: np.ndarray) -> np.ndarray | None:
    path = np.asarray(arm.get("path"), dtype=float)

    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return None

    point = path[min(len(path) - 1, 8)]
    return _normalize(point - center)


def _pairs_by_opposite_direction(
    node: PixelNode,
    neighbors: list[PixelNode],
) -> list[PixelPair]:
    center = np.asarray(node, dtype=float)
    remaining = set(neighbors)
    pairs: list[PixelPair] = []

    while len(remaining) >= 2:
        best_pair = None
        best_score = float("inf")
        remaining_list = list(remaining)

        for i, neighbor_a in enumerate(remaining_list):
            vec_a = _normalize(np.asarray(neighbor_a, dtype=float) - center)

            for neighbor_b in remaining_list[i + 1 :]:
                vec_b = _normalize(np.asarray(neighbor_b, dtype=float) - center)
                score = float(np.dot(vec_a, vec_b))

                if score < best_score:
                    best_pair = (neighbor_a, neighbor_b)
                    best_score = score

        if best_pair is None:
            break

        pairs.append(best_pair)
        remaining.remove(best_pair[0])
        remaining.remove(best_pair[1])

    return pairs


def _connect_pairs(
    graph: nx.Graph,
    removed_node: PixelNode,
    pairs: list[PixelPair],
) -> None:
    for node_a, node_b in pairs:
        if node_a == node_b:
            continue

        path = np.asarray([node_a, removed_node, node_b], dtype=int)
        length = _path_length(path)

        graph.add_edge(
            node_a,
            node_b,
            length=length,
            weight=length,
            path=path,
            removed_false_junction=removed_node,
        )


def _path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0

    diffs = np.diff(path.astype(float), axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=float)

    return vector / norm
