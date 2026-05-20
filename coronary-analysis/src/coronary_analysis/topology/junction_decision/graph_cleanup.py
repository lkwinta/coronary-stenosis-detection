from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from coronary_analysis.topology.junction_decision.model import (
    JunctionDecisionResult,
    JunctionLabel,
)
from coronary_analysis.topology.skeleton import DIRECTIONS


PixelNode = tuple[int, int]
PixelPair = tuple[PixelNode, PixelNode]


def remove_false_junctions_from_skeleton(
    skeleton: np.ndarray,
    junction_decision: JunctionDecisionResult,
    *,
    max_center_distance: float = 8.0,
) -> tuple[np.ndarray, nx.Graph]:
    graph = _build_skeleton_graph(skeleton)

    for decision in junction_decision.decisions:
        if _label_value(decision.label) != JunctionLabel.FALSE.value:
            continue

        node = _find_nearest_node(graph, decision.center, max_center_distance)
        if node is None:
            continue

        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= 2:
            pairs = _neighbor_pairs(node, neighbors, decision)
            _connect_pairs(graph, node, pairs)

        graph.remove_node(node)

    return _graph_to_skeleton(graph, skeleton.shape), graph


def _build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    skeleton = skeleton.astype(bool)
    graph = nx.Graph()

    rows, cols = np.nonzero(skeleton)
    for row, col in zip(rows, cols, strict=True):
        graph.add_node((int(row), int(col)))

    for row, col in list(graph.nodes):
        for d_row, d_col in DIRECTIONS:
            neighbor = (row + d_row, col + d_col)
            if neighbor not in graph or graph.has_edge((row, col), neighbor):
                continue

            length = float(np.hypot(d_row, d_col))
            graph.add_edge(
                (row, col),
                neighbor,
                length=length,
                weight=length,
                path=np.asarray([(row, col), neighbor], dtype=int),
            )

    return graph


def _graph_to_skeleton(graph: nx.Graph, shape: tuple[int, int]) -> np.ndarray:
    skeleton = np.zeros(shape, dtype=bool)

    for row, col in graph.nodes:
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            skeleton[row, col] = True

    for _, _, attrs in graph.edges(data=True):
        path = attrs.get("path")
        if path is None:
            continue

        for row, col in np.asarray(path, dtype=int):
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                skeleton[row, col] = True

    return skeleton


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
    decision: Any,
) -> list[PixelPair]:
    pairs = _pairs_from_best_pairing(neighbors, decision)
    if pairs:
        return pairs

    return _pairs_by_opposite_direction(node, neighbors)


def _pairs_from_best_pairing(
    neighbors: list[PixelNode],
    decision: Any,
) -> list[PixelPair]:
    best_pairing = getattr(decision, "best_pairing", None)
    if not best_pairing or "pairs" not in best_pairing:
        return []

    arm_to_neighbor = _map_arms_to_neighbors(neighbors, decision)

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
    decision: Any,
) -> dict[int, PixelNode]:
    center = np.asarray(decision.center, dtype=float)
    mapping: dict[int, PixelNode] = {}
    used_neighbors: set[PixelNode] = set()

    for arm_index, arm in enumerate(getattr(decision, "arms", [])):
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

        if best_neighbor is not None and best_score > 0.3:
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


def _label_value(label: Any) -> str:
    return getattr(label, "value", label)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=float)

    return vector / norm