from __future__ import annotations

import networkx as nx
import numpy as np

from coronary_analysis.topology.skeleton import DIRECTIONS


PixelNode = tuple[int, int]
PixelPair = tuple[PixelNode, PixelNode]


def build_skeleton_pixel_graph(skeleton: np.ndarray) -> nx.Graph:
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


def pixel_graph_to_skeleton(graph: nx.Graph, shape: tuple[int, int]) -> np.ndarray:
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
