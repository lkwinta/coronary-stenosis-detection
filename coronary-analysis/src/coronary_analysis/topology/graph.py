from skan import Skeleton, summarize

import numpy as np
import pandas as pd


def build_vessel_graph(skeleton: np.ndarray) -> tuple[Skeleton, pd.DataFrame]:
    skel_obj = Skeleton(skeleton.astype(bool))
    branch_data = summarize(skel_obj, separator="_")

    return (skel_obj, branch_data)


def compute_topology_stats(branch_data: pd.DataFrame) -> dict:
    distances = branch_data["branch_distance"]
    euclidean = branch_data["euclidean_distance"].clip(lower=1)

    return {
        "total_vessel_length": distances.sum(),
        "longest_branch": distances.max(),
        "shortest_branch": distances.min(),
        "num_branches": len(branch_data),
        "mean_tortuosity": (distances / euclidean).mean(),
        "branch_type_counts": branch_data["branch_type"].value_counts().to_dict(),
    }
