from skimage.morphology import skeletonize

import cv2
import numpy as np


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: int = 15,
) -> np.ndarray:
    pruned = skeleton.copy().astype(bool)

    while True:
        endpoints, _ = classify_skeleton_pixels(pruned.astype(np.uint8))

        if len(endpoints) == 0:
            break

        changed = False

        for r, c in endpoints:
            branch = [(r, c)]
            cur_r, cur_c = r, c

            for _ in range(min_branch_length):
                found = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = cur_r + dr, cur_c + dc

                        if (
                            0 <= nr < pruned.shape[0]
                            and 0 <= nc < pruned.shape[1]
                            and pruned[nr, nc]
                            and (nr, nc) not in branch
                        ):
                            branch.append((nr, nc))
                            cur_r, cur_c = nr, nc
                            found = True
                            break

                    if found:
                        break

                if not found:
                    break

            if len(branch) < min_branch_length:
                for br, bc in branch:
                    pruned[br, bc] = False
                changed = True

        if not changed:
            break

    return pruned.astype(np.uint8)


def classify_skeleton_pixels(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere(filtered == 11)
    junctions = np.argwhere(filtered >= 13)

    return (endpoints, junctions)
