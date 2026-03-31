import os
import parse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_lm_cad_pairs(image_dir: str) -> list[tuple[str, str]]:
    files_A_dir = os.path.join(image_dir, "A")
    files_B_dir = os.path.join(image_dir, "B")

    files_A = set(os.listdir(files_A_dir))
    files_B = set(os.listdir(files_B_dir))

    pairs = []

    for file_B in files_B:
        if not file_B.startswith("fg"):
            continue

        fila_A_num = parse.parse("fg_{}_{}.jpg", file_B)[0]
        file_A = f"{fila_A_num}.dcm_0.jpg"

        if file_A in files_A:
            pairs.append((file_A, file_B))

    return pairs


class LMCADDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        pairs: list[str] = None,
        transform: A.Compose | None = None,
        target_mode: str = "subtraction",
    ) -> None:
        self.image_dir = image_dir
        if pairs is None:
            self.pairs = get_lm_cad_pairs(image_dir)
        else:
            self.pairs = pairs
        self.transform = transform
        self.target_mode = target_mode

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_A, file_B = self.pairs[idx]

        image_A_path = os.path.join(self.image_dir, "A", file_A)
        image_B_path = os.path.join(self.image_dir, "B", file_B)

        image_A = cv2.imread(image_A_path, cv2.IMREAD_GRAYSCALE)
        image_B = cv2.imread(image_B_path, cv2.IMREAD_GRAYSCALE)

        if image_A is None:
            raise FileNotFoundError(f"Could not load background image: {file_A}")
        if image_B is None:
            raise FileNotFoundError(f"Could not load live image: {file_B}")

        image_A = image_A.astype(np.float32) / 255.0
        image_B = image_B.astype(np.float32) / 255.0

        if self.transform is not None:
            augmented = self.transform(image=image_B, image_b=image_A)
            image_B = augmented["image"]
            image_A = augmented["image_b"]

        if self.target_mode == "subtraction":
            target = np.clip(image_B - image_A, 0.0, 1.0)
        elif self.target_mode == "background":
            target = image_A
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        image_B = np.expand_dims(image_B, axis=0)  # [1, H, W]
        target = np.expand_dims(target, axis=0)  # [1, H, W]

        return torch.from_numpy(image_B).float(), torch.from_numpy(target).float()

    def __len__(self) -> int:
        return len(self.pairs)
